from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
import os
from os import getenv
import PyPDF2
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv
from io import BytesIO
from fastapi import UploadFile
from openai import OpenAI
import json
import requests
from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import euclidean

load_dotenv()
OPENAI_API_KEY = getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def get_job_by_id(id):
    # API URL
    url = f"http://af8f9979b16504831a400599cb562ea7-2055552796.eu-north-1.elb.amazonaws.com:8080/JobPost/getJobById?JobId={id}"

    # Request headers
    headers = {
        'accept': 'text/plain'
    }

    # Making the GET request
    response = requests.get(url, headers=headers)

    try:
        # Parse the response into a Python object (list of dictionaries)
        response_json = json.loads(response.text)

        # Access the first item in the list
        job_description_data = response_json[0]

        # Extract relevant data
        job_description_json = job_description_data["jobDescriptionJson"]
        return job_description_json


    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the response.")
    except KeyError as e:
        print(f"Error: Missing key {e} in the response.")


class explainability_template(BaseModel):
    justification: str
    # confidence_score: int


def score_and_justification(chat):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=chat,
        response_format=explainability_template,
    )

    justification = completion.choices[0].message.parsed.justification
    confidence_score = completion.choices[0].message.parsed.confidence_score
    return justification, confidence_score


class SkillResponse(BaseModel):
    exp: str


# Function to extract skills from the job description using OpenAI
def exp_score(cv_text, id, user_input):
    jd = get_job_by_id(id)
    can_resume = cv_text
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[{"role": "system", "content": f"""
        You are an HR assistant tasked with calculating the Experience Score for a candidate applying for a position at Descon. The Experience Score consists of two components:
        • Work Duration: Scored on a scale of 1 to 5 (with 5 for extensive experience).
        • Relevance of Experience: Scored on a scale of 1 to 5 based on the match between the candidate's past roles and Descon's industry.
        you are provided with below data candidate resume and job description. you should be very strict to calculate the Experience Score.
            you must consider the Job description against the candidate resume. You must be very very strict in calculating the Experience Score
             you will not give any cv a 5 score. even though the candidate have high experience.you should give 5 to a candidate that is ideal for the position.
            as your response should be used in candidate hiring so must be strict
            only return your response in digit/score.
        Use the following inputs:

        Candidate's CV text: {can_resume}
        Job description: {jd}
        Provide the final Experience Score as the average of these two components. """},

                  {"role": "user", "content": user_input}],
        response_format=SkillResponse,  # Use the defined subclass here
    )
    return completion.choices[0].message.parsed.exp


class rel_Response(BaseModel):
    rel_score: str


# Function to extract skills from the job description using OpenAI
def company_rel_score(cv_text, id, user_input):
    jd = get_job_by_id(id)
    can_resume = cv_text
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[{"role": "system", "content": f"""
        You are an expert evaluator tasked with calculating the Company Relevance Score for a candidate applying for a position at Descon.
        for calculating the company relevance score below is two factors you must consider
            • Industry Fit: 1 to 5 based on how closely the past employer's industry aligns with Descon's operations.
             • Company Reputation: 1 to 5 based on market data about the company's standing, innovation, and leadership in its field.
            you are provided with below data candidate resume and job description. you should be very strict to calculate the company relevance score.
            you must consider the Job description against the candidate resume. You must be very very strict in calculating the company relevance score.
            as your response should be used in candidate hiring so must be strict
            your response should be only digit/score.

        Candidate’s CV: {can_resume}
        Job Description: {jd}
         your response should be used in candidate hiring. so make sure to provide the overall company relevence score.       
         your response should be strict along with the factors must consider the Job Description against candidate resume
                . """},

                  {"role": "user", "content": user_input}],
        response_format=rel_Response,  # Use the defined subclass here
    )
    return completion.choices[0].message.parsed.rel_score


class tenure(BaseModel):
    job_tenure: str


# Function to extract skills from the job description using OpenAI
def job_tenure_sc(cv_text, id, user_input):
    jd = get_job_by_id(id)
    can_resume = cv_text
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[{"role": "system", "content": f"""
        You are an HR assistant tasked with calculating the Job Tenure Score for a candidate applying for a position at Descon. The Job Tenure Score consists of two components:
        • Average Tenure: Scored from 1 to 5 based on the average duration of each job, with longer tenures ranked higher.
        • Job-Hopping Frequency: Penalties applied for frequent job changes unless valid reasons are provided.

        you are provided with below data candidate resume and job description. you should be very strict to calculate the Job Tenure.
            you must consider the Job description against the candidate resume. You must be very very strict in calculating the Experience Score
             you will not give any cv a 5 score. even though the candidate have high experience.you should give 5 to a candidate that is ideal for the position.
            as your response should be used in candidate hiring so must be strict
            only return your response in digit/score.
        Use the following inputs:

        Candidate's CV text: {can_resume}
        Job description: {jd}
        Provide the final Experience Score as the average of these two components. """},

                  {"role": "user", "content": user_input}],
        response_format=tenure,  # Use the defined subclass here
    )
    return completion.choices[0].message.parsed.job_tenure


class career(BaseModel):
    career_progression_score: str


# Function to extract skills from the job description using OpenAI
def career_prog_sc(cv_text, id, user_input):
    jd = get_job_by_id(id)
    can_resume = cv_text
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[{"role": "system", "content": f"""
        You are an HR assistant tasked with calculating the career progression Score for a candidate applying for a position at Descon. The career progression Score consists of two components:
        • Continuous Progression: 1 to 5 based on evidence of promotions, role transitions, and increasing responsibilities.
        • Diversity of Roles: Additional points awarded for candidates with diverse experience across different functional areas relevant to Descon's operations.

        you are provided with below data candidate resume and job description. you should be very strict to calculate the career progression Score.
            you must consider the Job description against the candidate resume. You must be very very strict in calculating the career progression Score
             you will not give any cv a 5 score. even though the candidate have high experience.you should give 5 to a candidate that is ideal for the position.
            as your response should be used in candidate hiring so must be strict
            only return your response in digit/score.
        Use the following inputs:

        Candidate's CV text: {can_resume}
        Job description: {jd}
        Provide the final Experience Score as the average of these two components. """},

                  {"role": "user", "content": user_input}],
        response_format=career,  # Use the defined subclass here
    )
    return completion.choices[0].message.parsed.career_progression_score


class reference(BaseModel):
    refer_score: str


# Function to extract skills from the job description using OpenAI
def ref_sc(cv_text, id, user_input):
    jd = get_job_by_id(id)
    can_resume = cv_text
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[{"role": "system", "content": f"""
        You are an HR assistant tasked with calculating the reference Score for a candidate applying for a position at Descon. The reference Score consists of two components:
         • Reference Quality: 1 to 5 based on the credibility of the references (e.g., seniority, relevance to the role).
        • Strength of Endorsements: 1 to 5 based on the strength of the feedback and its relevance to Descon's needs.

        you are provided with below data candidate resume and job description. you should be very strict to calculate the reference Score.
            you must consider the Job description against the candidate resume. You must be very very strict in calculating the career progression Score
             you will not give any cv a 5 score. even though the candidate have high experience.you should give 5 to a candidate that is ideal for the position.
            as your response should be used in candidate hiring so must be strict
            only return your response in digit/score.
        Use the following inputs:

        Candidate's CV text: {can_resume}
        Job description: {jd}
        Provide the final Experience Score as the average of these two components. """},

                  {"role": "user", "content": user_input}],
        response_format=reference,  # Use the defined subclass here
    )
    return completion.choices[0].message.parsed.refer_score


async def extract_data_from_cv(id, cv_file: UploadFile):
    system_message_justification = {"role": "system",
                                    "content": "based on the provided job details and your generated output, provide justification of your output specially the candidate_score, the justification should justify that why."
                                               "the candidate_overallscore is low or high , and a confidence score for the accuracy of your outputs"
                                               "you only consider the candidate_score"}

    chat = []
    chat.append(system_message_justification)
    # Read and extract text from the uploaded CV file
    pdf_file = BytesIO(await cv_file.read())
    cv_text = extract_text_from_pdf(pdf_file)

    # Extract skills from the CV and JD using skill_generator
    # cv_skills = skill_generator(cv_text)  # Convert extracted skills to a set
    # jd_skills = skill_generator(get_job_by_id(id))  # Convert extracted skills to a set
    experience_score = exp_score(cv_text, id,
                                 "Calculate the experience score based on the provided CV text and job description.")
    company_relevance_score = company_rel_score(cv_text, id, "calculate the company relevance score")
    Job_TenureScore = job_tenure_sc(cv_text, id, "calculate the job tenure score")
    Career_progression_score = career_prog_sc(cv_text, id, "calculate the career progression score")
    Reference_score = ref_sc(cv_text, id, "calculate the reference score")
    weight = 5
    candidate_overall_score = (weight * float(experience_score.strip())) + (
                weight * float(company_relevance_score.strip())) + (weight * float(Job_TenureScore.strip())) + (
                                          weight * float(Career_progression_score.strip())) + (
                                          weight * float(Reference_score.strip()))
    new_candidate_overall_score = (candidate_overall_score / 125) * 10

    response = {
        "Experience_score": (experience_score),
        "Company_relevance_score": (company_relevance_score),
        "Job_Tenure_score": (Job_TenureScore),
        "Career_progression_score": (Career_progression_score),
        "Reference_score": (Reference_score),
        "candidate_score": round(new_candidate_overall_score, 2),

    }

    chat.append({"role": "user", "content": f"the following is the job details {get_job_by_id(id)}"})
    chat.append({"role": "assistant", "content": f"please give me the relevant resume text of candidate"})
    chat.append({"role": "user", "content": f"sure! here is the CV text {cv_text}"})
    chat.append(
        {"role": "assistant", "content": f"please have the details generated out of your provided context {response}"})

    justification, confidence_score = score_and_justification(chat)

    # Update the response with serialized values
    response.update({

        # "CV_Content": cv_text,
        "Justification": justification,
        "Confidence_Score": confidence_score,
    })

    return response


# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file: BytesIO) -> str:
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text.strip()
