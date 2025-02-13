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
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
from pydantic import BaseModel
from scipy.spatial.distance import euclidean


load_dotenv()
OPENAI_API_KEY = getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class explainability_template(BaseModel):

    justification: str
    confidence_score: int


def score_and_justification(chat):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=chat,
        response_format=explainability_template,
    )

    justification = completion.choices[0].message.parsed.justification
    confidence_score = completion.choices[0].message.parsed.confidence_score
    return justification, confidence_score


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


def euclidean_distance_bert(str1, str2):
    """
    Calculate the Euclidean distance between two strings using BERT embeddings.
    Args:
        str1 (str): First input string.
        str2 (str): Second input string.
    Returns:
        float: Euclidean distance between the embeddings of the two strings.
    """
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Tokenize the input strings and get their embeddings
    inputs1 = tokenizer(str1, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs2 = tokenizer(str2, return_tensors='pt', padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        # Get embeddings for each input string
        embeddings1 = model(**inputs1).last_hidden_state.mean(dim=1).squeeze(0)
        embeddings2 = model(**inputs2).last_hidden_state.mean(dim=1).squeeze(0)

    # Compute Euclidean distance between the two sentence embeddings
    distance = euclidean(embeddings1.numpy(), embeddings2.numpy())
    return distance



class jd_skills(BaseModel):
    skills: list[str]

def skill_generator(user_input):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[{"role":"system","content":"you are provided with the job description. just extract the skills from the job description"},
                  {"role": "user", "content": f"extract the data from {user_input}"}],
        response_format=jd_skills,
    )

    return str(completion.choices[0].message.parsed.skills)


def jaccard_similarity_skills(skills1, skills2):
    # Convert all skills to lowercase to make the comparison case-insensitive
    skills1 = [skill.lower() for skill in skills1]
    skills2 = [skill.lower() for skill in skills2]

    # Get the union of both skill sets
    all_skills = list(set(skills1).union(set(skills2)))

    # Create binary vectors for both skill lists based on the union of skills
    vector1 = [1 if skill in skills1 else 0 for skill in all_skills]
    vector2 = [1 if skill in skills2 else 0 for skill in all_skills]

    # Calculate Jaccard similarity score using sklearn's jaccard_score function
    similarity_score = jaccard_score(vector1, vector2)

    # Calculate matched skills (intersection)
    matched_skills = list(set(skills1).intersection(set(skills2)))

    # Return the Jaccard similarity score and matched skills
    return similarity_score * 100, matched_skills


def cosine_similarity_sklearn(str1, str2):
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Tokenize the input strings and get their embeddings
    inputs1 = tokenizer(str1, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs2 = tokenizer(str2, return_tensors='pt', padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        # Get embeddings for each input string
        embeddings1 = model(**inputs1).last_hidden_state.mean(dim=1)
        embeddings2 = model(**inputs2).last_hidden_state.mean(dim=1)

    # Compute cosine similarity between the two sentence embeddings
    similarity = cosine_similarity(embeddings1.numpy(), embeddings2.numpy())
    return similarity[0][0]


def extract_text_from_pdf(pdf_file: BytesIO) -> str:
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text.strip()

# Data structure for storing parsed profile data
class ProfileData(BaseModel):
    full_name: str
    email_address: str
    contact_number: int
    skills: list[str]
    education: str
    experience: str

class ski_match(BaseModel):
    match_skill: list[str]
    number_of_skill: int


def getmatch(generated_skills,skill_cv):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "extract the matched skills and number_of_matched_skills based on provided candidate cv skills and job description skills."},
            {"role": "user", "content": f"extract the data from {generated_skills},{skill_cv}"},
        ],
        response_format=ski_match,
    )


    matched_skills = completion.choices[0].message.parsed.match_skill
    number_of_skills = completion.choices[0].message.parsed.number_of_skill
    return matched_skills, number_of_skills

async def extract_data_from_cv(cv_file: UploadFile):
    system_message_justification = {"role":"system", "content":"based on the provided job details and your generated output, provide justification of your output specially the rank value, the justification should justify that why."
                                                               "the candidate's rank score is low or high , and a confidence score for the accuracy of your outputs"}
    chat = []
    chat.append(system_message_justification)

    # Read and extract text from the uploaded PDF
    pdf_file = BytesIO(await cv_file.read())  # Read the uploaded file into memory
    cv_text = extract_text_from_pdf(pdf_file)  # Extract text from the PDF

    # OpenAI API call
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"You are given a candidate's CV text"
                                          f"Your task is to extract following information of the candidate"
                                          f"1. Name"
                                          f"2. Email"
                                          f"3. contact_number"
                                          f"4. skills"
                                          f"5 education"
                                          f"6 total experience"
                                          f"If any of the above data is not found in the given CV text then leave it as None"},
            {"role": "user", "content": cv_text},
        ],
        response_format=ProfileData
    )

    # Parse the response to extract skills
    try:
        response = dict(response.choices[0].message.parsed)
        skill_cv = response['skills']

        matched_skills = []
        response['skil_score'] = 0
        rank = 0
        # rank = cosine_similarity_sklearn(str(cv_text),str(jd))

        response['rank'] = (float(rank)*100)*(response['skil_score']*1)
        response['matched_skills'] = matched_skills
        response['resume_json'] = cv_text
        chat.append({"role":"user", "content":f"the following is the job details "})
        chat.append({"role":"assistant", "content":f"please give me the resume text of candidate"})
        chat.append({"role": "user", "content": f"sure! here is the resume text {cv_text}"})
        chat.append({"role": "assistant", "content": f"please have the details generated out of your provided context {response}"})
        justification, confidence_score = score_and_justification(chat)
        response['rankReason'] = justification
        response['confidence_score'] = confidence_score
        print("cosine sim", rank)
        print(response['skil_score']/1)
        print("match skills",matched_skills)


        return response
    except Exception as e:
        return ValueError(f"Error extracting skills: {e}")
