import os
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import requests
from io import BytesIO
from controllers.resume_similarity import extract_data_from_cv
from pydantic import BaseModel
import requests
import json
import pandas as pd
from controllers.new_azm import score_and_justification, exp_score, company_rel_score, job_tenure_sc, career_prog_sc, \
    ref_sc, get_job_by_id

load_dotenv()
client = OpenAI()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
EXTERNAL_API_URL = "http://103.18.20.205:8090/resume_similarity"

import base64
from fastapi import UploadFile

import base64
from fastapi import UploadFile
from io import BytesIO

import base64
from fastapi import UploadFile
from io import BytesIO


async def pdf_to_base64(file: UploadFile):
    """
    Convert the uploaded PDF file (FastAPI UploadFile) to a Base64-encoded string.

    :param file: The uploaded file object.
    :return: Base64 encoded string.
    """
    print("Inside function")

    try:
        if not file:
            print("No file received")
            return None

        print(f"Received file: {file.filename}")

        # Reset file pointer to beginning (important!)
        file.file.seek(0)

        # Read file content as bytes
        file_content = file.file.read()

        # Check if file is empty
        if not file_content:
            print("Error: File is empty after reading!")
            return None

        # Convert file to Base64
        encoded_string = base64.b64encode(file_content).decode("utf-8")

        print(f"Base64 encoded length: {len(encoded_string)}")

        return encoded_string

    except Exception as e:
        print(f"Exception occurred: {e}")
        return None


def ad_resume(email, file):
    url = "http://af8f9979b16504831a400599cb562ea7-2055552796.eu-north-1.elb.amazonaws.com:8080/JobPost/addResumeFileBytes"

    payload = json.dumps({
        "email": str(email),
        "fileBase64": f"{file}"
    })
    headers = {
        'accept': 'text/plain',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)
    print("API HIT jee", response.text)


def store_chat_history(chat_messages, chat_id, csv_file="applicant_chat_history.csv"):
    # Check if the CSV file exists
    if os.path.exists(csv_file):
        # Load existing chat history
        chat_df = pd.read_csv(csv_file)
    else:
        # Create a new DataFrame if the file does not exist
        chat_df = pd.DataFrame(columns=["id", "applicant_chat_history"])

    # Check if the ID exists in the DataFrame
    if chat_id in chat_df["id"].values:
        # Update the existing row
        chat_df.loc[chat_df["id"] == chat_id, "applicant_chat_history"] = str(chat_messages)
    else:

        # Add a new row
        new_row = {"id": chat_id, "applicant_chat_history": str(chat_messages)}
        chat_df = pd.concat([chat_df, pd.DataFrame([new_row])], ignore_index=True)

    # Save the updated DataFrame back to the CSV file
    chat_df.to_csv(csv_file, index=False)
    print(f"Chat history for ID {chat_id} has been saved to {csv_file}.")


def get_all_job_posts():
    url = "http://af8f9979b16504831a400599cb562ea7-2055552796.eu-north-1.elb.amazonaws.com:8080/JobPost/getAllJobs"

    payload = {}
    headers = {
        'accept': 'text/plain'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return response.text


def get_jdid(title):
    # API URL
    url = "http://af8f9979b16504831a400599cb562ea7-2055552796.eu-north-1.elb.amazonaws.com:8080/JobPost/getAllJobs"

    # Request headers
    headers = {
        'accept': 'text/plain'
    }

    # Make the API request
    response = requests.get(url, headers=headers)

    # Parse the response as JSON
    data = response.json()

    # Extract jdId and title as key-value pairs
    jdid_title_dict = {item['title']: item['jdId'] for item in data}

    jd_id = jdid_title_dict.get(title)
    return jd_id


def email(fullname, emailaddress, phone_number, skills, resumejson, jdid, rank, rankReason):
    url = "http://af8f9979b16504831a400599cb562ea7-2055552796.eu-north-1.elb.amazonaws.com:8080/JobPost/addResume"
    print("Ranked score at the scale of 100 is ", str(rank))
    payload = json.dumps({
        "fullName": str(fullname),
        "emailAddress": str(emailaddress),
        "contactNumber": str(phone_number),
        "skills": skills,
        "resumeJSON": str(resumejson),
        "jdId": str(jdid),
        "rank": str(rank),
        "rankReason": str(rankReason),

    })
    headers = {
        'accept': 'text/plain',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    print("api hit was successfule and data has been saved to your db")
    return response.text


#####################################Azeemcode#######################################

def addresume(file, email: str):
    url = f"http://af8f9979b16504831a400599cb562ea7-2055552796.eu-north-1.elb.amazonaws.com:8080/Candidate/addResumeFile?EmailAddress={email}"

    headers = {'accept': 'text/plain'}

    # Read file content correctly
    with file.file as f:
        files = {
            'File': (file.filename, f.read(), file.content_type)  # Correct format
        }

        response = requests.post(url, headers=headers, files=files)

    print("CV uploaded successfully")
    print(response.text)


tools = [
    {
        "type": "function",
        "function": {
            "name": "email",
            "description": "Anticipate the details from chat. if not found in chat then ask from user",
            "parameters": {
                "type": "object",
                "properties": {
                    "fullname": {
                        "type": "string",
                        "description": "The name of the candidate. Ask from user if not parsed from user's resume"
                    },
                    "emailaddress": {
                        "type": "string",
                        "description": "The email of the candidate. Ask from user if not parsed from user's resume"
                    },
                    "phone_number": {  # Corrected from phone_numbeer
                        "type": "string",  # Changed 'string' to "string"
                        "description": "The contact number of the candidate. Ask from user if not parsed from user's resume"
                    },

                    "job_title": {
                        "type": "string",  # Changed from int to "integer"
                        "description": "The title of the job he is applying for, examples are 'AI Engineer', 'React Developer'. Ask from user if not found in chat"
                    },

                },
                "required": [
                    "fullname", "emailaddress", "phone_number",
                    "job_title"
                ]
            }
        }
    }
]


def list_of_all_jobs():
    url = "http://af8f9979b16504831a400599cb562ea7-2055552796.eu-north-1.elb.amazonaws.com:8080/JobPost/getAllJobs"

    headers = {
        'accept': 'text/plain'
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        jobs = response.json()  # Parse JSON response
        job_titles = [job["title"] for job in jobs]  # Extract job titles
        print("here is jobs", job_titles)
        return job_titles
        # Print each job title on a new line
        # for title in job_titles:
        #     print("here is jobs",title)
    else:
        print("Failed to fetch jobs:", response.status_code)


system_message = {
    "role": "system",
    "content": (f"""You are an AI assistant for DESCON Career Portal. Your task is to assist users in the job application process in a step-by-step conversational manner.Do not ask multiple questions in 1 response. always greet user when you receive response from user.
         for the step 4 additional questions 'do not ask user multiple questions in 1 response. ask user question one by one'.
         Follow the provided example template for structuring your responses. Always maintain a professional, friendly, and empathetic tone.

you are equipped with a tools. please make sure to call the tool 'email' once you get all the details from the chat for this tool.
Here is an example of how you should interact with the user:

---

**Example Interaction**:

**Step 1: Discovering the Job Posting**  
Aysha comes across the job posting on LinkedIn and clicks on the link, which directs her to the Beacon House Career Site. She might ask the ai assistant that how many jobs are available, then the ai assistant will guide her for the relevant job for her.

**Step 2: Initiating the Application Process**  
AI Assistant: "Welcome to DESCON Career Portal. How may I assist you today?"  
Aysha: "I would like to apply for a job."
AI Assistant: sure. can you please tell which job you are interested?
Ayesha: "I want to apply for HR job. is there any job available for this position.
AI Assistant: "yes we have HR job opening. here is relevant job opening. Please note: here you will show the relevant jobs."
Ayesha: I would like to apply for this job.
AI Assistant: "Thank you for your interest in joining our team. May I have your full name to address you?"  
Aysha: "My name is Aysha Khan"  
AI Assistant: "Thank you, Ms. Khan. Your profile has been created. Shall we proceed with your application?"  
Aysha: "Yes, please."
AI Assistant: To begin, please let me know which position you are interested in applying for.
ayesha: i want apply for maths teacher
**Step 3: Collecting Application Information**  
AI Assistant: "To begin, please upload your resume or curriculum vitae."  
Aysha uploads her resume.  
AI Assistant: "Thank you. I am now scanning your resume for relevant information."  
(The AI Assistant parses the resume to extract pertinent data.)  
AI Assistant: "I have extracted the following details:  
- Name: Aysha Khan  
- Email: aysha.khan@example.com  
- Phone Number: (555) 987-6543  

Does this information appear correct?"  
Aysha: "Yes, that's correct."
**Step 4: Confirming Application Submission**  
AI Assistant: "Your application is now complete. Would you like to review it before submission? please note must use '?' at the end of this question"  
Aysha: "Yes, I would appreciate that."  
(AI Assistant displays a summary of the application.)  
Aysha: "Everything looks in order."  
**ai assistant calls the email tool**

Please note: you will only show complete list of jobs if user asked. otherwise you will only show the user relevant jobs.
---

you are equipped with a tools. please make sure to call the tool 'email' once you get all the details from the chat for this tool.

Follow this structure to handle new user interactions. For each new query, guide the user through the job application process using clear, concise, and conversational language.
Note: Don't ask the user for email address and phone number when it is already in the chat data. Instead of asking irrelevant questions when you think the chat is complete then call the email tool.

Please Note: You will not entertain any other query other than above.e.g if user ask what is ai etc you will not respond to that. you are specific to this role that you only entertain the applicant.

""")
}


# Conversational bot logic
def conversational_Bot(message):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=message,
        tools=tools,
        temperature=0.2

    )
    return completion


def get_chat_by_id(chat_id, csv_file="applicant_chat_history.csv"):
    try:
        # Load the chat history from the CSV file
        history = pd.read_csv(csv_file)

        # Check if the ID exists in the DataFrame
        if chat_id in history["id"].values:
            # Retrieve the corresponding chat history
            chat_history_str = history.loc[history["id"] == chat_id, "applicant_chat_history"].values[0]

            # Convert the string representation of the chat history back to a list
            chat_history = eval(chat_history_str)

            # Yield individual messages as separate dictionaries
            for message in chat_history:
                yield message
        else:
            print(f"Chat with ID {chat_id} not found.")
            return None
    except FileNotFoundError:
        print(f"File {csv_file} not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def get_cv_text_by_id(csv_file, id):
    try:
        # Load the CSV file into a DataFrame
        cv_data = pd.read_csv(csv_file,on_bad_lines='skip')

        # Filter the DataFrame to find the row with the given ID
        result = cv_data[cv_data['id'] == (id)]

        # Check if the ID exists in the DataFrame
        if not result.empty:
            # Return the CV text for the matching ID
            return result.iloc[0]['data']
        else:
            return f"No CV text found for ID: {id}"
    except Exception as e:
        return f"An error occurred: {e}"


class rank_bo(BaseModel):
    rank_re: list[str]


# Function to extract skills from the job description using OpenAI
def skills_candidate(cv_text):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[{"role": "system",
                   "content": f""" you are given with the resume data of candidate extract the skills of user """},

                  {"role": "user", "content": cv_text}],
        response_format=rank_bo,  # Use the defined subclass here
    )
    return completion.choices[0].message.parsed.rank_re

class job_title_extract(BaseModel):
    job_title: str


# Function to extract skills from the job description using OpenAI
def title(chat):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[{"role": "system",
                   "content": f""" you are given with the chat. extract the job title """},

                  {"role": "user", "content": chat}],
        response_format=job_title_extract,  # Use the defined subclass here
    )
    print("here is job title",completion)
    return completion.choices[0].message.parsed.job_title


class justify(BaseModel):
    just: str


def score_jus(cv_text, id, rank):
    jd = get_job_by_id(str(id))
    all_score = score_all(cv_text, id)
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[{"role": "system", "content": f""" 
         you are given with the resume data of candidate and Job description {str(jd)}. here is the candidate rank score {rank}
You are an AI Hiring Evaluator. Your task is to assess a candidate’s rank score based on their resume and the job description (JD), following a structured **Chain-of-Thought** approach.  

### **Evaluation Criteria (Scored as Percentage)**  
1️⃣ **Experience Score** – Relevance of past roles & skills to JD.  
2️⃣ **Company Relevance Score** – Alignment with hiring company’s industry & standards.  
3️⃣ **Job Tenure Score** – Stability & commitment in past roles.  
4️⃣ **Career Progression Score** – Growth in responsibility & leadership.  
5️⃣ **Reference Score** – Credibility of endorsements & industry reputation.  

---

### **Evaluation Process (Step-by-Step Analysis)**  
🔹 **Step 1: Understanding the Role**  
- Identify **key JD requirements** (skills, experience, industry fit).  
- Assess candidate’s **resume for alignment** with JD.  

🔹 **Step 2: Breakdown of Scores**  
- Each category is **scored as a percentage** (e.g., **40% Experience Score** instead of 2/5).  
- Highlight **strengths & gaps** in each factor.  

🔹 **Step 3: Candidate vs. JD Comparison**  
- Compare **skills, experience, and industry alignment** with role needs.  
- Identify **missing skills, certifications, or experience gaps**.  

🔹 **Step 4: Identifying Mismatches**  
- Determine if **skills are transferable or insufficient**.  
- Address **potential concerns (e.g., short tenures, industry fit, leadership gaps).**  

🔹 **Step 5: Final Decision & Justification**  
✅ **High Score (80%+ Fit):** Strong alignment with JD, solid tenure, relevant skills & certifications.  
🚫 **Low Score (<40% Fit):** Mismatch in industry, missing key skills/certifications, unstable job history.  

**Example Justification:**  
*"Candidate scored **40%**, showing a major gap in HR expertise, industry fit, and compliance knowledge. Their AI engineering background does not align with the Senior HR Manager role. Lacks key HR certifications and leadership experience. **Not recommended for this position.**"*  

---

### **Your Role as an AI Hiring Evaluator**  
✔ **Strict & Job-Focused:** Evaluate only on **JD relevance**.  
✔ **Concise & Data-Driven:** Provide **score breakdown & justification**.  
✔ **Professional & Objective:** Base decisions on **facts, not assumptions**.  

---
🚀 **Ensure a clear, structured, and concise evaluation for accurate hiring decisions.**  


"""},

                  {"role": "user", "content": f"here is {cv_text} and all scores  of the candidate {all_score}"}],
        response_format=justify,  # Use the defined subclass here
    )
    print("here is reason", completion.choices[0].message.parsed.just)
    return completion.choices[0].message.parsed.just


class TotalScore(BaseModel):
    experience: str
    company_relevance: str
    job_tenure: str
    career_progression: str
    reference: str


def score_all(cv_text, id):
    jd = get_job_by_id(id)
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[{"role": "system", "content": f"""

You are an HR assistant responsible for calculating various candidate evaluation scores. Be extremely strict in assessment, as your response is used in hiring decisions. No candidate should receive a perfect score (5) unless they are an ideal fit.

Scoring Categories:

    experience: score(1-5)
        Work Duration: Length of experience.
        Relevance of Experience: Alignment with Descon’s industry.

    company_relevance: score(1-5)
        Industry Fit: Past employer's alignment with Descon.
        Company Reputation: Market standing, innovation, and leadership.

    job_tenure: Score (1-5)
        Average Tenure: Duration of previous jobs.
        Job-Hopping Frequency: Penalties for frequent changes unless justified.

    career_progression: score (1-5)
        Continuous Progression: Promotions and increasing responsibilities.
        Diversity of Roles: Varied experience across relevant functions.

    reference: Score (1-5)
        Reference Quality: Credibility, seniority, and relevance.
        Strength of Endorsements: Relevance and quality of recommendations.

Evaluation Process:

    Assess the candidate’s resume strictly against the job description {jd}.
    Only return a single-digit score (1-5) for each category.
    Ensure fairness and rigor in scoring.

Inputs:

    Job Description: {jd}
    Candidate Resume: Provided for evaluation.

Final Output: A strict and unbiased score for each category based on the provided inputs. """},

                  {"role": "user", "content": cv_text}],
        response_format=TotalScore,  # Use the defined subclass here
    )
    return {
        "experience": completion.choices[0].message.parsed.experience,
        "company_relevance": completion.choices[0].message.parsed.company_relevance,
        "job_tenure": completion.choices[0].message.parsed.job_tenure,
        "career_progression": completion.choices[0].message.parsed.career_progression,
        "reference": completion.choices[0].message.parsed.reference
    }


def all_func(cv_text, id):
    # Get individual scores using `score_all`
    scores = score_all(cv_text, id)

    experience_score = float(scores["experience"].strip())
    company_relevance_score = float(scores["company_relevance"].strip())
    job_tenure_score = float(scores["job_tenure"].strip())
    career_progression_score = float(scores["career_progression"].strip())
    reference_score = float(scores["reference"].strip())

    weight = 5  # Assigning equal weight to each score

    # Compute the overall weighted score
    candidate_overall_score = (weight * experience_score) + (weight * company_relevance_score) + (
                weight * job_tenure_score) + (weight * career_progression_score) + (weight * reference_score)

    # Normalize the score to a 10-point scale
    new_candidate_overall_score = (candidate_overall_score / 125) * 10

    print(
        f"Experience: {experience_score}, Company Relevance: {company_relevance_score}, Job Tenure: {job_tenure_score}, Career Progression: {career_progression_score}, Reference: {reference_score}")
    print(f"Final Candidate Score: {new_candidate_overall_score}")

    return new_candidate_overall_score


# def all_func(cv_text,id):
#     experience_score = exp_score(cv_text, id,"Calculate the experience score based on the provided CV text and job description.")
#     company_relevance_score = company_rel_score(cv_text, id, "calculate the company relevance score")
#     Job_TenureScore = job_tenure_sc(cv_text, id, "calculate the job tenure score")
#     Career_progression_score = career_prog_sc(cv_text, id, "calculate the career progression score")
#     Reference_score = ref_sc(cv_text, id, "calculate the reference score")
#     weight = 5
#     candidate_overall_score = (weight * float(experience_score.strip())) + (
#             weight * float(company_relevance_score.strip())) + (weight * float(Job_TenureScore.strip())) + (
#                                       weight * float(Career_progression_score.strip())) + (
#                                       weight * float(Reference_score.strip()))
#     new_candidate_overall_score = (candidate_overall_score / 125) * 10
#     print(experience_score,company_relevance_score,Job_TenureScore,Career_progression_score,Reference_score)
#     print(new_candidate_overall_score)
#     return new_candidate_overall_score


async def applicant_bot(query, id, file):
    message = []
    new_jobs = {"role": "assistant", "content": f"here is list of opening  jobs {str(list_of_all_jobs())}"}
    # print(new_jobs)
    message.append(new_jobs)
    history = pd.read_csv("applicant_chat_history.csv")
    #print("here is history",history)
    columns = history['id']
    if file:
        if int(id) in list(columns):
            print("Processing CV")
            data = await extract_data_from_cv(file)
            file_64 = await pdf_to_base64(file)
            chat_history = get_chat_by_id(int(id), csv_file="applicant_chat_history.csv")
            #print("here is chat history",chat_history)
            for msg in chat_history:
                message.append(msg)
            new_job_title = title(str(message))
            #print("the job title is ",new_job_title)

            new_cvtext = get_cv_text_by_id("cv_text.csv", str(id))
            rank = all_func(new_cvtext, str(get_jdid(new_job_title)))
            skills = skills_candidate(new_cvtext)
            justification = score_jus(new_cvtext, str(get_jdid(new_job_title)), int(rank) * 10)
            #job_title = str(get_jdid(str(title(message)['job_title'])))
            email(
                {data['full_name']},
                {data['email_address']},
                {data['contact_number']},
                skills,  # skills
                "resume in json",
                new_job_title,
                #str(get_jdid(str(['job_title']))),
                int(rank) * 10,  ## candidate score
                str(justification)  # justification

            )
            # print("here is base64",file_64)
            ad_resume(data['email_address'],file_64)
            #print("here is email", data['email_address'])
            #addresume(file, data['email_address'])
           # print("name is ", data['full_name'])
            # Load the existing CSV file into a DataFrame
            cv_data = pd.read_csv("cv_text.csv",on_bad_lines='skip')
            # Define the new row to be added
            new_row = {"id": id, "data": data['resume_json']}
            # Concatenate the new row to the existing DataFrame
            updated_data = pd.concat([cv_data, pd.DataFrame([new_row])], ignore_index=True)
            # Save the updated DataFrame back to the CSV file
            updated_data.to_csv("cv_text.csv", index=False)
            chat_history = get_chat_by_id(int(id), csv_file="applicant_chat_history.csv")
            for msg in chat_history:
                message.append(msg)
            new_rol = {"role": "assistant",
                       "content": f"please confirm the details i extracted from you CV. Name: {data['full_name']} Email:{data['email_address']}, : Contact Number: {data['contact_number']}, Your Skills: {data['skills']}, Education: {data['education']}, Experience: {data['experience']} "}

            message.append(new_rol)
            store_chat_history(message, id)
            # data  = pd.read_csv("cv_text.csv")
            new_rol = {"role": "assistant",
                       "content": f"please confirm the details i extracted from you CV. Name: {data['full_name']} Email:{data['email_address']}, : Contact Number: {data['contact_number']}, Your Skills: {data['skills']}, Education: {data['education']},  Experience: {data['experience']}"}
            print("new role",new_rol['content'])
            return new_rol['content']

        elif not int(id) in list(columns):
            print("Processing CV in elif not int(id) ")
            data = await extract_data_from_cv(file)
            file_64 = await pdf_to_base64(file)
            # print("here is base64",file_64)
            ad_resume(data['email_address'], file_64)
            cv_data = pd.read_csv("cv_text.csv",on_bad_lines='skip')
            # Define the new row to be added
            new_row = {"id": id, "data": data['resume_json']}
            # Concatenate the new row to the existing DataFrame
            updated_data = pd.concat([cv_data, pd.DataFrame([new_row])], ignore_index=True)
            # Save the updated DataFrame back to the CSV file
            updated_data.to_csv("cv_text.csv", index=False)
            message.append(system_message)
            new_rol = {"role": "assistant",
                       "content": f"please confirm the details i extracted from you CV. Name: {data['full_name']} Email:{data['email_address']}, : Contact Number: {data['contact_number']}, Your Skills: {data['skills']}, Education: {data['education']},  Experience: {data['experience']}"}
            message.append(new_rol)
            store_chat_history(message, id)
            new_rol = {"role": "assistant",
                       "content": f"please confirm the details i extracted from you CV. Name: {data['full_name']} Email:{data['email_address']}, : Contact Number: {data['contact_number']}, Your Skills: {data['skills']}, Education: {data['education']},  Experience: {data['experience']}"}
            return new_rol['content']

    elif not file:
        if int(id) in list(columns):
            print("inside if")
            # print("inside if")

            chat_history = get_chat_by_id(int(id), csv_file="applicant_chat_history.csv")
            for msg in chat_history:
                message.append(msg)
            new_rol = {"role": "user", "content": query}
            message.append(new_rol)
        elif not int(id) in list(columns):
            print("inside elif")
            message.append(system_message)
            new_rol = {"role": "user", "content": query}
            message.append(new_rol)

        response = conversational_Bot(message)
        content = response.choices[0].message.content
        if response.choices[0].message.content:
            print(response.choices[0].message.content)
            message.append({"role": "assistant", "content": response.choices[0].message.content})
           # print(message)
            store_chat_history(message, id)  # Save chat history in the `if` block
            return {"chat": response.choices[0].message.content}
        elif str(response.choices[0].message.tool_calls[0].function.name) == 'email':

            store_chat_history(message, id)  # Save chat history in the `elif` block
            return {
                "chat": "Your application has been successfully submitted! Thank you for applying. We will review your application and contact you soon."}




            # tool_call_arguments = response.choices[0].message.tool_calls[0].function.arguments
            # result_argument = json.loads(tool_call_arguments)
            # print("here is result arguments-------", tool_call_arguments)
            # new_cvtext = get_cv_text_by_id("cv_text.csv", str(id))
            # # print("newcv test is ",new_cvtext)
            # rank = all_func(new_cvtext, str(get_jdid(str(result_argument['job_title']))))
            # skills = skills_candidate(new_cvtext)
            # print("extracted skills are ", skills)
            # print("the data type is ", type(skills))
            # justification = score_jus(new_cvtext, str(get_jdid(str(result_argument['job_title']))), int(rank) * 10)
            # # Call the email function with proper arguments
            # email(
            #     result_argument['fullname'],
            #     result_argument['emailaddress'],
            #     result_argument['phone_number'],
            #     skills,  # skills
            #     "resume in json",
            #     str(get_jdid(str(result_argument['job_title']))),
            #     int(rank) * 10,  ## candidate score
            #     str(justification)  # justification
            #
            # )
