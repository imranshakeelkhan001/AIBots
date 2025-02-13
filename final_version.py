import os
import requests
from openai import OpenAI
from generate_token import token,update_env_key
import json
from dotenv import load_dotenv
load_dotenv()


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
access_token = os.getenv('access_token')

# OpenAI API key
client = OpenAI()

def get_skills_for_job_title(job_title, access_token):
    url = "https://emsiservices.com/skills/versions/latest/skills"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    params = {
        "q": job_title,
        "typeIds": "ST1,ST2,ST3",  # Technical and common skills
        "fields": "id,name,type",
        "limit": "10"  # Adjust the limit as needed
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()


# Function to get related skills using skill IDs
def get_related_skills(skill_ids, access_token):
    url = "https://emsiservices.com/skills/versions/latest/related"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "ids": skill_ids
    }
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()

def skills_extract(job_title, access_token):
    Req_job_title = job_title  # Replace with your desired job title

    try:
        # Step 1: Get skills for the job title
        skills_data = get_skills_for_job_title(job_title, access_token)
        skills = skills_data.get("data", [])

        if not skills:
            return {"error": f"No skills found for job title '{Req_job_title}'."}

        # Extract skill IDs from the skills data
        skill_ids = [skill["id"] for skill in skills]

        # Step 2: Get related skills using skill IDs
        related_skills_data = get_related_skills(skill_ids, access_token)
        related_skills = related_skills_data.get("data", [])

        if not related_skills:
            return {"error": "No related skills found."}

        # Prepare related skills for return
        result = []
        for skill in related_skills:
            skill_name = skill.get("name")
            skill_type = skill.get("type", {}).get("name")
            skill_url = skill.get("infoUrl")
            result.append({
                "name": skill_name,
                "type": skill_type,
                "url": skill_url
            })

        return result

    except requests.exceptions.HTTPError as http_err:
        return {"error": f"HTTP error occurred: {http_err}"}
    except Exception as err:
        return {"error": f"An error occurred: {err}"}


# Tools definition with parameters for the `skills_extract` function
tools = [
    {
        "type": "function",
        "function": {
            "name": "skills_extract",
            "description": "Extract job-specific skills from a query using the Lightcast API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_title": {
                        "type": "string",
                        "description": "this is the job role or job title of the candidate to be hired. This "
                                       "paramater is compulsory for the function. this should not be python dictionary form"
                    },
                },
                "required": ["job_title"]
            }
        }
    }
]

# Initial system message
message = [
    {
        "role": "system",
        "content": (
            "You are a helpful DESCON HR assistant.Start conversation with greetings by using your name ' I am Descon HR Assistant'.your response should be conversational.'do not ask are you looking for a job description for a specific role'."
            "you will greet on every response provided by user with connecting words. you will not ask multiple questions from user."
            "Your job is to provide the Job Description.For this purpose, "
            "you will get the details from the user by asking 3 ~4 questions if needed, each question should be engaging and connected, and greet user by receving the information for each question and based on these details "
            ", you will extract the skills that are specific for the desired job using the function 'skills_extract'. Make sure to call the function "
            "when required before generating job desciption because the extracted skills will be used in Job description. Do not ask the user multiple questions in one response,and make sure do not mentioned that you are extracting the skills, it is not allowed to tell user about it."
            " and provide a complete job description when you have extracted the skills.please note: 'only provide job description do not write the message like 'Certainly! Here's a professional job description for an Electrical Engineer position at DESCON, tailored for a LinkedIn job post:' After providing the job description. "
            "User may ask you for some changes, try to make these changes specific only in that job description that you just provided. Note:  'Do not provide new job description at any cost' "
            "unless the user want any other category job description that user asked and provide the desired output that user want."
            "As you are a DESCON HR Assistant, of a company DESCON, so include company name of DESCON in job description. as you are creating job description for your comoany."
        )
    }
]

# Conversational bot logic
def conversational_Bot(message):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=message,
        tools=tools,
        temperature=0.2

    )
    return completion




while True:
    query = input("enter a query: ")
    new_rol = {"role":"user", "content": query}
    message.append(new_rol)
    #print(message)
    response = conversational_Bot(message)
    content = response.choices[0].message.content
    #print(content)

    # print(response.choices[0].message.content)
    # print(response.choices[0].message.tool_calls)
    if response.choices[0].message.content:
        #print(response.choices[0].message.content)
        message.append({"role":"assistant","content":response.choices[0].message.content})
    elif response.choices[0].message.tool_calls:
        # print(response.choices[0].message.tool_calls[0].function.name)
        if str(response.choices[0].message.tool_calls[0].function.name) == "skills_extract":
             job_title= json.loads(response.choices[0].message.tool_calls[0].function.arguments)["job_title"]
             skill_output=skills_extract(job_title,access_token)
             print(skill_output)
             message.append({"role":"assistant","content":f" this is the output of tool call{skill_output}"})
             message.append({"role":"user","content":"provide me professional job description based on the skills got from above tools call (donot include links) provided for linkedin job post. In skills section of your job description use the above provided skills"})
             #print(message)
             response=conversational_Bot(message)
    print(response.choices[0].message.content)