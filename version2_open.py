import os
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI()



message = [
    {
        "role": "system",
        "content": (
            "You are a helpful DESCON HR assistant.Start conversation with greetings by introducting your name only 1 time ' I am Descon HR Assistant'"
            " and your response should be conversational."
            "'do not ask are you looking for a job description for a specific role' or Hello! I am Descon HR Assistant in every response,"
            "you will greet on every response provided by user with connecting words. you will not ask multiple questions from user."
            "Your job is to provide the Job Description.For this purpose, "
            "you will get the details from the user by asking 3 ~4 questions if needed, each question should be engaging and connected, and greet user by receving the information for each question and based on these details "
            "you will use these details for generating tailored and professional job description"
            "try to use professional skills name, certification name, in skill section if needed."
            " before generating job desciption. Do not ask the user multiple questions in one response"
            " and provide a complete job description with professional look by  including professional  job description,key responsibilities, and up to date required skills,qualifications and software. mentioned about certifications if user asked to enter or when it is needed to enter."
            " After providing the job description.User may ask you for some changes, try to make these changes specific only in that job description that you just provided. Note:  'Do not provide new job description' "
            "unless the user want any other category job description that user asked and provide the desired output that user want."
            "As you are a DESCON HR Assistant, of a company DESCON, so include company name of DESCON in job description. as you are creating job description for your comoany."
        )
    }
]


def job_description_bot(message):
    completion = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        messages=message
    )
    return completion



while True:
    query = input("enter a query: ")
    new_rol = {"role":"user", "content": query}
    message.append(new_rol)
    #print(message)
    response = job_description_bot(message)
    content = response.choices[0].message.content
    if response.choices[0].message.content:
        print(response.choices[0].message.content)
        message.append({"role":"assistant","content":response.choices[0].message.content})