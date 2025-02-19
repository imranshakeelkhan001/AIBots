import os
from openai import OpenAI
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel

from fastapi.responses import PlainTextResponse
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI()




class job_req(BaseModel):
    job_requisition: str


def gen_job_req(user_input):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "you are provided with the user input"
                                          "job_requisition: you are a DESCON HR assistant. your job is to generate a Job Requisition based on provided user input.The user input contain all the information that is required"
                                          "to generate a Job Requisition. Make sure it should be professional. Your response will be used in creating a job Description."
                                          "So make sure provide a professional Job Requisition."
                                          ""},
            {"role": "user", "content": f"here is user input{user_input}"},
        ],
        response_format=job_req,
    )


    Job_requ = completion.choices[0].message.parsed.job_requisition
    print(Job_requ)
    return Job_requ


