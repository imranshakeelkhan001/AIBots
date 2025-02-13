import os
from os import getenv
import PyPDF2
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv
from io import BytesIO
from fastapi import UploadFile

# Load environment variables
load_dotenv()
OPENAI_API_KEY = getenv("OPENAI_API_KEY")
MODEL = getenv("MODEL")

# Initialize OpenAI client
from openai import OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file: BytesIO) -> str:
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text.strip()

# Data structure for storing parsed profile data
class ProfileData(BaseModel):
    name: str
    email: str
    contact_number: int
    education: str
    total_experience_in_years_with_job_title: List[str]
    certifications: List[str]


# Function to process a single CV and return skills
async def extract_skills_from_cv(cv_file: UploadFile):

    # Read and extract text from the uploaded PDF
    pdf_file = BytesIO(await cv_file.read())  # Read the uploaded file into memory
    cv_text = extract_text_from_pdf(pdf_file)  # Extract text from the PDF

     # OpenAI API call
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content":         f"You are given a candidate's CV text"
        f"Your task is to extract following information of the candidate"
        f"1. Name"
        f"2. Email"
        f"3. contact_number"
        f"4. education"
        f"5. total_experience_in_years_with_job_title"
        f"6. certifications"
        f"If any of the above data is not found in the given CV text then leave it as None"},
            {"role": "user", "content": cv_text},
        ],
          # Limit tokens to focus only on skills
        response_format= ProfileData
    )

    # Parse the response to extract skills
    try:
        return response.choices[0].message.parsed
    except Exception as e:
        return ValueError(f"Error extracting skills: {e}")
