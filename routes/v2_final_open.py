from fastapi import APIRouter, Body
from controllers.v2_final_open import jobD_generator

router = APIRouter()

@router.get("/V2_Job_Description")
async def V2_job_description(query:str,id:int):
    job_des= await jobD_generator(query,id)
    return job_des
