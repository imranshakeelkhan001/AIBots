from fastapi import APIRouter, Body
import json
from controllers.v1_final_lightcast import jd_generator

router = APIRouter()

@router.get("/V1_Job_Description_with_LightcastAPI")
async def  JD_gen(query:str, id: int):
    jd= await jd_generator(query, id)
    return jd