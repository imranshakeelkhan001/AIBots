from fastapi import APIRouter, Body
from controllers.Job_requisition import gen_job_req
router = APIRouter()

@router.get("/Job_requisition_bot")
def Job_req(prompt: str ):
    extended_content = gen_job_req(prompt)
    return  extended_content
