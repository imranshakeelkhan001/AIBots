from controllers.resume_similarity import extract_data_from_cv
from fastapi import APIRouter, File, UploadFile

router = APIRouter()

@router.post("/resume_similarity")
async def res_sim(cv_files: UploadFile = File(...)):
    # Pass the list of uploaded files directly to the generate_justification function
    struct_data = await extract_data_from_cv(cv_files)
    return struct_data
