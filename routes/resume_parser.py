from controllers.resume_parser import extract_skills_from_cv
from fastapi import APIRouter, File, UploadFile
router = APIRouter()

router = APIRouter()

@router.post("/resume_parser")
async def res_par(cv_files: UploadFile= File(...)):
    # Pass the list of uploaded files directly to the generate_justification function
    struct_data = await extract_skills_from_cv(cv_files)
    return struct_data
