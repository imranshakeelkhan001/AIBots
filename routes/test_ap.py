from controllers.test_ap import extract_data_from_cv
from fastapi import APIRouter, File, UploadFile

router = APIRouter()



@router.post("/upload_file_extract_data")
async def upld_fil(jd_id: str, skill_factor: int, cv_files: UploadFile = File(...)):
    # Pass the list of uploaded files directly to the generate_justification function
    struct_data = await extract_data_from_cv(jd_id, skill_factor, cv_files)
    return struct_data



