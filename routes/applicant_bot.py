from controllers.applicant_bot import applicant_bot
from fastapi import APIRouter
from fastapi import UploadFile, File
from typing import Optional
router = APIRouter()

@router.post("/applicant_bot")
async def applicant(query: Optional[str] , id: int, file: Optional[UploadFile] = File(None)):    # Pass the list of uploaded files directly to the generate_justification function
    struct_data = await applicant_bot(query, id, file)
    return struct_data
