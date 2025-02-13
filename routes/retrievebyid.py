from fastapi import APIRouter, Body
import json
from controllers.retrievebyid import retrive_history

router = APIRouter()

@router.get("/Retrievebyid")
async def  jd_by_id( id: int):
    history= await  retrive_history(id)
    return history