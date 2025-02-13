from fastapi import APIRouter, Body
import json
from controllers.hr_bot import hr_chatbot
router = APIRouter()




@router.get("/hr_bot")
async def  hr(query:str, id: int):
    hr= await hr_chatbot (query, id)
    return hr
