from fastapi import APIRouter, Body
import json
from controllers.feedback_offer_bot import offer_generator

router = APIRouter()

@router.get("/offer_generator")
async def  offer_gen(query:str, id: int):
    offer= await offer_generator(query, id)
    return offer