from fastapi import APIRouter, Body
from controllers.suggest_reply import multi_cmnt_suggest
router = APIRouter()

@router.get("/suggest_reply")
def multi_comment_suggest(prompt: str ):
    extended_content = multi_cmnt_suggest(prompt)
    return  extended_content
