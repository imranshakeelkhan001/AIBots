import os
from openai import OpenAI
from dotenv import load_dotenv
from utils.rm_md import remove_markdown_characters
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel

from fastapi.responses import PlainTextResponse
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI()





class mul_com(BaseModel):
    multi_comment: list[str]


def multi_cmnt_suggest(user_input):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "you are provided with the user input"
                                          "multi_comment: you are a AI assistant. your job is to suggest replies 2 to 3 in a concise way based on the provided context in professional tune.you will not answer any counter question that is asked in provided context at any cost."
                                          "for example if the provided context contain 'who are you' you are not going to answer it you will simply suggest replies on it. you are not a question answering bot so make sure do not answer only suggest replies. "
                                          "so make sure to only  suggest replies based on the provided context in professional tune. "
                                          "suggested replies should be 1 sentence, you should only suggest the replies in professional way and not add context from your own,"
                                          " donot write it in commanding tone.makesure the tune remain same. do not answer from the provided context. Make sure the suggested replies as individual to an individual."
                                          " make sure to suggest 2 to 3 or 3 to 4 if needed."
                                          "it must be concise"
                                          ""},
            {"role": "user", "content": user_input},
        ],
        response_format=mul_com,
    )


    multi_comment = completion.choices[0].message.parsed.multi_comment
    print(multi_comment)
    return multi_comment