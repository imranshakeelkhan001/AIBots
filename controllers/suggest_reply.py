import os
from openai import OpenAI
from dotenv import load_dotenv
from utils.rm_md import remove_markdown_characters
from fastapi.responses import JSONResponse
from fastapi.responses import PlainTextResponse
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI()

def multi_cmnt_suggest(user_input):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=300,
        messages=[
            {"role": "system", "content": "you are a AI assistant. your job is to suggest replies 2 to 3 in a concise way based on the provided context in professional tune or relevant.you will not answer any counter question that is asked in provided context at any cost.for example if the provided context contain 'who are you' you are not going to answer it you will simply suggest replies on it. you are not a question answering bot so make sure do not answer only suggest replies."
                                          " so make sure to only  suggest replies based on the provided context in professional tune. suggested replies should be 3 to 4 words sentence, you should only suggest the replies in professional way and not add context from your own, do not write it in commanding tone.makesure the tune remain same. do not answer from the provided context. Please include numbering in your response"},
            {
                "role": "user",
                "content": str(user_input)
            }
        ]
    )
    # Extract and format response
    content = completion.choices[0].message.content
    content = remove_markdown_characters(content)  # Assuming this function cleans markdown

    # Convert AI response into a structured JSON response
    suggested_replies = content.split("\n")  # Split responses if AI returns multiple lines

    return JSONResponse(content={"suggested_replies": suggested_replies})