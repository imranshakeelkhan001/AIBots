from fastapi import FastAPI
from routes.v1_final_lightcast import router as v1_job_description
# from routes.v2_final_open import router as v2_job_description
from routes.retrievebyid import router as history_byid
from routes.applicant_bot import router as applicant_chatbot
from routes.test_ap import  router as test_router
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from routes.resume_parser import router as resume_parser
from routes.resume_similarity import router as resume_sim
from routes.feedback_offer_bot import router as feedback_bot
from routes.suggest_reply import router as suggest_reply
from routes.hr_bot import router as hr
from dotenv import load_dotenv





load_dotenv()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,  # Pass the class directly, not an instance
    allow_origins=["*"],       # Allow all origins
    allow_methods=["*"],       # Allow all HTTP methods
    allow_headers=["*"],       # Allow all headers
    allow_credentials=True     # Allow credentials (cookies or auth headers)
)
app.include_router(v1_job_description)
# app.include_router(v2_job_description)
app.include_router(history_byid)
#app.include_router(resume_parser)
app.include_router(applicant_chatbot)
app.include_router(resume_sim)
app.include_router(hr)
app.include_router(feedback_bot)
app.include_router(suggest_reply)
app.include_router(test_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
