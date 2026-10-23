from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from groq import Groq

app = FastAPI(title="InsightAI LLM Engine", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client using environment variable
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class AnalyzeRequest(BaseModel):
    text: str
    mode: Optional[str] = "summary"
    question: Optional[str] = None

@app.post("/api/analyze")
async def analyze_text(request: AnalyzeRequest):
    clean_text = request.text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        if request.mode == "summary":
            prompt = f"Please provide a concise, professional summary of the following text:\n\n{clean_text}"
        elif request.mode == "key_points":
            prompt = f"Extract the key bullet points from the following text:\n\n{clean_text}"
        elif request.mode == "question":
            q = request.question or "What is this text about?"
            prompt = f"Based on the following text, answer this question: '{q}'\n\nText:\n{clean_text}"
        else:
            prompt = f"Analyze the following text:\n\n{clean_text}"

        # Call Llama 3 model via Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert AI assistant that analyzes documents, extracts key points, and answers questions accurately."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",  # Fast and powerful open-source LLM
        )

        result_text = chat_completion.choices[0].message.content

        return {
            "mode": request.mode,
            "word_count": len(clean_text.split()),
            "result": result_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
