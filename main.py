from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import re

app = FastAPI(title="InsightAI Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    text: str
    mode: Optional[str] = "summary"
    question: Optional[str] = None

def extract_real_summary(text: str, num_sentences: int = 3) -> str:
    """Extractive Summarization logic that selects top important sentences."""
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    if len(sentences) <= num_sentences:
        return text

    # Score sentences based on word frequency
    words = re.findall(r'\w+', text.lower())
    freq = {}
    for word in words:
        if len(word) > 3:  # ignore trivial short words
            freq[word] = freq.get(word, 0) + 1

    sentence_scores = {}
    for sentence in sentences:
        score = sum(freq.get(w.lower(), 0) for w in re.findall(r'\w+', sentence))
        sentence_scores[sentence] = score

    # Pick top scoring sentences
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    top_sentences = [s[0] for s in sorted_sentences[:num_sentences]]
    
    # Maintain original sentence order
    summary = [s for s in sentences if s in top_sentences]
    return " ".join(summary)

@app.post("/api/analyze")
async def analyze_text(request: AnalyzeRequest):
    clean_text = request.text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    words = clean_text.split()
    word_count = len(words)

    if request.mode == "summary":
        if word_count < 10:
            summary = clean_text
        else:
            summary = extract_real_summary(clean_text)
        
        return {
            "mode": "summary",
            "word_count": word_count,
            "result": summary
        }

    elif request.mode == "key_points":
        sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', clean_text) if s.strip()]
        points = sentences[:4] if len(sentences) >= 4 else sentences
        return {
            "mode": "key_points",
            "word_count": word_count,
            "result": points or ["Text too short to extract distinct key points."]
        }

    elif request.mode == "question":
        q = request.question or "What is this about?"
        # Simple query lookup in sentences
        relevant_sentences = [s for s in re.split(r'(?<=[.!?]) +', clean_text) if any(w in s.lower() for w in q.lower().split() if len(w) > 3)]
        
        ans = " ".join(relevant_sentences[:2]) if relevant_sentences else f"Based on the input document, no direct sentence specifically answered '{q}', but here is the primary context: {clean_text[:120]}..."
        return {
            "mode": "question",
            "question": q,
            "result": ans
        }

    raise HTTPException(status_code=400, detail="Invalid mode")
