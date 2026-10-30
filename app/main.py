"""Phase 6 - Web API + page for the hallucination detector.

Run from the project root:
    uvicorn app.main:app --reload
Then open http://127.0.0.1:8000
"""

import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.pipeline import HallucinationDetector

STATIC_DIR = Path(__file__).parent / "static"
state: dict = {}
lock = threading.Lock()  # the models are shared, so handle one request at a time


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["detector"] = HallucinationDetector()  # load all models once at startup
    yield
    state.clear()


app = FastAPI(title="RAG-Based Explainable Hallucination Detector", lifespan=lifespan)


class CheckRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: Literal["rag", "plain", "custom"] = "rag"
    answer: str | None = Field(default=None, max_length=4000)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "ready": "detector" in state}


@app.post("/api/check")
def check(req: CheckRequest):
    """mode = rag    : LLM answers using retrieved evidence, then we verify it
       mode = plain  : LLM answers from its own knowledge, then we verify it
       mode = custom : verify the answer text supplied by the user"""
    if req.mode == "custom" and not (req.answer and req.answer.strip()):
        raise HTTPException(status_code=400, detail="Paste an answer to check in custom mode.")

    with lock:
        try:
            return state["detector"].run(
                req.question,
                answer=req.answer if req.mode == "custom" else None,
                use_rag=(req.mode != "plain"),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")