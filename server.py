"""
FastAPI server — exposes the pipeline as a REST endpoint
so your NestJS backend can call it via HTTP.

Start: uvicorn server:app --reload --port 8001
"""

from fastapi import FastAPI, HTTPException  # pip install fastapi uvicorn
from pydantic import BaseModel
from typing import Any, Optional
import logging

from pipeline import run_pipeline

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Content Idea Generator", version="1.0.0")


class BrandProfileRequest(BaseModel):
    # Mirrors your Prisma BrandProfile model (all optional)
    niche: Optional[str] = None
    audience: Optional[str] = None
    styleSummary: Optional[str] = None
    avgSentenceLength: Optional[float] = None
    commonPhrases: Optional[list] = None
    emotionalTone: Optional[str] = None
    formalityScore: Optional[float] = None
    humorUsage: bool = False
    storytellingStyle: Optional[str] = None
    topicPreferences: Optional[list] = None
    vocabularyComplexity: Optional[str] = None
    tone: Optional[str] = None
    ideaCount: int = 15


class IdeaResponse(BaseModel):
    ideas: list[dict[str, Any]]
    total: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate-ideas", response_model=IdeaResponse)
def generate_ideas(body: BrandProfileRequest):
    try:
        brand_data = body.model_dump(exclude={"ideaCount"})
        ideas = run_pipeline(brand_data, idea_count=body.ideaCount)
        return IdeaResponse(ideas=ideas, total=len(ideas))
    except Exception as exc:
        logging.error("Pipeline error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))