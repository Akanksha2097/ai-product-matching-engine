from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.matching import MatchingEngine
from src.preprocessing.normalization import normalize_catalog

CATALOG = normalize_catalog(pd.read_csv(Path(__file__).parents[2] / "data/demo/catalog.csv"))
ENGINE = MatchingEngine(CATALOG)
app = FastAPI(title="MatchWise AI API", version="0.1.0")


class MatchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    k: int = Field(default=5, ge=1, le=20)
    model: str = Field(default="hybrid", pattern="^(hybrid|fuzzy|tfidf)$")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "catalog_records": str(len(CATALOG))}


@app.get("/model-info")
def model_info() -> dict:
    return {"name": "MatchWise hybrid matcher", "models": ["fuzzy", "tfidf", "hybrid"], "thresholds": {"auto_match": 0.78, "review": 0.48}, "score_note": "Scores are ranking scores, not calibrated probabilities."}


@app.post("/match")
def match(request: MatchRequest) -> dict:
    results = ENGINE.match(request.query, request.k, request.model)
    return {"query": request.query, "results": [r.__dict__ for r in results]}


@app.post("/batch-match")
def batch_match(rows: list[MatchRequest]) -> dict:
    if len(rows) > 1000:
        raise HTTPException(413, "Batch limit is 1,000 rows")
    return {"results": [{"query": r.query, "matches": [m.__dict__ for m in ENGINE.match(r.query, r.k, r.model)]} for r in rows]}
