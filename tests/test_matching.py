from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import app
from src.models.matching import MatchingEngine
from src.preprocessing.normalization import normalize_catalog
from src.review.store import ReviewStore

ROOT = Path(__file__).parents[1]


def test_normalization_aliases_and_deduplication():
    frame = pd.DataFrame([{"product_id": "1", "brand": "Hewlett Packard", "title": "67 XL", "quantity": 2}, {"product_id": "1", "brand": "HP", "title": "other"}])
    result = normalize_catalog(frame)
    assert result.iloc[0].brand_norm == "hp"
    assert len(result) == 1


def test_engine_flags_pack_conflict():
    catalog = normalize_catalog(pd.read_csv(ROOT / "data/demo/catalog.csv"))
    row = catalog[catalog.product_id.eq("off-001")].iloc[0]
    _, reasons = MatchingEngine(catalog).score_candidate("HP 67XL black cartridge 12 pack", row)
    assert "Conflicting package quantity" in reasons


def test_engine_accepts_abbreviated_model_identifier():
    catalog = normalize_catalog(pd.read_csv(ROOT / "data/demo/catalog.csv"))
    result = MatchingEngine(catalog).match("Sony XM4 headphones", k=1)[0]
    assert result.product_id == "off-008"
    assert "Conflicting model or variant identifier" not in result.reasons


def test_engine_distinguishes_base_and_xl_variants():
    catalog = normalize_catalog(pd.read_csv(ROOT / "data/demo/catalog.csv"))
    engine = MatchingEngine(catalog)
    base = catalog[catalog.product_id.eq("off-003")].iloc[0]
    _, reasons = engine.score_candidate("HP 67 XL black ink cartridge", base)
    assert "Conflicting model or variant identifier" in reasons
    assert engine.match("HP 67 XL black ink cartridge", k=1)[0].product_id in {
        "off-001",
        "off-002",
    }


def test_api_contract():
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    response = client.post("/match", json={"query": "Sony XM4 headphones", "k": 2})
    assert response.status_code == 200
    assert len(response.json()["results"]) == 2
    assert client.post("/batch-match", json=[{"query": "Sony XM4"}]).status_code == 200
    assert client.post("/match", json={"query": "x"}).status_code == 422


def test_api_rejects_invalid_scoring_options_and_large_batches():
    client = TestClient(app)
    assert client.post("/match", json={"query": "Sony XM4", "k": 0}).status_code == 422
    assert client.post("/match", json={"query": "Sony XM4", "model": "unknown"}).status_code == 422
    assert client.post("/match", json={"query": "   "}).status_code == 422
    assert client.post("/batch-match", json=[{"query": "Sony XM4"}] * 1001).status_code == 413


def test_api_model_info_discloses_uncalibrated_scores():
    response = TestClient(app).get("/model-info")
    assert response.status_code == 200
    assert "not calibrated probabilities" in response.json()["score_note"]


def test_review_persists(tmp_path):
    store = ReviewStore(tmp_path / "reviews.sqlite")
    store.record("HP 67 XL", "off-001", "reject", "variant mismatch")
    history = ReviewStore(tmp_path / "reviews.sqlite").history()
    assert len(history) == 1
    assert history.iloc[0]["note"] == "variant mismatch"
