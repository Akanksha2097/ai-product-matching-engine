"""Baseline, semantic-compatible, and hybrid matching models."""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from rapidfuzz.fuzz import ratio

from src.candidate_generation.retrieval import CandidateRetriever
from src.preprocessing.normalization import normalize_brand, parse_quantity


@dataclass
class MatchResult:
    product_id: str
    score: float
    model: str
    decision: str
    reasons: list[str]


def token_similarity(a: str, b: str) -> float:
    return ratio(a, b) / 100.0


def model_tokens(text: str) -> set[str]:
    """Extract model identifiers while preserving suffixes such as ``XL``."""
    raw_tokens = re.findall(r"[a-z]*\d+[a-z0-9]*|[a-z]+", text.lower())
    tokens: set[str] = set()
    index = 0
    while index < len(raw_tokens):
        token = raw_tokens[index]
        if token.isdigit() and index + 1 < len(raw_tokens):
            suffix = raw_tokens[index + 1]
            if suffix in {"xl", "pro", "max", "plus", "mini"}:
                tokens.add(token + suffix)
                index += 2
                continue
        if any(character.isdigit() for character in token) and token not in {
            "1",
            "2",
            "3",
            "6",
            "12",
        }:
            tokens.add(token)
        index += 1
    return tokens


def model_identifiers_compatible(left: set[str], right: set[str]) -> bool:
    """Allow abbreviated identifiers such as XM4 to match WH-1000XM4."""
    return any(
        first == second
        or (len(first) >= 3 and second.endswith(first))
        or (len(second) >= 3 and first.endswith(second))
        for first in left
        for second in right
    )


class MatchingEngine:
    def __init__(self, catalog: pd.DataFrame):
        self.catalog = catalog
        self.retriever = CandidateRetriever(catalog)

    def score_candidate(self, query: str, row: pd.Series, model: str = "hybrid") -> tuple[float, list[str]]:
        text = token_similarity(query.lower(), row.title_norm)
        tfidf = float(row.get("tfidf_score", 0.0))
        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        candidate_tokens = set(re.findall(r"[a-z0-9]+", row.title_norm))
        brand_match = bool(row.brand_norm and row.brand_norm in normalize_brand(query))
        query_quantity = parse_quantity("", query)
        candidate_quantity = row.get("quantity_norm")
        quantity_conflict = bool(query_quantity is not None and pd.notna(candidate_quantity)
                                 and query_quantity != float(candidate_quantity))
        query_models = model_tokens(query)
        candidate_models = model_tokens(row.title_norm)
        model_conflict = bool(
            query_models
            and candidate_models
            and not model_identifiers_compatible(query_models, candidate_models)
        )
        if model == "fuzzy":
            score = text
        elif model == "tfidf":
            score = min(1.0, tfidf)
        else:
            overlap = len(query_tokens & candidate_tokens) / max(1, len(query_tokens | candidate_tokens))
            score = min(1.0, 0.45 * tfidf + 0.35 * text + 0.15 * overlap + 0.05 * brand_match)
        reasons = ["High text similarity" if text >= .75 else "Partial text similarity"]
        if brand_match:
            reasons.append("Brand matches query")
        if quantity_conflict:
            score *= 0.55
            reasons.append("Conflicting package quantity")
        if model_conflict:
            score *= 0.45
            reasons.append("Conflicting model or variant identifier")
        if not brand_match:
            reasons.append("Brand not explicit or does not match")
        return score, reasons

    def match(self, query: str, k: int = 5, model: str = "hybrid") -> list[MatchResult]:
        candidate_k = min(len(self.catalog), max(k, k * 5))
        candidates = self.retriever.retrieve(query, k=candidate_k)
        scored = []
        for _, row in candidates.iterrows():
            score, reasons = self.score_candidate(query, row, model)
            conflict = any(reason.startswith("Conflicting") for reason in reasons)
            decision = "reject" if conflict else "auto-match" if score >= .78 else "review" if score >= .48 else "reject"
            scored.append(MatchResult(str(row.product_id), round(score, 4), model, decision, reasons))
        return sorted(scored, key=lambda x: x.score, reverse=True)[:k]
