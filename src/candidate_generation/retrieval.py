"""Efficient lexical candidate retrieval."""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class CandidateRetriever:
    def __init__(self, catalog: pd.DataFrame):
        self.catalog = catalog.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(self.catalog["search_text"])

    def retrieve(self, query: str, brand: str = "", category: str = "", k: int = 5) -> pd.DataFrame:
        query_vec = self.vectorizer.transform([query.lower()])
        scores = linear_kernel(query_vec, self.matrix).ravel()
        if brand:
            scores += np.where(self.catalog.brand_norm.eq(brand.lower()), 0.15, 0)
        if category:
            scores += np.where(self.catalog.category_norm.eq(category.lower()), 0.05, 0)
        indices = np.argsort(-scores)[:k]
        result = self.catalog.iloc[indices].copy()
        result["tfidf_score"] = scores[indices]
        return result


def fuzzy_score(query: str, candidate: str) -> float:
    return ratio(query.lower(), candidate.lower()) / 100.0
