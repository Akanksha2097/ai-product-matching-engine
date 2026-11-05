"""Evaluate Abt–Buy with left-entity-disjoint splits and measured retrieval metrics.

Download the five source CSVs from the URLs in docs/data_sources.md into
data/raw/abt_buy/. Source records stay out of Git; only aggregate results ship.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import faiss
import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit

from src.preprocessing.normalization import clean_text

ROOT = Path(__file__).parents[1]
DATA = ROOT / "data/raw/abt_buy"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predicted, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "false_positive_rate": round(float(fp / max(1, tn + fp)), 4),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def select_threshold(y: np.ndarray, scores: np.ndarray, min_precision: float = 0.9) -> float:
    candidates = np.unique(np.r_[scores, 1.000001])
    eligible = [(metrics(y, scores, float(t))["recall"], float(t)) for t in candidates
                if metrics(y, scores, float(t))["precision"] >= min_precision]
    return min(eligible, key=lambda x: (-x[0], x[1]))[1] if eligible else 1.000001


def run() -> dict:
    a = pd.read_csv(DATA / "tableA.csv").set_index("id")
    b = pd.read_csv(DATA / "tableB.csv").set_index("id")
    pairs = pd.concat([pd.read_csv(DATA / f"{split}.csv") for split in ("train", "valid", "test")], ignore_index=True)
    pairs["left_title"] = pairs.ltable_id.map(a.name).fillna("").map(clean_text)
    pairs["right_title"] = pairs.rtable_id.map(b.name).fillna("").map(clean_text)
    if pairs[["left_title", "right_title"]].eq("").any().any():
        raise ValueError("Benchmark contains missing pair titles")

    # The published row-level splits reuse entities. Group on the left product ID.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    train_idx, remainder = next(splitter.split(pairs, pairs.label, pairs.ltable_id))
    rest = pairs.iloc[remainder]
    valid_rel, test_rel = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=43)
                               .split(rest, rest.label, rest.ltable_id))
    valid_idx, test_idx = remainder[valid_rel], remainder[test_rel]

    start = time.perf_counter()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    vectorizer.fit(pd.concat([a.name, b.name]).fillna("").map(clean_text))
    left_unique = a.name.fillna("").map(clean_text)
    right_unique = b.name.fillna("").map(clean_text)
    left_tfidf, right_tfidf = vectorizer.transform(left_unique), vectorizer.transform(right_unique)
    left_pos = {key: i for i, key in enumerate(a.index)}
    right_pos = {key: i for i, key in enumerate(b.index)}
    li = pairs.ltable_id.map(left_pos).to_numpy(dtype=int)
    ri = pairs.rtable_id.map(right_pos).to_numpy(dtype=int)
    tfidf = np.asarray(left_tfidf[li].multiply(right_tfidf[ri]).sum(axis=1)).ravel()
    fuzzy = np.array([ratio(x, y) / 100 for x, y in zip(pairs.left_title, pairs.right_title)])

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    left_emb = model.encode(left_unique.tolist(), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    right_emb = model.encode(right_unique.tolist(), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    semantic = np.sum(left_emb[li] * right_emb[ri], axis=1)
    duration = time.perf_counter() - start

    index = faiss.IndexFlatIP(right_emb.shape[1])
    index.add(np.ascontiguousarray(right_emb.astype("float32")))
    _, neighbors = index.search(np.ascontiguousarray(left_emb.astype("float32")), 10)
    positives = pairs[pairs.label.eq(1)].drop_duplicates(["ltable_id", "rtable_id"])
    retrieval = {}
    for k in (1, 5, 10):
        hits = [right_pos[r.rtable_id] in neighbors[left_pos[r.ltable_id], :k]
                for r in positives.itertuples()]
        retrieval[f"recall_at_{k}"] = round(float(np.mean(hits)), 4)

    x = np.column_stack([fuzzy, tfidf, semantic])
    labels = pairs.label.to_numpy(dtype=int)
    classifier = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    classifier.fit(x[train_idx], labels[train_idx])
    hybrid = classifier.predict_proba(x)[:, 1]

    comparison = {}
    for name, scores in (("rapidfuzz", fuzzy), ("tfidf", tfidf),
                         ("semantic", semantic), ("hybrid_logistic", hybrid)):
        threshold = select_threshold(labels[valid_idx], scores[valid_idx])
        comparison[name] = {"validation_threshold": round(threshold, 6),
                            "test": metrics(labels[test_idx], scores[test_idx], threshold)}
    result = {
        "source": "DeepMatcher Abt–Buy published labeled pairs; source records not redistributed",
        "model": MODEL_NAME,
        "split_policy": "left-entity-disjoint 60/20/20; right entities may recur across splits",
        "n_pairs": len(pairs),
        "split_counts": {"train": len(train_idx), "validation": len(valid_idx), "test": len(test_idx)},
        "positive_pairs": int(labels.sum()),
        "retrieval": retrieval,
        "comparison": comparison,
        "feature_and_embedding_seconds": round(duration, 2),
        "feature_pairs_per_second": round(len(pairs) / duration, 2),
    }
    target = ROOT / "data/demo/benchmark_evaluation.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
