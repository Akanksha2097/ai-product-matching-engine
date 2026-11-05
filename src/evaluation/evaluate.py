"""Reproducible evaluation over explicit demo labels."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.models.matching import MatchingEngine


def evaluate(catalog_path: str, pairs_path: str, output_path: str | None = None) -> dict:
    catalog = pd.read_csv(catalog_path)
    pairs = pd.read_csv(pairs_path)
    from src.preprocessing.normalization import normalize_catalog
    engine = MatchingEngine(normalize_catalog(catalog))
    rows = []
    for _, pair in pairs.iterrows():
        results = engine.match(pair.query_title, k=1, model="hybrid")
        top = results[0]
        rows.append({"label": int(pair.label), "prediction": int(top.product_id == pair.reference_id), "score": top.score})
    y_true = [r["label"] for r in rows]
    y_pred = [r["prediction"] for r in rows]
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result = {"provenance": "local synthetic smoke fixture; not ground truth", "n": len(rows), "precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f1), 4), "false_positive_rate": round(float(fp / max(1, fp + tn)), 4), "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]], "rows": rows}
    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
    return result
