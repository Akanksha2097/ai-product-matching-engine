"""Conservative product attribute normalization."""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

BRAND_ALIASES = {"hewlett packard": "hp", "hewlett-packard": "hp", "apple inc": "apple"}
UNIT_ALIASES = {"pcs": "each", "piece": "each", "pieces": "each", "ct": "each", "count": "each"}


def clean_text(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).lower().strip()
    text = re.sub(r"[^a-z0-9%./+ -]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_brand(value: Any) -> str:
    brand = clean_text(value)
    return BRAND_ALIASES.get(brand, brand)


def parse_quantity(value: Any, title: Any = "") -> float | None:
    text = clean_text(value) or clean_text(title)
    match = re.search(r"(?:pack of|\bpack\b|\bcount\b|\bct\b|x)\s*(\d+(?:\.\d+)?)", text)
    if not match:
        match = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:pack|pk|count|ct)\b", text)
    return float(match.group(1)) if match else (float(value) if str(value).replace('.', '', 1).isdigit() else None)


def normalize_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"product_id", "brand", "title"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    result = frame.copy()
    for col in ["product_id", "brand", "title", "category", "unit"]:
        if col not in result:
            result[col] = ""
    result["product_id"] = result["product_id"].astype(str)
    result["brand_norm"] = result["brand"].map(normalize_brand)
    result["title_norm"] = result["title"].map(clean_text)
    result["category_norm"] = result["category"].map(clean_text)
    result["unit_norm"] = result["unit"].map(lambda x: UNIT_ALIASES.get(clean_text(x), clean_text(x)))
    result["quantity_norm"] = [parse_quantity(q, t) for q, t in zip(result.get("quantity", ""), result["title"])]
    result["search_text"] = (result["brand_norm"] + " " + result["title_norm"]).str.strip()
    return result.drop_duplicates("product_id").reset_index(drop=True)
