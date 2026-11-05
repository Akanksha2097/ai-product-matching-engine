"""Small, cached Open Food Facts catalog ingestion for the public demo."""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

LOG = logging.getLogger(__name__)
API = "https://world.openfoodfacts.org/cgi/search.pl"
DEFAULT_USER_AGENT = (
    "MatchWiseAI/0.1 (https://github.com/Akanksha2097/ai-product-matching-engine)"
)


def fetch_products(cache_path: str | Path, *, page_size: int = 30, retries: int = 3) -> pd.DataFrame:
    """Fetch one bounded page of real products; subsequent calls read cached Parquet."""
    if not 1 <= page_size <= 50:
        raise ValueError("page_size must be between 1 and 50")
    path = Path(cache_path)
    if path.exists():
        return pd.read_parquet(path)
    params = {
        "action": "process", "json": "1", "page_size": str(page_size),
        "fields": "code,product_name,brands,categories,quantity,packaging",
        "sort_by": "unique_scans_n", "countries_tags": "en:united-states",
    }
    url = API + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url, headers={"User-Agent": os.getenv("OPENFOODFACTS_USER_AGENT", DEFAULT_USER_AGENT)}
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
            products = payload.get("products", [])
            if not isinstance(products, list):
                raise TypeError("Open Food Facts response lacks a products list")
            rows = []
            for product in products:
                code = str(product.get("code") or "").strip()
                title = str(product.get("product_name") or "").strip()
                if code and title:
                    rows.append({
                        "product_id": code,
                        "brand": str(product.get("brands") or ""),
                        "title": title,
                        "category": str(product.get("categories") or ""),
                        "quantity": str(product.get("quantity") or ""),
                        "unit": str(product.get("packaging") or ""),
                    })
            frame = pd.DataFrame(rows)
            if frame.empty:
                raise ValueError("Open Food Facts returned no usable products")
            frame = frame.drop_duplicates("product_id").reset_index(drop=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            return frame
        except Exception:
            LOG.exception("Open Food Facts attempt %s/%s failed", attempt + 1, retries)
            if attempt < retries - 1:
                time.sleep(2**attempt)
    raise RuntimeError("Open Food Facts ingestion failed after bounded retries")
