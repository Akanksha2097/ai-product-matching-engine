"""Local SQLite review history. Hosted filesystem persistence is not guaranteed."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


class ReviewStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reviewed_at TEXT NOT NULL,
                query TEXT NOT NULL,
                product_id TEXT NOT NULL,
                decision TEXT NOT NULL CHECK (decision IN ('accept', 'reject', 'needs_evidence')),
                note TEXT NOT NULL DEFAULT ''
            )""")

    def record(self, query: str, product_id: str, decision: str, note: str = "") -> None:
        if decision not in {"accept", "reject", "needs_evidence"}:
            raise ValueError("Invalid review decision")
        with sqlite3.connect(self.path) as conn:
            conn.execute("INSERT INTO reviews (reviewed_at, query, product_id, decision, note) VALUES (?, ?, ?, ?, ?)",
                         (datetime.now(UTC).isoformat(), query, product_id, decision, note))

    def history(self) -> pd.DataFrame:
        with sqlite3.connect(self.path) as conn:
            return pd.read_sql_query("SELECT * FROM reviews ORDER BY id DESC", conn)
