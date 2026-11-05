"""Create a small real-data Parquet cache without redistributing API records."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.ingestion.openfoodfacts import fetch_products

if __name__ == "__main__":
    destination = Path(__file__).parents[1] / "data/cache/openfoodfacts.parquet"
    frame = fetch_products(destination)
    print(f"Cached {len(frame)} Open Food Facts products at {destination}")
