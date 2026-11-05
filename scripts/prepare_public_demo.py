"""Export the small cached Open Food Facts sample for the offline demo."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
source = ROOT / "data/cache/openfoodfacts.parquet"
target = ROOT / "data/demo/openfoodfacts.csv"
frame = pd.read_parquet(source)
frame.to_csv(target, index=False)
print(f"Wrote {len(frame)} attributed public products to {target}")
