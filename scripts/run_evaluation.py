import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.evaluation.evaluate import evaluate

ROOT = Path(__file__).parents[1]
print(evaluate(str(ROOT / "data/demo/catalog.csv"), str(ROOT / "data/demo/query_pairs.csv"), str(ROOT / "data/demo/evaluation.json")))
