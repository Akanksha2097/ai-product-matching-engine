# Evaluation

Run `python scripts/run_evaluation.py` to write `data/demo/evaluation.json`. This tiny synthetic fixture is a smoke test, not a benchmark.

For real benchmark evaluation, download the five Abt–Buy files described in [data_sources.md](data_sources.md) and run `OMP_NUM_THREADS=1 python scripts/evaluate_benchmark.py`. The script trains logistic regression on RapidFuzz, TF-IDF, and MiniLM features; FAISS indexes the right-side product embeddings. The published pair files reuse product IDs across train/test, so the script pools them and creates a new left-entity-disjoint 60/20/20 split. Right-side products can still recur across splits, a remaining leakage/realism limitation. Thresholds are selected on validation for at least 0.90 precision, then applied unchanged to the held-out test. Scores for individual baselines are similarity scores, not calibrated probabilities.

Measured on 2026-11-04: retrieval recall@1 0.7403, recall@5 0.9484, recall@10 0.9786. Held-out F1: RapidFuzz 0.0376, TF-IDF 0.2913, MiniLM 0.0822, hybrid logistic 0.2581. These high-precision operating points sacrifice recall, and the hybrid did not outperform TF-IDF. Full metrics, confusion matrices, and run time are in `data/demo/benchmark_evaluation.json`.

The benchmark runner reports precision, recall, F1, false-positive rate, confusion matrices, retrieval recall@K, and feature/embedding throughput. A production study should enforce disjointness on both sides of the pair graph, repeat splits across seeds, and investigate difficult false positives and domain transfer.
