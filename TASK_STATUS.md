# MatchWise AI — Task Status

Updated: 2026-11-04 (US Central time)

## Completed and verified

- Initialized the project structure and local Git repository on `main`. Removed the broken Python 3.13 virtual environment created in this project; local checks currently use Python 3.12.
- Implemented reproducible demo catalog, normalization, candidate retrieval, three scoring approaches, hybrid decision engine, evaluation, FastAPI, Streamlit, tests, CI, and documentation.
- Ingested 28 real Open Food Facts records into Parquet and an attributed offline CSV; benchmark and food catalogs stay separate.
- Downloaded and evaluated the 9,575-pair DeepMatcher Abt–Buy benchmark locally without redistributing source records. Implemented MiniLM, FAISS retrieval, and a trained logistic hybrid.
- Added SQLite human-review history and export.
- Published and verified the public repository: https://github.com/Akanksha2097/ai-product-matching-engine on `main`.
- Deployed and loaded the free Streamlit Community Cloud app: https://matchwise-ai-akanksha2097.streamlit.app/ . The initial Poetry build and a later `src` import failed; `requirements.txt` and installing the local package resolved both failures.
- Captured hosted-app screenshots in `docs/streamlit-overview.png`, `docs/streamlit-matcher.png`, `docs/streamlit-batch.png`, and `docs/streamlit-evaluation.png`. Fixed a dark-theme/light-background contrast issue after visual inspection.

## Verification

- Ruff passed; pytest passed (7 tests, including all Streamlit pages and API validation checks).
- Synthetic fixture evaluation ran: precision 1.0, recall 0.6, F1 0.75, false-positive rate 0.0 on 8 rows. These are smoke-test figures, not benchmark claims.
- Python compilation passed. Open Food Facts and DeepMatcher source URLs returned HTTP 200.
- Hosted browser checks passed for Overview, Product Matcher query entry, batch CSV upload using `data/demo/batch_sample.csv`, visible batch results and download button, Human Review `needs_evidence` save confirmation and export button, Model Evaluation charts/confusion matrix, and Technical Architecture.
- GitHub Actions run 5 passed on commit `29cfa69`; earlier run 1's missing dev dependency was fixed.

## Remaining / limitations

- CI runs on every push; the newest run should always be checked before sharing a changed revision.
- More pack/variant edge-case tests would improve rigor; the current suite has 7 tests.
- Benchmark accuracy is measured on Abt–Buy, not the 28-record food demo. The app's interactive hybrid remains heuristic and has no validated threshold on the food domain.
- The held-out split is by left-side entity; right-side entities can recur across partitions. The hybrid underperformed TF-IDF at this operating point.
- Databricks notebook source exists in `notebooks/databricks/matchwise_demo.py` but has not been imported or executed. A workspace ID screenshot alone does not provide a workspace URL or authenticated session; no Databricks CLI or connector is available. This optional extension does not block the core project.
- Hosted SQLite review history is ephemeral; users should export it. The FastAPI service was tested locally, not publicly deployed. No billable infrastructure was intentionally provisioned.
