# MatchWise AI — Task Status

Updated: 2026-11-04 (latest verification)

## Completed

- Initialized the project structure and local Git repository on `main`. Removed the broken Python 3.13 virtual environment created in this project; local checks currently use Python 3.12.
- Implemented reproducible demo catalog, normalization, candidate retrieval, three scoring approaches, hybrid decision engine, evaluation, FastAPI, Streamlit, tests, CI, and documentation.
- Ingested 28 real Open Food Facts records into Parquet and an attributed offline CSV; benchmark and food catalogs stay separate.
- Downloaded and evaluated the 9,575-pair DeepMatcher Abt–Buy benchmark locally without redistributing source records. Implemented MiniLM, FAISS retrieval, and a trained logistic hybrid.
- Added SQLite human-review history and export.

## Verification

- Ruff passed; pytest passed (5 tests, including all Streamlit pages).
- Synthetic fixture evaluation ran: precision 1.0, recall 0.6, F1 0.75, false-positive rate 0.0 on 8 rows. These are smoke-test figures, not benchmark claims.
- Python compilation passed. Open Food Facts and DeepMatcher source URLs returned HTTP 200.
- Streamlit started locally on port 8502 after upgrading an incompatible preinstalled version; interaction flows have not been end-to-end verified.
- Git 2.50.1 works outside the agent sandbox; public GitHub repository created and first commit cc9e42c pushed/verified on `main`.

## Remaining / external

- Verify batch upload/download and review interaction in a live browser, then capture portfolio screenshots.
- Complete remaining API error and pack/variant edge-case checks; only 5 tests currently exist.
- Benchmark accuracy is measured on Abt–Buy, not the 28-record food demo. The app's interactive hybrid remains heuristic and has no validated threshold on the food domain.
- Produce verified screenshots and complete user-flow checks before treating the dashboard as portfolio-ready.
- First GitHub Actions run failed because the clean environment lacked `httpx2`; fixed the dev dependency and a follow-up push/CI run is pending.
- Streamlit Community Cloud is signed out in the shared browser; user sign-in is required for deployment.
- Streamlit Community Cloud and Databricks status require account-level access; no paid resource will be provisioned.
