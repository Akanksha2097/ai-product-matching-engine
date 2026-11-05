# Architecture

`ingestion` retrieves permitted public data with caching and retries. `preprocessing` validates and standardizes attributes. `candidate_generation` avoids all-pairs comparison with TF-IDF retrieval. `models` produces interpretable fuzzy, TF-IDF, and hybrid ranking scores. `decision_engine` behavior is embedded in the engine's review thresholds. FastAPI exposes a validated API and Streamlit provides interactive review and batch workflows.
