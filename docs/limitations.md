# Limitations

- The food catalog is only a 28-record public sample. Benchmark source records are not redistributed.
- The dashboard's interactive matcher is a transparent heuristic, not the Abt–Buy trained classifier; its food-catalog accuracy has not been measured.
- Sentence Transformers and FAISS are optional dependencies for benchmark reproduction; the default demo stays CPU-light.
- Scores are not calibrated probabilities.
- Human reviews persist to local SQLite, but Streamlit Community Cloud filesystem storage is ephemeral; users should export review history.
- The benchmark split is disjoint by left-side entity only; right-side entities may recur. Validation-selected thresholds did not guarantee the same precision on test.
- The public Streamlit deployment was verified on November 4, 2026, but Community Cloud uptime and local-file persistence are not guaranteed.
