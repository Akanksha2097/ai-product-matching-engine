# Architecture

The Mermaid source below represents the implemented system, including the intentionally separate demo and benchmark paths.

```mermaid
flowchart LR
    U[Single query or batch CSV] --> V[API and schema validation]
    F[Open Food Facts demo catalog] --> N[Conservative normalization]
    S[Synthetic variant fixture] --> N
    V --> N
    N --> R[TF-IDF candidate retrieval]
    R --> O[Over-retrieved candidate pool]
    O --> M[Fuzzy / TF-IDF / hybrid scoring]
    M --> G[Quantity and model guardrails]
    G --> D{Decision thresholds}
    D -->|high confidence| A[Auto-match]
    D -->|uncertain| H[Human review]
    D -->|conflict or low score| X[Reject]
    A --> API[FastAPI and Streamlit output]
    H --> DB[(SQLite review history)]
    X --> API

    B[Abt-Buy labeled benchmark] --> E[Entity-disjoint split and evaluation]
    E --> T[TF-IDF / MiniLM / logistic comparison]
    T --> J[Metrics JSON and dashboard]
    J -. informs validation only .-> D
```

`ingestion` retrieves permitted public data with caching and retries. `preprocessing` validates and standardizes attributes. `candidate_generation` avoids all-pairs comparison with TF-IDF retrieval. `models` produces interpretable fuzzy, TF-IDF, and hybrid ranking scores. The matching engine retrieves more candidates than the requested output count, applies variant guardrails, ranks the pool, and only then returns top-k. FastAPI exposes a validated API and Streamlit provides interactive review and batch workflows.

The dashed edge is deliberate: benchmark measurements inform evaluation practice, but the Abt-Buy results are not presented as calibration or accuracy for the separate food demo.
