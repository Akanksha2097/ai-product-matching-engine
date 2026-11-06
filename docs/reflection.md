# AI collaboration and system-design reflection

## How AI assisted

AI assistance helped outline the modular architecture, compare retrieval and ranking options, generate test ideas, and audit the implementation against the course rubric. It was also useful for identifying places where a plausible text match could hide a product-variant error. Every generated claim was checked against source code, tests, or saved evaluation output before it was included in the documentation.

## Suggestions accepted

I accepted the suggestion to keep semantic embeddings and FAISS as an optional benchmark path while retaining a lightweight lexical demo. This made the default application practical to deploy while still allowing a meaningful comparison with MiniLM. I also accepted the recommendation to over-retrieve candidates before final ranking; regression testing showed that scoring only the first lexical hit could reject a base `67` cartridge without considering the correct `67 XL` record.

## Suggestions revised or rejected

I rejected the assumption that a more complex hybrid must be the best model. Held-out results showed TF-IDF F1 (0.2913) above the logistic hybrid (0.2581), so the documentation states that result instead of marketing the hybrid as an improvement. I also revised a first-pass identifier rule that treated `XM4` and `WH-1000XM4` as conflicting. The final rule permits meaningful suffix abbreviations while preserving the distinction between `67` and `67 XL`, and tests lock in both behaviors.

## Verification and learning

The main learning was that reliability depends as much on candidate generation and decision order as on the score formula. A guardrail applied after an overly narrow retrieval step can be logically correct and still produce the wrong final result. I verified changes with Ruff, nine pytest cases, the synthetic evaluation script, actual FastAPI responses, and the dashboard rendering test. Future work should add labeled food-domain cases, category-level fairness analysis, durable review storage, and graph-disjoint benchmark evaluation.
