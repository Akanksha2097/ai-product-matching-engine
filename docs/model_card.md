# Model card

## Model and task

MatchWise AI 0.1 is a retrieval and ranking system for product catalog entity resolution. Given a product description, it ranks likely reference-catalog records and recommends auto-match, human review, or rejection. It assists catalog analysts; it does not replace accountable review for irreversible catalog changes.

## Data and attributes

The hosted demo uses 28 attributed Open Food Facts records. A separate 10-record synthetic catalog covers controlled brand, pack-size, and model-variant cases. Measured model comparison uses 9,575 labeled Abt-Buy pairs; those source records are not redistributed. Inputs include titles, brands, categories, units, quantities, and identifiers derived from text. The demo and benchmark domains are kept separate.

## Algorithm

Text and attributes are normalized without deleting model suffixes. TF-IDF retrieves a candidate pool larger than the requested result count. Fuzzy similarity, TF-IDF similarity, token overlap, brand agreement, and conflict penalties contribute to ranking. Optional benchmark code adds MiniLM embeddings, FAISS retrieval, and logistic regression. A final guardrail checks package quantities and model identifiers before thresholds route a result to auto-match, review, or reject. Explanations report the same evidence and conflicts used by scoring.

## Evaluation and observed behavior

The synthetic smoke test currently reports precision 1.0, recall 0.8, F1 0.8889, and false-positive rate 0.0 over eight examples. On the held-out Abt-Buy split, TF-IDF had the best reported F1 (0.2913) at the selected high-precision operating points; the logistic hybrid reached 0.2581 and did not beat it. Candidate recall@10 was 0.9786. Regression tests cover normalization, abbreviated model identifiers, XL/base variant conflicts, package conflicts, API validation, batch limits, review persistence, and every dashboard page.

## Limitations and biases

The food demo is small and its thresholds are not calibrated on food labels. The Abt-Buy split is disjoint only on the left entity, so right-side reuse can make evaluation less realistic. Lexical methods favor products with similar naming and may under-rank aliases or multilingual descriptions. Simple suffix compatibility can still miss unusual manufacturer naming conventions. Scores are ranking signals, not probabilities; uncertain cases must remain reviewable.

## Intended and non-intended use

Intended uses are analyst assistance, catalog deduplication experiments, candidate prioritization, and educational evaluation. Non-intended uses include autonomous deletion or merging, safety-critical inventory decisions, cross-domain accuracy claims, and decisions about people. Human review is required when errors could cause material harm.

## Improvements

Next steps are to label the food domain and calibrate its thresholds, enforce graph-disjoint benchmark splits on both entity sides, add multilingual and alias-aware training data, monitor accuracy by category and brand, and persist reviewer corrections in durable storage. Reviewer feedback could later support retraining, but only with versioning and audit controls.
