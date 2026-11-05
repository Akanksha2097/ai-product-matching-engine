# Methodology

Records are normalized conservatively: brand aliases are mapped, text is lowercased, identifiers remain distinct, and package quantities are extracted without removing variant information. Candidate generation uses TF-IDF n-grams, with optional brand/category boosts. The baseline is fuzzy string similarity; the lexical semantic-compatible model is TF-IDF cosine similarity; the hybrid model combines TF-IDF, fuzzy similarity, token overlap, brand compatibility, and a package-conflict penalty.

Scores rank candidates and are not calibrated probabilities. Thresholds are configuration defaults for the demo and should be selected on a validation set when a licensed labeled benchmark is added. Automated decisions favor precision; ambiguous cases go to review.
