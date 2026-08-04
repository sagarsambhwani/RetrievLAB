# Project Guidelines for RetrievLab

## Design Principle: Algorithms vs Configurations

RetrievLab compares **retrieval strategies**, not duplicate implementations.

- **Rule 1 — One algorithm, one implementation**: Each retrieval algorithm should have a single implementation (e.g. `BM25Retriever`, `DenseRetriever`, `HybridRetriever`, `FAISSRetriever`). Do not create multiple files/classes for variations like `bm25_stemmed.py`.
- **Rule 2 — Behavior should be configurable**: Preprocessing or parameter tuning should be injected as configuration (e.g., passing `tokenizer` to `BM25Retriever`), leaving the core retrieval algorithm untouched.
- **Rule 3 — Experiments compare configurations**: Compare different configurations of the same algorithm (e.g. Basic vs +Stopwords vs +Stemming vs +Tuned Parameters).
- **Rule 4 — Introduce a new retriever only for fundamentally different approaches**: Create new retrievers only when the paradigm itself changes (e.g. BM25 vs Dense vs Hybrid/RRF vs FAISS vs SPLADE vs ColBERT).
- **Rule 5 — Preserve experimental reproducibility**: Never overwrite existing configurations. Always store configuration, benchmark, evaluation metrics, and experiment reports.

> **Guiding Principle**: Implement algorithms once. Explore behavior through configuration. Validate improvements through experiments.
