# Project Guidelines for RetrievLab

## Core Principle: Build vs Use (Infrastructure & Algorithms)

- **Implement research infrastructure and simple algorithms yourself**: Build evaluation harnesses, benchmark loaders, chunkers, and baseline algorithms from scratch to ensure complete transparency, modularity, and first-principles understanding.
- **Use established libraries for mature algorithms and models**: Use battle-tested, optimized libraries (e.g., FAISS, FastEmbed, LightGBM, Hugging Face) for mature models and production-grade indexing/ranking algorithms.
- **Understand library assumptions**: When utilizing external libraries, understand what the library is doing under the hood, what assumptions it makes (e.g., vector normalization, metric spaces, tokenization defaults), and how those assumptions impact experimental results.

## Design Principle: Algorithms vs Configurations

RetrievLab compares **retrieval strategies**, not duplicate implementations.

- **Rule 1 — One algorithm, one implementation**: Each retrieval algorithm should have a single implementation (e.g. `BM25Retriever`, `DenseRetriever`, `HybridRetriever`, `FAISSRetriever`). Do not create multiple files/classes for variations like `bm25_stemmed.py`.
- **Rule 2 — Behavior should be configurable**: Preprocessing or parameter tuning should be injected as configuration (e.g., passing `tokenizer` to `BM25Retriever`), leaving the core retrieval algorithm untouched.
- **Rule 3 — Experiments compare configurations**: Compare different configurations of the same algorithm (e.g. Basic vs +Stopwords vs +Stemming vs +Tuned Parameters).
- **Rule 4 — Introduce a new retriever only for fundamentally different approaches**: Create new retrievers only when the paradigm itself changes (e.g. BM25 vs Dense vs Hybrid/RRF vs FAISS vs SPLADE vs ColBERT).
- **Rule 5 — Preserve experimental reproducibility**: Never overwrite existing configurations. Always store configuration, benchmark, evaluation metrics, and experiment reports.

> **Guiding Principle**: Implement algorithms once. Explore behavior through configuration. Validate improvements through experiments.

## Git Workflow & Safety Guidelines

- **Guiding Light**: Preserve First $\rightarrow$ Inspect Second $\rightarrow$ Propose Third $\rightarrow$ Ask Fourth $\rightarrow$ Change Last.
- **Granular Commits**: Commit individual files/units one-by-one using RLB semantic prefixes (`feat`, `exp`, `eval`, `docs`, `fix`, `test`, `perf`, `chore`).
- **Push at the End**: Defer `git push` until all individual local commits are completed and verified.
- **PowerShell Syntax**: Always use `;` or separate commands on Windows PowerShell; avoid `&&`.


