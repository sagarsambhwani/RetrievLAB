# 🏛️ Design Principle: Algorithms vs Configurations

RetrievLab should compare **retrieval strategies**, not duplicate implementations.

## Rule 1 — One algorithm, one implementation

Each retrieval algorithm should have a single implementation.

Examples:

- `BM25Retriever`
- `DenseRetriever`
- `HybridRetriever`
- `FAISSRetriever`

Avoid creating multiple versions of the same algorithm such as:

```text
bm25.py
bm25_v2.py
bm25_stemmed.py
bm25_stopwords.py
```

---

## Rule 2 — Behavior should be configurable

Changes such as preprocessing or parameter tuning should be injected as configuration rather than implemented as new retrievers.

Example:

```python
BM25Retriever(
    tokenizer=BasicTokenizer()
)

BM25Retriever(
    tokenizer=StemmedTokenizer()
)

BM25Retriever(
    tokenizer=StopwordTokenizer()
)
```

The retrieval algorithm remains identical; only the preprocessing pipeline changes.

---

## Rule 3 — Experiments compare configurations

Every experiment should compare different configurations of the same algorithm.

Example:

```text
BM25
├── Basic
├── + Stopwords
├── + Stemming
└── + Tuned Parameters
```

The goal is to answer questions such as:

- Does stemming improve retrieval?
- Does stopword removal help?
- Which BM25 parameters perform best?

---

## Rule 4 — Introduce a new retriever only for a fundamentally different approach

Create a new retriever only when the retrieval method itself changes.

Examples:

- BM25
- Dense Retrieval
- Hybrid Retrieval (RRF)
- FAISS Retrieval
- SPLADE
- ColBERT

These represent different retrieval paradigms and therefore deserve separate implementations.

---

## Rule 5 — Preserve experimental reproducibility

Never overwrite a previous configuration.

Instead, store:

- configuration
- benchmark
- evaluation metrics
- experiment report

This allows future comparisons across datasets and retrieval strategies.

---

## Core Principle: Build vs Use (Infrastructure & Algorithms)

- **Implement research infrastructure and simple algorithms yourself**: Build evaluation harnesses, benchmark loaders, chunkers, and baseline algorithms from scratch to ensure complete transparency, modularity, and first-principles understanding.
- **Use established libraries for mature algorithms and models**: Use battle-tested, optimized libraries (e.g., FAISS, FastEmbed, LightGBM, Hugging Face) for mature models and production-grade indexing/ranking algorithms.
- **Understand library assumptions**: When utilizing external libraries, understand what the library is doing under the hood, what assumptions it makes (e.g., vector normalization, metric spaces, tokenization defaults), and how those assumptions impact experimental results.

---

## Guiding Principle

> **Implement algorithms once. Explore behavior through configuration. Validate improvements through experiments.**

