# ADR-0002: Tokenizer Abstraction & Stemming Architecture

**Status**: Accepted  
**Deciders**: RetrievLab Team  
**Date**: 2026-08-14  

---

## Context
Lexical retrieval algorithms (e.g., BM25) rely heavily on text normalization and tokenization. 
In Sprint 1, `BM25Retriever` hardcoded regex-based word extraction. To explore the impact of stopword filtering, n-grams, subwords, and stemming, we needed a clean architectural pattern that follows our core design principles:
1. *One algorithm, one implementation*: Keep `BM25Retriever` untouched.
2. *Behavior should be configurable*: Inject tokenization strategy via dependency injection.
3. *Build vs. Use*: Build the experimentation infrastructure and tokenization pipelines in-house; use established, optimized libraries for mature linguistic algorithms like stemming.

## Decision
1. Introduce abstract base class `BaseTokenizer` in `retrievlab.preprocessing.interface` defining `tokenize(text: str) -> list[str]`.
2. Inject `tokenizer: BaseTokenizer` into `BM25Retriever` initialization.
3. Implement `StemmedTokenizer` by wrapping the high-performance, mature `py_rust_stemmers` (Snowball / Porter2) library instead of maintaining a 350-line procedural stemmer from scratch.

## External Library Assumptions & Impact on Experiments
- **Algorithm**: Snowball (Porter2) English stemmer.
- **Linguistic Model**: Heuristic suffix stripping based on vowel/consonant measure regions ($R_1, R_2$).
- **Assumptions & Caveats**:
  - *No Part-of-Speech tagging*: Operates solely on token surface forms without context.
  - *Over-stemming risk*: Words with different semantic meanings may collapse to the same stem (e.g., `experiment` vs `experience` $\rightarrow$ `experi`), causing potential false positive collisions and reducing Precision@K.
  - *Under-stemming risk*: Highly irregular or non-affix morphological variations (e.g., `go` vs `went`) will not match, setting a ceiling on Recall@K for irregular forms.

## Consequences
### Positive
- `BM25Retriever` is completely decoupled from tokenization details.
- High-speed Rust-backed Snowball stemmer eliminates execution bottlenecks during indexing.
- Clean experimental matrix comparing `BasicWordTokenizer`, `StopwordTokenizer`, and `StemmedTokenizer`.

### Negative / Tradeoffs
- Adds a lightweight dependency on `py_rust_stemmers` (or standard `snowballstemmer`).
