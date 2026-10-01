# ADR-0009: Candidate Pool Generation & Provenance Architecture

**Status**: Accepted  
**Deciders**: RetrievLab Team  
**Date**: 2026-09-17  

---

## Context

In single-stage retrieval pipelines (Sprint 1 and Sprint 2), retrievers return a ranked list of top-$K$ `SearchResult` objects directly to the caller. However, modern high-precision search pipelines require a **two-stage architecture**:
1. **Stage 1 (Candidate Generation)**: Fast, high-recall retrieval across multiple complementary modalities (e.g. sparse BM25 and dense embedding search) to gather a candidate pool.
2. **Stage 2 (Re-Ranking)**: Computationally intensive scoring (e.g. neural cross-encoders or gradient-boosted decision trees) applied strictly to the pre-filtered candidate pool.

Prior to Sprint 3, RetrievLab lacked a formal abstraction to represent candidate pools. Re-rankers require not only the underlying text chunks, but also the **retrieval provenance** of each candidate (which retriever found it, what score was assigned, and what rank position it occupied in Stage 1). Furthermore, the candidate pool depth ($K_{\text{cand}}$) directly controls the trade-off between the theoretical recall ceiling and downstream re-ranking latency.

---

## Decision

Introduce a dedicated candidate selection subsystem under `src/retrievlab/selection/`:
1. Define **`Candidate`** as a Pydantic model encapsulating the raw `Chunk`, along with multi-retriever provenance dictionaries (`retriever_scores: dict[str, float]`, `retriever_ranks: dict[str, int]`, and `sources: list[str]`).
2. Define **`CandidatePool`** as a container for query metadata and deduplicated `Candidate` instances.
3. Define the abstract base interface **`CandidateGenerator`** in `src/retrievlab/selection/interface.py`.
4. Implement **`SingleRetrieverCandidateGenerator`** and **`MultiRetrieverCandidateGenerator`** in `src/retrievlab/selection/generator.py` to orchestrate candidate retrieval across arbitrary `Retriever` implementations.
5. Standardize the default operating depth at **$K_{\text{cand}} = 50$ per retriever** for multi-stage retrieval workflows.

---

## Architectural Principles & Implementation Details

1. **Provenance Preservation**:
   When multiple retrievers return the same document (e.g. both BM25 and FAISS dense retrieve chunk `c1`), `MultiRetrieverCandidateGenerator` deduplicates the candidate by `chunk.id` while preserving all modality scores and ranks:
   ```python
   candidate.retriever_scores["bm25"] = 18.45
   candidate.retriever_scores["dense"] = 0.882
   candidate.retriever_ranks["bm25"] = 2
   candidate.retriever_ranks["dense"] = 1
   candidate.sources = ["bm25", "dense"]
   ```
   This ensures downstream feature extractors and tabular rankers can leverage multi-retriever agreement without re-querying indices.

2. **Decoupled Selection Interface**:
   `CandidateGenerator` decouples candidate generation from both indexing and ranking. New candidate generation strategies (e.g., adaptive candidate pooling, threshold-based pruning, or multi-vector selection) can be introduced without modifying the core `Retriever` or `ReRanker` interfaces.

3. **Controlled Cardinality ($K_{\text{cand}} = 50$)**:
   Downstream neural cross-encoders compute $O(L^2)$ transformer attention over candidate pairs. A union pool of $K=50$ per retriever bounds the candidate cardinality to $\le 100$ items (empirically averaging ~85.5 candidates per query on BEIR SciFact), preventing system RAM or GPU VRAM exhaustion.

---

## Validation & Experimental Findings ([`exp022_candidate_pool_recall_curve.md`](file:///e:/Downloads/RetrievLab/results/sprint_3/exp022_candidate_pool_recall_curve.md))

The depth recall curve was systematically evaluated across $K_{\text{cand}} \in [5, 10, 20, 50, 100, 200, 500]$ on BEIR SciFact ($N=300$ queries, 5,183 documents):
* **Recall Ceiling**: MultiRetriever Union at $K=50$ achieved **96.2% Recall** ceiling (Recall = `0.9620`), capturing almost every relevant document while maintaining an average pool size of only **85.5 candidates**.
* **Diminishing Returns**: Increasing $K_{\text{cand}}$ from 50 to 500 yielded only a +2.6%p recall gain (`0.9620` $\rightarrow$ `0.9880`) while increasing the downstream candidate volume by **6.7×** (85.5 to 572.3 candidates), which would unacceptably multiply Cross-Encoder inference latency.

---

## Consequences

### Positive
* Provides a standardized, type-safe data structure (`CandidatePool`) passed between Stage 1 and Stage 2.
* Downstream feature extractors have immediate access to multi-modal provenance metadata without recomputing Stage-1 scores.
* $K_{\text{cand}} = 50$ provides an empirically verified Pareto operating point between recall potential and computational cost.

### Tradeoffs / Considerations
* Candidate pool generation incurs a small CPU overhead (~34 ms/query) to query multiple retrievers and merge dictionaries.
* Fixed depth $K_{\text{cand}}$ does not adapt dynamically to query ambiguity; queries with few strong matches still extract 50 candidates per retriever.
