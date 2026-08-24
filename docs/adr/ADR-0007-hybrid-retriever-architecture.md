# ADR-0007: Hybrid Retriever Architecture & Candidate Generation Orchestration

**Status**: Accepted  
**Deciders**: RetrievLab Team  
**Date**: 2026-08-24  

---

## Context
In Sprint 1 and Sprint 2 experiments ([`exp010`](file:///e:/Downloads/RetrievLab/results/sprint_1/exp010_query_analysis.md), [`exp016`](file:///e:/Downloads/RetrievLab/results/sprint_2/exp013_rrf.md)), we established that lexical retrieval (BM25) and dense semantic retrieval (FastEmbed BGE) exhibit orthogonal error patterns:
- BM25 excels at exact keyword matching, code identifiers, and rare tokens, but fails when queries use paraphrases without lexical overlap.
- Dense retrieval excels at semantic similarity and synonym matching, but can miss exact keyword constraints.

In **RLB-210**, we implemented **Reciprocal Rank Fusion (RRF)** as an unsupervised rank aggregation method. To make hybrid retrieval a first-class citizen in RetrievLab that works seamlessly with benchmark runners ([`evaluate_retriever`](file:///e:/Downloads/RetrievLab/src/retrievlab/evaluation/evaluate.py)) and downstream evaluation harnesses, we require a standardized `Retriever` implementation for hybrid candidate generation.

## Decision
Introduce **`HybridRetriever`** in `src/retrievlab/retrieval/hybrid.py` implementing the [`Retriever`](file:///e:/Downloads/RetrievLab/src/retrievlab/retrieval/interface.py) interface.

### Architectural Principles

1. **Interface Compliance & Uniformity**:
   `HybridRetriever` implements `Retriever.retrieve(query: str, top_k: int, chunks: list[Chunk]) -> list[SearchResult]`. Downstream evaluation harnesses, CLI commands, and RAG generation pipelines interact with `HybridRetriever` identically to single-modality retrievers.

2. **Orchestrator / Composite Pattern**:
   `HybridRetriever` accepts an arbitrary sequence of sub-retrievers (`retrievers: Sequence[Retriever]`, $N \ge 1$). This permits combining BM25 + Dense, or future multi-retriever ensembles (BM25 + Dense + SPLADE + FAISS).

3. **Decoupling Retrieval from Rank Fusion**:
   The fusion strategy is injected as a dependency (`fusion_strategy: ReciprocalRankFusion | None = None`), adhering to *Design Principles Rule 2 (Behavior should be configurable)*. It defaults to standard $k=60$ RRF, but allows custom smoothing constants, weights, or alternative rank aggregators.

4. **Candidate Pool Depth Management (`candidate_k`)**:
   Before performing rank fusion, each sub-retriever must retrieve a sufficiently deep candidate pool. `HybridRetriever` accepts an explicit `candidate_k: int | None = None` (defaulting to `max(top_k, 20)`), preventing truncation artifacts from starving the fusion step of relevant items ranked just outside `top_k`.

## Consequences

### Positive
- Unified, modular API adhering to RetrievLab's *One algorithm, one implementation* and *Configurable behavior* principles.
- Direct plug-and-play compatibility with `evaluate_retriever`, `BenchmarkSuite`, and `EvaluationReport`.
- Clean isolation between query execution/candidate generation and rank fusion math.

### Negative / Tradeoffs
- Multi-retriever querying multiplies execution latency by $N$ (or requires parallel execution in future scaling sprints).
- Sub-retrievers receive the full `chunks` list each query, which will be optimized in Sprint 2 Epic 4 when pre-indexed vector stores (FAISS) are connected.
