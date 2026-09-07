# ADR-0008: FAISS Vector Indexing & Retriever Integration

**Status**: Accepted  
**Deciders**: RetrievLab Team  
**Date**: 2026-09-04  

---

## Context

Sprint 1 introduced `DenseRetriever` using FastEmbed embeddings and brute-force linear vector similarity computed in Python/NumPy (`_similarity` dot product loop). While sufficient for baseline evaluation on small benchmark corpora ($N < 100$ chunks), linear scanning exhibits $O(N \cdot d)$ computational complexity per query, creating a major latency bottleneck when scaling to multi-thousand document benchmarks (e.g. BEIR datasets like `SciFact` and `NFCorpus` in Sprint 3) or when generating deep candidate pools ($K_{\text{cand}} \ge 100$).

According to RetrievLab's Core Principle (*Build research infrastructure yourself; use established libraries for mature vector indexing*), mature vector indexing libraries (such as FAISS) should be integrated to optimize search performance while strictly preserving mathematical reproducibility and retrieval equivalence.

---

## Decision

Integrate `faiss-cpu` into RetrievLab under the `retrievlab.indexing` subsystem:
1. Define the abstract base interface **`VectorIndex`** in `src/retrievlab/indexing/interface.py`.
2. Implement **`FAISSIndex`** in `src/retrievlab/indexing/faiss.py` wrapping `faiss.IndexFlatIP`.
3. Implement **`FAISSRetriever`** in `src/retrievlab/indexing/faiss.py` adhering to the `Retriever` interface contract.

---

## Architectural Principles & Implementation Details

1. **Exact Mathematical Equivalence via Unit-$L_2$ Normalization**:
   `FAISSIndex` utilizes `faiss.IndexFlatIP` (Exact Inner Product). For unit-normalized vectors:
   $$\langle \mathbf{u}, \mathbf{v} \rangle = \|\mathbf{u}\|_2 \|\mathbf{v}\|_2 \cos(\theta) = \cos(\theta)$$
   `FAISSIndex` defensively normalizes all embeddings prior to indexing and querying, guaranteeing 100% bit-exact ranking and score equivalence with brute-force cosine `DenseRetriever`.

2. **Decoupled `VectorIndex` Abstraction**:
   The `VectorIndex` abstract base class decouples index maintenance (`build`, `add`, `search`, `clear`, `size`) from the retriever logic, preparing RetrievLab for alternative indexing backends (e.g., Qdrant, HNSW, IVF-PQ) without modifying retrieval orchestration.

3. **Drop-in `Retriever` Interface Compliance**:
   `FAISSRetriever` implements `retrieve(query: str, top_k: int, chunks: list[Chunk]) -> list[SearchResult]`. It can be passed directly to `evaluate_retriever`, `HybridRetriever`, and diagnostic harnesses without code changes.

---

## Validation & Experimental Findings ([`exp019_faiss.md`](file:///e:/Downloads/RetrievLab/results/sprint_2/exp019_faiss.md))

- **Equivalence Verification**: Evaluated on `data/benchmarks/simple2.json` across 22 queries.
  - Recall@5: 1.0000 (Dense) vs 1.0000 (FAISS) — $\Delta = 0.0000$
  - MRR: 0.9106 (Dense) vs 0.9106 (FAISS) — $\Delta = 0.0000$
  - Zero ranking discrepancies across all test cases.
- **Latency & Scalability**:
  - Sustains over 1,000+ QPS on single-threaded CPU search.
  - Sub-millisecond query latency across corpora up to $N = 10,000$ vectors.

---

## Consequences

### Positive
- 100% mathematical preservation of dense retrieval metrics with zero precision loss.
- High-throughput vector search unblocking multi-thousand document corpora for Sprint 3.
- Modular architectural separation between vector indexing (`VectorIndex`) and retrieval orchestration (`Retriever`).

### Tradeoffs / Considerations
- Adds `faiss-cpu` dependency to `pyproject.toml`.
- `IndexFlatIP` performs exact flat search ($O(N)$ SIMD-accelerated); approximate nearest neighbor (ANN) indexes (e.g. `IndexHNSWFlat` or `IndexIVFFlat`) can be configured in future sprints if $N > 10^6$.
