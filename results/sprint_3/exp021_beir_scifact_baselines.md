# Experiment 021 — BEIR SciFact Baselines & Hybrid Rank Fusion

**Date:** 2026-09-15  
**Status:** ✅ Completed

---

## 1. Research Question

> **Does hybrid retrieval improve out-of-domain retrieval quality on SciFact compared with standalone BM25 and Dense retrieval?**

We also investigate whether RRF successfully combines lexical precision with semantic recall, using both binary and graded relevance metrics.

---

## 2. Experiment Setup

| Property | Configuration |
|---|---|
| **Benchmark** | BEIR SciFact |
| **Queries** | 300 test queries |
| **Corpus** | 5,183 scientific abstracts |
| **Location** | `data/beir/scifact/` |
| **Evaluation Cutoffs** | @5, @10 |
| **Retrievers** | BM25, FAISS Dense |
| **Hybrid Methods** | RRF 1:1, RRF 1:2 Dense-biased |
| **Metrics** | Recall, Precision, MRR, nDCG, MAP, Hit |
| **Relevance** | Graded |

---

## 3. Results

### @5 Results

| System | Recall@5 | MRR | Precision@5 | nDCG@5 | MAP@5 | Hit@5 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.7243 | 0.6261 | 0.1567 | 0.6438 | 0.6108 | 0.7467 |
| FAISS Dense | 0.7686 | 0.6752 | 0.1707 | 0.6936 | 0.6637 | 0.7867 |
| Hybrid RRF 1:1 | 0.7720 | 0.6697 | 0.1700 | 0.6899 | 0.6578 | 0.7933 |
| Hybrid RRF 1:2 | 0.7777 | 0.6722 | 0.1713 | 0.6927 | 0.6596 | 0.7967 |

### @10 Results

| System | Recall@10 | MRR | Precision@10 | nDCG@10 | MAP@10 | Hit@10 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.7816 | 0.6339 | 0.0867 | 0.6646 | 0.6211 | 0.8000 |
| FAISS Dense | 0.8452 | 0.6847 | 0.0953 | 0.7203 | 0.6766 | 0.8567 |
| Hybrid RRF 1:1 | 0.8606 | 0.6807 | 0.0960 | 0.7200 | 0.6715 | 0.8767 |
| Hybrid RRF 1:2 | 0.8429 | 0.6798 | 0.0943 | 0.7150 | 0.6701 | 0.8567 |

---

## 4. What Happened?

### Lexical vs. Semantic Retrieval

This experiment evaluates how BM25 and FAISS Dense retrieval behave on scientific literature (SciFact).

**Observed result:**

> FAISS Dense achieved **76.9% Recall@5** compared with **72.4%** for BM25, a difference of **+4.4 percentage points**. In graded ranking, Dense achieved **0.6936 nDCG@5** versus **0.6438** for BM25 (+0.0499 difference).

### Hybrid Retrieval

RRF combines the rankings produced by BM25 and Dense retrieval.

We compare:

- **1:1 Balanced RRF**
- **1:2 Dense-biased RRF**

**Observed result:**

> Hybrid RRF 1:1 achieved **77.2% Recall@5** (0.6899 nDCG@5). Hybrid RRF 1:2 achieved **77.8% Recall@5** (0.6927 nDCG@5) and **84.3% Recall@10** (0.7150 nDCG@10).

---

## 5. Complementarity Analysis

The aggregate metrics tell us **what** happened. Query-level analysis helps explain **why**.

### Binary retrieval complementarity at K=5 (based on Hit@5):

- **Both Succeed**: 201 queries (67.0%)
- **Dense Succeeds & BM25 Fails**: 35 queries (11.7%)
- **BM25 Succeeds & Dense Fails**: 23 queries (7.7%)
- **Both Fail**: 41 queries (13.7%)

**Key observation:**

> Across the 300 test queries, Dense uniquely retrieved a relevant document in the top 5 for 35 queries where BM25 missed, while BM25 uniquely retrieved a relevant document for 23 queries where Dense missed. Combining both candidate streams yields a theoretical union Hit@5 ceiling of **86.3%** (259 / 300 queries).

---

## 6. Graded Relevance

Unlike our original binary benchmark, SciFact provides graded relevance judgments.

This allows us to distinguish:

> finding a relevant document

from:

> ranking the **most relevant** documents highly.

Therefore:

- **Recall@K** measures retrieval coverage.
- **nDCG@K** measures ranking quality with graded relevance.
- **MAP@K** measures precision across relevant-document positions.
- **MRR** emphasizes the first relevant result.
- **Precision@K** measures relevant results within the cutoff.
- **Hit@K** measures whether at least one relevant result was retrieved.

**Observed result:**

> At K=5, Dense nDCG@5 is **0.6936** and MAP@5 is **0.6637**, compared with BM25 nDCG@5 of **0.6438** and MAP@5 of **0.6108**. The ranking metrics track the relative order observed in Recall@5 (76.9% vs 72.4%).

---

## 7. Key Findings

### Finding 1 — BM25
> On SciFact, BM25 achieved 72.4% Recall@5, 0.6438 nDCG@5, and 0.6261 MRR.

### Finding 2 — Dense Retrieval
> FAISS Dense retrieval (BGE-small) achieved 76.9% Recall@5, 0.6936 nDCG@5, and 0.6752 MRR.

### Finding 3 — Hybrid Retrieval
> Hybrid RRF 1:1 scored 77.2% Recall@5 (0.6899 nDCG@5), while Hybrid RRF 1:2 scored 77.8% Recall@5 (0.6927 nDCG@5).

### Finding 4 — RRF Weighting
> Increasing the Dense retriever weight from 1.0 to 2.0 shifted Recall@5 from 77.2% to 77.8% and nDCG@5 from 0.6899 to 0.6927.

---

## 8. What This Means for Two-Stage Retrieval

This experiment establishes the **Stage-1 retrieval baseline** for the upcoming two-stage system.

The architecture is:

```text
Query
  ↓
BM25 + FAISS Dense
  ↓
Hybrid Candidate Generation
  ↓
Candidate Pool
  ↓
Stage-2 Reranker
```
