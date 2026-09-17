# Experiment 024 — Two-Stage Cross-Encoder Neural Re-Ranking Benchmark

**Date:** 2026-09-17  
**Status:** ✅ Completed

---

## 1. Research Question

> **Does adding a Stage-2 Cross-Encoder neural re-ranker (`ms-marco-MiniLM-L-6-v2`) on top of a Stage-1 candidate pool ($K_{\text{cand}}=50$) improve top-K ranking quality ($n\text{DCG}@5$, $n\text{DCG}@10$, $\text{MAP}@5$, $\text{MRR}$) compared with single-stage BM25, Dense, and Hybrid RRF baselines on SciFact?**

In a Two-Stage Retrieval pipeline, the Cross-Encoder operates exclusively as a **re-ranker of existing candidates**; it computes token-to-token cross-attention over `(query, document)` sequence pairs to re-score candidates, but cannot recover relevant documents that were missing from the Stage-1 candidate pool. This experiment evaluates whether this neural re-ranking step translates into measurable ranking improvements over first-stage retrieval baselines on out-of-domain scientific text.

---

## 2. Experiment Setup

| Property | Configuration |
|---|---|
| **Benchmark** | BEIR SciFact |
| **Queries** | 300 test queries |
| **Corpus** | 5,183 scientific abstracts |
| **Location** | `data/beir/scifact/` |
| **Candidate Depth ($K_{\text{cand}}$)** | 50 per retriever (operating point from Exp 022) |
| **Evaluated Systems** | 6 systems (3 Single-Stage Baselines, 3 Two-Stage Re-Rankers) |
| **Neural Model** | `cross-encoder/ms-marco-MiniLM-L-6-v2` (`max_length=256`, `batch_size=64`) |
| **Hardware** | NVIDIA GeForce GTX 1650 (4 GB VRAM) via PyTorch CUDA |
| **Metrics** | Recall@K, Precision@K, nDCG@K, MRR, MAP@K, Hit@K, Mean Latency (ms) |
| **Relevance** | Graded judgments from SciFact `qrels` |

---

## 3. Results

### Summary Comparison Table (@5)

| System | Recall@5 | Precision@5 | nDCG@5 | MRR | MAP@5 | Hit@5 | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| **BM25 Single-Stage** | 0.7243 | 0.1567 | 0.6438 | 0.6339 | 0.6108 | 0.7467 (74.7%) | 40.4 ms |
| **FAISS Dense Single-Stage** | 0.7686 | 0.1707 | **0.6936** | **0.6847** | **0.6637** | 0.7867 (78.7%) | **13.3 ms** |
| **Hybrid RRF (1:2) Single-Stage** | **0.7777** | **0.1713** | 0.6927 | 0.6798 | 0.6596 | **0.7967 (79.7%)** | 37.2 ms |
| **Two-Stage: BM25 ($K=50$) + CE** | 0.7337 | 0.1587 | 0.6527 | 0.6464 | 0.6181 | 0.7600 (76.0%) | 314.2 ms |
| **Two-Stage: Dense ($K=50$) + CE** | 0.7535 | 0.1653 | 0.6704 | 0.6639 | 0.6360 | 0.7767 (77.7%) | 494.3 ms |
| **Two-Stage: Union ($K=50$) + CE** | 0.7452 | 0.1640 | 0.6626 | 0.6580 | 0.6277 | 0.7700 (77.0%) | 1273.9 ms |

### Summary Comparison Table (@10)

| System | Recall@10 | Precision@10 | nDCG@10 | MAP@10 | Hit@10 |
|---|---:|---:|---:|---:|---:|
| **BM25 Single-Stage** | 0.7816 | 0.0867 | 0.6646 | 0.6211 | 0.8000 (80.0%) |
| **FAISS Dense Single-Stage** | **0.8452** | **0.0953** | **0.7203** | **0.6766** | **0.8567 (85.7%)** |
| **Hybrid RRF (1:2) Single-Stage** | 0.8429 | 0.0943 | 0.7150 | 0.6701 | **0.8567 (85.7%)** |
| **Two-Stage: BM25 ($K=50$) + CE** | 0.7763 | 0.0883 | 0.6676 | 0.6258 | 0.7933 (79.3%) |
| **Two-Stage: Dense ($K=50$) + CE** | 0.8196 | 0.0940 | 0.6927 | 0.6469 | 0.8333 (83.3%) |
| **Two-Stage: Union ($K=50$) + CE** | 0.8082 | 0.0927 | 0.6838 | 0.6381 | 0.8233 (82.3%) |

### Pairwise Deltas vs. Baselines (@5)

- **Two-Stage BM25 + CE vs. Single-Stage BM25:**
  - $n\text{DCG}@5$: **$+0.0089$** ($0.6438 \rightarrow 0.6527$)
  - $\text{Recall}@5$: **$+0.94\%p$** ($72.43\% \rightarrow 73.37\%$)
  - $\text{MRR}$: **$+0.0125$** ($0.6339 \rightarrow 0.6464$)
  - $\text{Hit}@5$: **$+1.33\%p$** ($74.67\% \rightarrow 76.00\%$)

- **Two-Stage Union + CE vs. Single-Stage Hybrid RRF (1:2):**
  - $n\text{DCG}@5$: **$-0.0301$** ($0.6927 \rightarrow 0.6626$)
  - $\text{Recall}@5$: **$-3.25\%p$** ($77.77\% \rightarrow 74.52\%$)
  - $\text{MRR}$: **$-0.0218$** ($0.6798 \rightarrow 0.6580$)
  - $\text{MAP}@5$: **$-0.0319$** ($0.6596 \rightarrow 0.6277$)
  - $\text{Hit}@5$: **$-2.67\%p$** ($79.67\% \rightarrow 77.00\%$)

- **Two-Stage Union + CE vs. Single-Stage FAISS Dense:**
  - $n\text{DCG}@5$: **$-0.0310$** ($0.6936 \rightarrow 0.6626$)
  - $\text{Recall}@5$: **$-2.34\%p$** ($76.86\% \rightarrow 74.52\%$)

---

## 4. What Happened?

### 1. Cross-Encoder Successfully Re-Ranks BM25 Candidates
When applied to candidates retrieved strictly by BM25 ($K_{\text{cand}}=50$), the Cross-Encoder produced consistent improvements across all ranking metrics:
- $n\text{DCG}@5$ improved from **$0.6438$ to $0.6527$** ($+0.0089$).
- $\text{Recall}@5$ increased from **$72.4\%$ to $73.4\%$** ($+0.94\%p$).
- $\text{MRR}$ increased from **$0.6339$ to $0.6464$** ($+0.0125$).

The Cross-Encoder changed the ordering of BM25 candidates, producing the measured improvement.

### 2. Dense and Hybrid Single-Stage Baselines Outperform Cross-Encoder
In contrast to the BM25 candidate pool, applying the Cross-Encoder to Dense or Union candidate pools did not yield higher metrics than the first-stage dense rankings:
- Single-Stage FAISS Dense achieved **$0.6936\text{ nDCG}@5$**, compared with **$0.6704$** for Dense + CE ($-0.0232$) and **$0.6626$** for Union + CE ($-0.0310$).
- Single-Stage Hybrid RRF achieved **$0.6927\text{ nDCG}@5$**, outperforming Union + CE by $+0.0301$.

### 3. Cross-Encoder Re-Ranks, But Cannot Exceed Candidate Pool Limits
The Cross-Encoder only re-orders what Stage 1 provides. While Experiment 022 demonstrated that the Multi-Retriever candidate pool holds a theoretical $96.2\%$ recall ceiling at $K=50$, the Cross-Encoder placed relevant documents in the top 5 for $74.5\%$ of queries, falling short of single-stage Hybrid RRF ($77.8\%$) and FAISS Dense ($76.9\%$).

---

## 5. Complementarity & Diagnostics Analysis

### Query-Level Win/Loss Distribution on $n\text{DCG}@5$

Comparing **Two-Stage Union ($K=50$) + Cross-Encoder** directly against **Single-Stage Hybrid RRF (1:2)** across all 300 test queries:

- **Two-Stage Wins:** 41 queries (13.7%)
- **Hybrid RRF Wins:** 57 queries (19.0%)
- **Ties:** 202 queries (67.3%)

In 41 queries, the Cross-Encoder successfully promoted a relevant document higher than RRF. However, in 57 queries, the Cross-Encoder placed a relevant document lower in rank compared to RRF's first-stage position, resulting in a net negative delta on aggregate metrics.

### Latency Profiles Across Architectures

| System | Mean Latency per Query | Throughput (QPS) | Relative Overhead vs. FAISS |
|---|---:|---:|---:|
| **FAISS Dense Single-Stage** | **13.3 ms** | 75.2 QPS | $1.0\times$ (Baseline) |
| **BM25 Single-Stage** | 40.4 ms | 24.8 QPS | $3.0\times$ |
| **Hybrid RRF (1:2)** | 37.2 ms | 26.9 QPS | $2.8\times$ |
| **Two-Stage BM25 + CE** | 314.2 ms | 3.2 QPS | $23.6\times$ |
| **Two-Stage Dense + CE** | 494.3 ms | 2.0 QPS | $37.2\times$ |
| **Two-Stage Union + CE** | 1,273.9 ms | 0.8 QPS | $95.8\times$ |

Neural re-ranking introduced substantial compute overhead: re-ranking the ~85.5 deduplicated candidates per query in the Union pool required **1,273.9 ms per query** on the GPU (averaging 0.8 QPS), compared with **13.3 ms** for FAISS Dense and **37.2 ms** for Hybrid RRF.

---

## 6. Methodological Deep-Dive: Hypothesizing the Performance Gap

### Why Did the Cross-Encoder Underperform Dense Baselines?

While this experiment demonstrates the empirical gap, determining the exact causal mechanism requires further research. However, a primary plausible hypothesis involves **training domain mismatch**:

1. **Pre-Training Distribution vs. Target Distribution**:
   - `cross-encoder/ms-marco-MiniLM-L-6-v2` was trained on MS MARCO, which consists of general-domain Bing search queries and informal web passages.
   - SciFact consists of specialized biomedical scientific claims evaluated against formal research abstracts containing heavy domain-specific nomenclature (e.g. gene abbreviations, chemical assays, pharmacological interactions).
2. **Dense Embeddings Alignment**:
   - The dense retriever utilized `bge-small-en-v1.5` (via FastEmbed), which was trained on diverse retrieval tasks and large-scale academic corpora.
   - It is plausible that the dense embedding space captured scientific abstract associations more effectively than the web-trained cross-encoder, leading the cross-encoder to demote technical evidence passages that lacked colloquial web phrasing.

---

## 7. Key Findings

### Finding 1 — Cross-Encoder Improves Lexical Baselines
> Re-ranking BM25 candidates with the Cross-Encoder improved $n\text{DCG}@5$ from **$0.6438$ to $0.6527$** ($+0.0089$) and $\text{Recall}@5$ from **$72.4\%$ to $73.4\%$** ($+0.94\%p$).

### Finding 2 — Single-Stage Dense and Hybrid Outperform the Two-Stage Cross-Encoder
> Single-stage FAISS Dense achieved **$0.6936\text{ nDCG}@5$** and Hybrid RRF achieved **$0.6927\text{ nDCG}@5$**, both outperforming the Two-Stage Union + Cross-Encoder (**$0.6626\text{ nDCG}@5$**).

### Finding 3 — High Query-Level Divergence
> Across 300 test queries, the Cross-Encoder outperformed Hybrid RRF on 41 queries (13.7%), but was outperformed on 57 queries (19.0%), with 202 ties (67.3%).

### Finding 4 — Substantial Neural Re-Ranking Latency Overhead
> Two-Stage Union re-ranking averaged **1,273.9 ms per query** on GPU, representing a **$95.8\times$ latency increase** compared to FAISS Dense single-stage search (**13.3 ms**).

---

## 8. What This Means for System Architecture & Next Steps

These empirical findings provide clear guidance for the next phase of Sprint 3:

1. **Off-the-shelf Web Cross-Encoders Cannot Be Assumed Superior**:
   Deploying a general web cross-encoder on specialized technical benchmarks without in-domain fine-tuning carries risk of ranking degradation relative to dense retrieval baselines.

2. **Implications for RLB-331 (LightGBM LambdaMART)**:
   The upcoming **RLB-331** experiment will evaluate tabular Learning-to-Rank trained on our 16 extracted signals (`LTRDataset`).
   - Unlike the zero-shot web Cross-Encoder, LightGBM will train directly on SciFact relevance judgments.
   - Furthermore, LightGBM includes retrieval scores (`dense_score`, `rrf_score`) directly as input features, allowing the model to preserve first-stage dense signals rather than replacing them entirely.
   - Whether LightGBM will outperform single-stage baselines remains an open empirical question to be tested.

```text
First-Stage Retrieval:
  Query ──► MultiRetrieverCandidateGenerator (K=50) ──► CandidatePool (~85.5 items)
                                                            │
Second-Stage Scoring Options:                               │
  ├─ Option A (Exp 024): Cross-Encoder (Zero-shot Web) ─────┼──► 0.6626 nDCG@5 (1,273.9 ms)
  ├─ Option B (Exp 021): Single-Stage Hybrid RRF Baseline ──┼──► 0.6927 nDCG@5 (37.2 ms)
  └─ Option C (RLB-331): LightGBM LambdaMART (In-Domain) ───┴──► To Be Evaluated
```
