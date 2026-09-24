# Experiment 027 — Cascaded Coarse-to-Fine Re-Ranking Benchmark

**Date:** 2026-09-24  
**Status:** ✅ Completed

---

## 1. Research Question

> **Can a two-stage cascaded re-ranking architecture (LightGBM coarse filter $\to$ Cross-Encoder fine refiner) improve ranking quality and accelerate re-ranking latency compared with standalone neural Cross-Encoder scoring?**

Full neural Cross-Encoders evaluate complete `(query, document)` sequence cross-attention, which scales linearly in compute with the number of candidate documents ($O(N \cdot L^2)$). In contrast, tabular Gradient-Boosted Decision Trees (LightGBM LambdaMART) score candidates across multi-signal features in sub-millisecond time. This experiment investigates whether LightGBM can act as an effective intermediate coarse filter ($K_{\text{cand}}=50 \to K_{\text{inter}} \in [10, 30]$), pruning irrelevant distractors before invoking neural cross-attention.

---

## 2. Experiment Setup

| Property | Configuration |
|---|---|
| **Benchmark** | BEIR `SciFact` Test Split ($N=300$ queries) |
| **Corpus** | 5,183 scientific research abstracts (claim verification domain) |
| **Location** | `data/beir/scifact/` & `data/processed/scifact_embeddings.npy` |
| **Stage 1 Candidate Generator** | `MultiRetrieverCandidateGenerator` (BM25 + FAISS Dense `IndexFlatIP`, $K_{\text{cand}}=50$) |
| **Coarse Filter Model** | `LightGBMRanker` (LambdaMART, 16 multi-signal features, trained on 809 SciFact train queries) |
| **Fine Refiner Model** | `CrossEncoderReRanker` (`cross-encoder/ms-marco-MiniLM-L-6-v2`, sequence length $256$) |
| **Evaluated Systems** | 1. Single-Stage Hybrid RRF (1:2)<br/>2. Two-Stage Pure LightGBM ($K=50$)<br/>3. Two-Stage Pure Cross-Encoder ($K=50$)<br/>4. Cascades: LightGBM ($K=50 \to 10, 15, 20, 30$) $\to$ Cross-Encoder |
| **Metrics** | nDCG@10, nDCG@5, MRR, Recall@10, Precision@10, Pure Re-rank Latency (ms), Total Latency (ms) |
| **Relevance** | Graded Relevance Assessments (Qrels) |

---

## 3. Results

### Summary Comparison Table (SciFact Test Set, $N=300$ queries)

| System / Architecture | nDCG@10 | nDCG@5 | MRR | Recall@10 | Precision@10 | Rerank Latency (ms) | Total Latency (ms) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **Single-Stage: Hybrid RRF (1:2)** | 0.7150 | 0.6927 | 0.6798 | 0.8429 | 0.0943 | 0.00 | 37.92 |
| **Two-Stage: Pure LightGBM ($K=50$)** | **0.7617** | **0.7388** | **0.7257** | **0.8836** | **0.0993** | **26.50** | **61.78** |
| **Two-Stage: Pure Cross-Encoder ($K=50$)** | 0.6838 | 0.6626 | 0.6522 | 0.8082 | 0.0907 | 459.64 | 494.91 |
| **Cascade: LightGBM ($50 \to 10$) $\to$ Cross-Encoder** | 0.7194 | 0.6799 | 0.6726 | 0.8836 | 0.0993 | 80.24 | 115.51 |
| **Cascade: LightGBM ($50 \to 15$) $\to$ Cross-Encoder** | 0.7050 | 0.6731 | 0.6640 | 0.8551 | 0.0957 | 107.10 | 142.37 |
| **Cascade: LightGBM ($50 \to 20$) $\to$ Cross-Encoder** | 0.6988 | 0.6708 | 0.6602 | 0.8401 | 0.0943 | 133.97 | 169.24 |
| **Cascade: LightGBM ($50 \to 30$) $\to$ Cross-Encoder** | 0.6953 | 0.6719 | 0.6606 | 0.8294 | 0.0930 | 187.70 | 222.97 |

### Pareto Efficiency vs. Pure Cross-Encoder ($K=50$)

| Cascaded Configuration | nDCG@10 Delta | Re-rank Latency Speedup | End-to-End Latency Speedup |
|:---|:---:|:---:|:---:|
| **LightGBM ($50 \to 10$) $\to$ Cross-Encoder** | **+0.0355** | **5.73x** | **4.28x** |
| **LightGBM ($50 \to 15$) $\to$ Cross-Encoder** | +0.0212 | 4.29x | 3.48x |
| **LightGBM ($50 \to 20$) $\to$ Cross-Encoder** | +0.0150 | 3.43x | 2.92x |
| **LightGBM ($50 \to 30$) $\to$ Cross-Encoder** | +0.0114 | 2.45x | 2.22x |

---

## 4. What Happened?

### The Distractor Pruning Effect (Why Cascades Beat Pure Cross-Encoder)
When evaluating the full unpruned candidate pool ($K=50$, averaging $\sim 85$ candidates per query), the general-domain neural Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) achieved an nDCG@10 of `0.6838`. 

When pre-filtered by LightGBM to the top 10 candidates, the Cross-Encoder's nDCG@10 increased to `0.7194` (**+0.0355 gain, a +5.2% relative improvement**), while pure re-ranking latency dropped from $459.64\text{ ms}$ to $80.24\text{ ms}$ (**$5.73\times$ speedup**). 

The general-domain Cross-Encoder is susceptible to out-of-domain lexical and topical distractors when exposed to the full candidate pool. LightGBM's 16 multi-signal features (which combine BM25 exact matches, dense cosine similarity, rank discrepancy, and token overlap) effectively prune confusing negative distractors, presenting the neural refiner with a cleaner, higher-precision candidate set.

### Intermediate Cutoff Sensitivity ($K_{\text{inter}}$ Sweep)
As the intermediate cutoff was widened from $10 \to 15 \to 20 \to 30$:
1. **Quality Decreased Monotonically**: nDCG@10 dropped from `0.7194` ($K=10$) to `0.7050` ($K=15$), `0.6988` ($K=20$), and `0.6953` ($K=30$), approaching the lower bound of the unpruned Cross-Encoder (`0.6838`).
2. **Latency Scaled Linearly**: Pure re-ranking latency rose from $80.24\text{ ms}$ ($K=10$) to $187.70\text{ ms}$ ($K=30$), directly confirming that neural inference cost is strictly proportional to candidate depth.

---

## 5. Complementarity / Diagnostics Analysis

### Model Comparison Diagnostic at $K=10$ (SciFact Test Set):
- **Pure LightGBM vs. Pure Cross-Encoder**:
  - LightGBM strictly outperformed Pure Cross-Encoder across all metrics: nDCG@10 (`0.7617` vs `0.6838`, **+0.0779 delta**), MRR (`0.7257` vs `0.6522`), and Recall@10 (`0.8836` vs `0.8082`).
  - Re-rank latency for LightGBM was $26.50\text{ ms}$, compared with $459.64\text{ ms}$ for Cross-Encoder (**$17.3\times$ faster**).

- **Pure LightGBM vs. Cascaded LightGBM $\to$ Cross-Encoder ($K=10$)**:
  - Re-sorting LightGBM's top 10 candidates with Cross-Encoder resulted in an nDCG@10 of `0.7194`, which is `-0.0423` lower than Pure LightGBM (`0.7617`).
  - Recall@10 remained identical (`0.8836`), but the Cross-Encoder introduced local rank inversions within the top 5 (MRR dropped from `0.7257` to `0.6726`).

---

## 6. Methodological Deep-Dive

### Why In-Domain Tabular GBDT Outperforms Zero-Shot Neural Cross-Attention on SciFact
The experimental results demonstrate that `LightGBMRanker` trained directly on SciFact candidate features outperforms the off-the-shelf MS MARCO Cross-Encoder. This occurs because:
1. **Domain Alignment**: LightGBM was fitted on query-candidate pairs from the target domain, learning the exact relative value of exact keyword overlaps vs dense semantic similarity for scientific claims.
2. **Multi-Retriever Rank Agreement**: Features such as `retrieved_by_both` and `rank_discrepancy` provide explicit confidence signals that single-pass cross-attention cannot directly capture.
3. **Cross-Encoder Domain Gap**: `ms-marco-MiniLM-L-6-v2` was pre-trained on conversational Bing queries (MS MARCO). When applied zero-shot to complex scientific abstracts without domain fine-tuning, its token-level cross-attention over-weights peripheral semantic matches.

---

## 7. Key Findings

### Finding 1 — Cascading Strictly Dominates Pure Neural Cross-Encoding
> Pre-filtering candidates with LightGBM ($50 \to 10$) increased Cross-Encoder nDCG@10 by **+0.0355** while accelerating re-ranking throughput by **$5.73\times$** ($80.24\text{ ms}$ vs $459.64\text{ ms}$).

### Finding 2 — Smaller Intermediate Cutoffs Yield Higher Precision
> Across $K_{\text{inter}} \in [10, 15, 20, 30]$, $K=10$ delivered the highest nDCG@10 (`0.7194`) and lowest latency ($80.24\text{ ms}$). Increasing candidate depth introduced distractors that degraded neural ranking quality.

### Finding 3 — In-Domain Tabular LambdaMART Outperforms Off-the-Shelf Cross-Encoders
> Pure LightGBM LambdaMART established the strongest quality and efficiency benchmark on SciFact: **nDCG@10 of 0.7617 at $26.50\text{ ms}$ re-ranking latency**, outperforming single-stage Hybrid RRF by +0.0467 and pure neural Cross-Encoder by +0.0779.

---

## 8. What This Means for System Architecture

1. **LightGBM as the Default Production Ranker**: Where in-domain training data or click logs exist, tabular LambdaMART provides superior ranking accuracy at a fraction of the compute and memory cost of neural Transformers.
2. **Cascade Deployment Pattern**: For pipelines utilizing neural Cross-Encoders, candidate pools should never be passed directly to neural self-attention without prior tabular coarse filtering.
3. **Adaptive Routing Foundation**: Confident queries can terminate immediately after LightGBM scoring ($26.5\text{ ms}$), reserving cascaded Cross-Encoder evaluation ($80.2\text{ ms}$) strictly for borderline or high-discrepancy candidate sets.

```text
User Query
    │
    ▼
[Stage 1: Candidate Generation] ──► MultiRetriever (BM25 + FAISS Dense, K_cand=50) [~35 ms]
    │
    ▼
[Stage 2A: Coarse Filter]       ──► LightGBM LambdaMART (Scores 50 -> Prunes to Top 10) [~26 ms]
    │
    ├──► Confident Path (High Top-1 Margin) ──► Final SearchResults (Total: ~61 ms, nDCG@10: 0.7617)
    │
    └──► Refiner Path (Boundary / Ambiguous)  ──► Cross-Encoder (Scores Top 10) [~54 ms]
                                                  └──► Final SearchResults (Total: ~115 ms, nDCG@10: 0.7194)
```
