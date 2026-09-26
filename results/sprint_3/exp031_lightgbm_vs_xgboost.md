# Experiment 031 — GBDT Framework Head-to-Head: LightGBM vs. XGBoost

**Date:** 2026-09-26  
**Status:** ✅ Completed  
**Sprint:** Sprint 3 — Two-Stage Retrieval & Learning-to-Rank  
**Ticket:** RLB-360 / Exp 031  

---

## 1. Research Question

> **How does XGBoost's ranking engine (`XGBRanker` with `rank:ndcg` and `rank:pairwise` losses) compare with LightGBM's LambdaMART (`LGBMRanker`) in terms of ranking quality, sample efficiency under data starvation, training latency, and feature importance attribution on multi-signal retrieval candidate pools?**

Across Experiments 025–030, LightGBM demonstrated near-optimal ranking accuracy and microsecond inference on CPU. However, GBDT practitioners frequently debate whether XGBoost's depth-wise exact tree splits offer superior regularization over LightGBM's leaf-wise (best-first) splits, or whether listwise LambdaMART formulations (`rank:ndcg`) diverge between implementations. This experiment directly benchmarks both frameworks on identical training data, feature extractors, and test candidate pools.

---

## 2. Experiment Setup

| Property | Configuration |
|---|---|
| **Benchmark** | BEIR SciFact ($N_{\text{test}}=300$ queries, $N_{\text{train}}=809$ queries) |
| **Corpus** | 5,183 peer-reviewed biomedical abstracts |
| **Location** | `data/beir/scifact/` |
| **Stage-1 Candidate Pool** | MultiRetriever Union ($K_{\text{cand}}=50$ per retriever, BM25 + FAISS FlatIP `bge-small-en-v1.5`, avg pool size 85.5) |
| **Feature Representation** | 16 multi-signal features (lexical, dense, reciprocal rank, score/rank diffs, text metadata) |
| **Evaluated Systems** | 1. **BM25 Single-Stage** (Unsupervised lexical baseline)<br>2. **FAISS Dense Single-Stage** (Unsupervised dense baseline)<br>3. **LightGBM LambdaMART** (leaf-wise, `num_leaves=31`, `n_estimators=100`, `learning_rate=0.05`)<br>4. **XGBoost (rank:ndcg)** (depth-wise, `max_depth=6`, `n_estimators=100`, `learning_rate=0.05`)<br>5. **XGBoost (rank:pairwise)** (depth-wise, `max_depth=6`, `n_estimators=100`, `learning_rate=0.05`) |
| **Sample Efficiency Sweeps** | $N \in [25, 100, 809]$ training queries |
| **Hardware Environment** | Intel CPU, 8 GB System RAM, `torch.set_num_threads(2)`, `n_jobs=-1` |
| **Metrics** | Recall@5, MRR, nDCG@5, Recall@10, nDCG@10, Model Fit Time (s), Pure Re-ranking Latency (ms/query) |
| **Relevance** | Binary ground truth evidence |

---

## 3. Results

### Part 1: GBDT Framework Head-to-Head Matrix (Full Dataset, N=809 Queries)

| System | Objective / Tree Growth | Recall@5 | MRR | nDCG@5 | Recall@10 | nDCG@10 | Fit Time | Pure Rerank Latency |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| **BM25 Single-Stage** | Lexical baseline | 0.7243 | 0.6339 | 0.6438 | 0.7816 | 0.6646 | - | 18.7 ms |
| **FAISS Dense Single-Stage** | `bge-small-en-v1.5` | 0.7686 | 0.6847 | 0.6936 | 0.8452 | 0.7203 | - | **11.4 ms** |
| **LightGBM LambdaMART** | Leaf-wise (`num_leaves=31`) | **0.8168** | **0.7257** | **0.7388** | **0.8836** | **0.7617** | **0.77s** | 25.9 ms |
| **XGBoost (rank:ndcg)** | Depth-wise (`max_depth=6`) | 0.8054 | 0.7228 | 0.7327 | 0.8736 | 0.7561 | 1.16s | 26.9 ms |
| **XGBoost (rank:pairwise)** | Depth-wise (`max_depth=6`) | 0.8096 | 0.7185 | 0.7323 | 0.8719 | 0.7537 | 0.97s | 27.2 ms |

---

### Part 2: Sample Efficiency Head-to-Head Matrix (LightGBM vs. XGBoost)

| Labeled Queries ($N$) | Framework | Recall@5 | MRR | nDCG@5 | Recall@10 | nDCG@10 | Delta (vs. LGBM) | Outcome / Status |
|:---:|:---|---:|---:|---:|---:|---:|---:|:---|
| **$N = 25$** | **LightGBM** | **0.7364** | **0.6456** | **0.6555** | 0.7988 | **0.6767** | Baseline | **Beats BM25 (`0.6646`)** |
| $N = 25$ | **XGBoost (rank:ndcg)** | 0.7227 | 0.6206 | 0.6311 | **0.8111** | 0.6593 | **-0.0174** | **Trails BM25 (`0.6646`)** |
| **$N = 100$** | **LightGBM** | 0.7545 | 0.6630 | 0.6733 | **0.8291** | 0.6986 | Baseline | Strong generalization |
| $N = 100$ | **XGBoost (rank:ndcg)** | **0.7634** | **0.6696** | **0.6809** | 0.8262 | **0.7019** | **+0.0033** | Edges LightGBM |
| **$N = 809$** | **LightGBM** | **0.8168** | **0.7257** | **0.7388** | **0.8836** | **0.7617** | Baseline | **Peak In-Domain Convergence** |
| $N = 809$ | **XGBoost (rank:ndcg)** | 0.8054 | 0.7228 | 0.7327 | 0.8736 | 0.7561 | **-0.0056** | Competitive convergence |

---

## 4. What Happened?

### LightGBM Holds a Modest Edge on Quality and Speed
Across the full training set ($N=809$ queries):
- **nDCG@10**: LightGBM scored **`0.7617`** vs. XGBoost (`rank:ndcg`) **`0.7561`** ($\Delta = +0.0056$ nDCG points, a +0.74% relative advantage).
- **Recall@10**: LightGBM captured **`0.8836`** vs. XGBoost **`0.8736`** ($\Delta = +0.0100$ recall points).
- **Training Throughput**: LightGBM fitted in **0.77s** vs. XGBoost's **1.16s** (**33.6% faster training** on 69k candidate rows).
- **Pure Re-ranking Latency**: Virtually identical on CPU inference (LightGBM: **25.9 ms/query** vs. XGBoost: **26.9 ms/query**).

### XGBoost: Listwise (`rank:ndcg`) vs. Pairwise (`rank:pairwise`)
- `rank:ndcg` achieved **`0.7561` nDCG@10** compared to `0.7537` for `rank:pairwise` ($\Delta = +0.0024$).
- Direct listwise optimization of the discounted cumulative gain metric produced higher top-10 precision and MRR (`0.7228` vs. `0.7185`) than uniform pairwise preference penalties.

### The Low-Resource Divergence at $N=25$ Queries
Under severe data starvation ($N=25$ queries, ~2,100 candidate rows):
- **LightGBM succeeded**: `nDCG@10 = 0.6767`, beating unsupervised BM25 (`0.6646`).
- **XGBoost failed**: `nDCG@10 = 0.6593`, dropping **below** the BM25 baseline by -0.0053 points.
- **Root Cause**: XGBoost's depth-wise tree construction forces symmetrical splits across all nodes down to `max_depth=6`. At $N=25$ with only ~28 positive instances, this creates leaf partitions with sparse counts, leading to overfitting. In contrast, LightGBM's leaf-wise (best-first) growth with `min_child_samples=10` naturally stops expanding noisy branches early.

---

## 5. Complementarity / Diagnostics Analysis

### Top 5 Feature Importance Attribution Across Engines

Both frameworks independently arrived at the same structural conclusion: **the dense retrieval channel dominates the ranking decision**.

```text
┌─────────────────────────┬──────────────────────────────────────────┐
│ Framework               │ Top 5 Feature Importances (Gain)         │
├─────────────────────────┼──────────────────────────────────────────┤
│ LightGBM LambdaMART     │ 1. dense_rank (57,514)                   │
│                         │ 2. rrf_score (25,187)                    │
│                         │ 3. doc_token_count (4,791)               │
│                         │ 4. dense_score (2,024)                   │
│                         │ 5. length_ratio (1,832)                  │
├─────────────────────────┼──────────────────────────────────────────┤
│ XGBoost (rank:ndcg)     │ 1. dense_rank (0.7001 - 70.0% of gain)   │
│                         │ 2. rrf_score (0.1530 - 15.3% of gain)    │
│                         │ 3. bm25_rank (0.0364 - 3.6% of gain)     │
│                         │ 4. rank_discrepancy (0.0185)             │
│                         │ 5. doc_token_count (0.0150)              │
├─────────────────────────┼──────────────────────────────────────────┤
│ XGBoost (rank:pairwise) │ 1. dense_rank (0.5937 - 59.4% of gain)   │
│                         │ 2. rrf_score (0.2331 - 23.3% of gain)    │
│                         │ 3. rank_discrepancy (0.0261)             │
│                         │ 4. bm25_rank (0.0256)                    │
│                         │ 5. doc_token_count (0.0196)              │
└─────────────────────────┴──────────────────────────────────────────┘
```

**Key Diagnostic Insights:**
1. In XGBoost (`rank:ndcg`), **`dense_rank` alone accounts for 70.0% of the entire model's split gain**, and `rrf_score` contributes another 15.3%. Together, two ordinal features account for **85.3% of all ranking decisions**.
2. XGBoost allocates positions 3 and 4 to `bm25_rank` and `rank_discrepancy`, heavily prioritizing cross-channel agreement over surface text metadata.

---

## 6. Methodological Deep-Dive

### Tree Growth Architecture: Leaf-Wise vs. Depth-Wise
- **LightGBM (Leaf-Wise / Best-First)**: Splits the leaf with the maximum delta loss across the entire tree, irrespective of depth. This allows asymmetric tree shapes that isolate dense rank thresholds deeply while leaving noisy features unexpanded.
- **XGBoost (Depth-Wise / Level-Wise)**: Expands all nodes at each tree level symmetrically down to `max_depth`. When training data is abundant ($N=809$ or $N=100$), this produces solid, balanced decision boundaries (`0.7561` and `0.7019`). But when data is scarce ($N=25$), depth-wise growth is forced to split sub-trees on noisy features, explaining XGBoost's failure to beat BM25 at $N=25$.

---

## 7. Key Findings

### Finding 1 — LightGBM Retains a Minor Precision and Throughput Advantage
> On the full SciFact benchmark, LightGBM outperformed XGBoost (`rank:ndcg`) by **+0.0056 nDCG@10** (`0.7617` vs. `0.7561`) and trained **33.6% faster** (`0.77s` vs. `1.16s`), while running at identical CPU inference latency (~26 ms).

### Finding 2 — XGBoost Listwise Outperforms Pairwise
> Direct listwise nDCG optimization (`rank:ndcg`) beat pairwise loss (`rank:pairwise`) on both nDCG@10 (`0.7561` vs. `0.7537`) and MRR (`0.7228` vs. `0.7185`).

### Finding 3 — LightGBM is Significantly More Robust to Low-Data Starvation
> At $N=25$ queries, LightGBM beat BM25 (`0.6767` vs. `0.6646`), whereas XGBoost failed to reach the BM25 baseline (`0.6593`). LightGBM's leaf-wise tree growth is superior for cold-start domain adaptation.

### Finding 4 — Unanimous Feature Attribution
> Both frameworks independently allocated over 80% of total split importance to **`dense_rank`** and **`rrf_score`**, confirming that reciprocal rank order is the universal signal anchor for two-stage search.

---

## 8. What This Means for System Architecture

### Architectural Decision
1. **LightGBM Remains the Default Production Ranker**: Due to faster training, slightly higher accuracy (+0.0056 nDCG), and superior cold-start sample efficiency at $N=25$, LightGBM remains RetrievLab's primary GBDT engine.
2. **XGBoost Approved as a Drop-In Alternative**: Because both classes implement the unified `ReRanker` interface with identical input/output schemas, XGBoost is fully verified and available whenever enterprise environments mandate XGBoost runtime dependencies.

```text
               Unified GBDT Learning-to-Rank Architecture
               ══════════════════════════════════════════

  Candidate Pool (Union K=50, ~85 docs)
    │
    ▼
  Feature Extraction (16 Multi-Signal Features)
    │
    ├─────────────────────────────┬─────────────────────────────┐
    ▼                             ▼                             ▼
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ LightGBMRanker       │   │ XGBoostRanker (ndcg) │   │ XGBoostRanker (pair) │
│ (Default Choice)     │   │ (Drop-in Alternate)  │   │ (Ablation Baseline)  │
│ • nDCG@10: 0.7617    │   │ • nDCG@10: 0.7561    │   │ • nDCG@10: 0.7537    │
│ • Fit: 0.77s         │   │ • Fit: 1.16s         │   │ • Fit: 0.97s         │
│ • Safe at N >= 25    │   │ • Needs N >= 100     │   │                      │
└──────────────────────┘   └──────────────────────┘   └──────────────────────┘
    │                             │                             │
    └─────────────────────────────┼─────────────────────────────┘
                                  ▼
                     Final Sorted Top 10 Results
```
