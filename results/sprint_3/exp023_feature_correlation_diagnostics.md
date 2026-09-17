# Experiment 023 — Signal Association & Feature Collinearity Diagnostics

**Date:** 2026-09-16  
**Status:** ✅ Completed

---

## 1. Research Question

> **What information exists in our 16-dimensional candidate feature space, how does each feature associate with relevance labels, and where is that information redundant?**

Before training second-stage ranking algorithms (such as LightGBM LambdaMART), we must diagnose the statistical properties of our candidate features. Correlation diagnostics evaluate the monotonic (Spearman) and linear (Pearson) associations of individual features with ground-truth relevance, while inter-feature collinearity diagnostics map shared representation clusters across lexical, dense, fusion, and document-length signals.

---

## 2. Experiment Setup

| Property | Configuration |
|---|---|
| **Benchmark** | BEIR SciFact |
| **Queries** | 300 test queries |
| **Corpus** | 5,183 scientific abstracts |
| **Candidate Depth ($K_{\text{cand}}$)** | 50 (operating point from Exp 022) |
| **Total Evaluated Pairs ($N$)** | 25,663 `(query, candidate)` pairs |
| **Mean Pool Size per Query** | 85.5 candidates |
| **Evaluated Feature Suite** | 16 standardized features (`retrievlab.features.FeatureExtractor`) |
| **Relevance Ground Truth** | Graded judgments ($y \in \{0, 1, 2\}$) from SciFact `qrels` |
| **Statistical Metrics** | Spearman Rank Correlation ($\rho$), Pearson Linear Correlation ($r$), $p$-values |

---

## 3. Results

### Feature-to-Label Association (Ranked by Absolute Spearman $\rho$)

| Rank | Feature Name | Spearman $\rho$ | Spearman $p$-value | Pearson $r$ | Pearson $p$-value | Mean $\pm$ Std |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `retrieved_by_both` | **+0.2111** | 1.90e-256 | +0.2111 | 1.90e-256 | 0.17 $\pm$ 0.37 |
| 2 | `rrf_score` | **+0.1758** | 3.10e-177 | +0.2889 | 0.00e+00 | 0.01 $\pm$ 0.01 |
| 3 | `dense_rank` | **-0.1758** | 3.74e-177 | -0.0929 | 3.02e-50 | 430.41 $\pm$ 480.37 |
| 4 | `dense_reciprocal_rank` | **+0.1758** | 3.74e-177 | +0.5415 | 0.00e+00 | 0.05 $\pm$ 0.13 |
| 5 | `dense_score` | **+0.1608** | 3.23e-148 | +0.1136 | 2.01e-74 | 0.42 $\pm$ 0.36 |
| 6 | `bm25_rank` | **-0.1543** | 2.12e-136 | -0.0770 | 5.10e-35 | 430.41 $\pm$ 480.37 |
| 7 | `bm25_reciprocal_rank` | **+0.1543** | 2.12e-136 | +0.4900 | 0.00e+00 | 0.05 $\pm$ 0.13 |
| 8 | `bm25_score` | **+0.1441** | 4.58e-119 | +0.2705 | 0.00e+00 | 8.54 $\pm$ 9.23 |
| 9 | `rank_discrepancy` | **-0.1390** | 7.35e-111 | -0.2126 | 3.62e-260 | 809.45 $\pm$ 359.80 |
| 10 | `token_overlap_ratio` | **+0.1165** | 2.67e-78 | +0.1585 | 5.82e-144 | 0.38 $\pm$ 0.15 |
| 11 | `jaccard_similarity` | **+0.0860** | 2.35e-43 | +0.1193 | 5.31e-82 | 0.04 $\pm$ 0.02 |
| 12 | `doc_token_count` | **-0.0119** | 0.0571 | +0.0054 | 0.3906 | 239.35 $\pm$ 93.46 |
| 13 | `modality_preference` | **+0.0110** | 0.0788 | +0.0138 | 0.0271 | 0.50 $\pm$ 0.46 |
| 14 | `length_ratio` | **-0.0040** | 0.5246 | -0.0027 | 0.6677 | 21.85 $\pm$ 13.62 |
| 15 | `query_token_count` | **+0.0016** | 0.7918 | +0.0027 | 0.6659 | 13.03 $\pm$ 5.43 |
| 16 | `exact_query_match` | **+0.0000** | 1.0000 | +0.0000 | 1.0000 | 0.00 $\pm$ 0.00 |

> **Note on Statistical Significance:** Due to the large sample size ($N = 25,663$), $p$-values are extremely small across active features. In this analysis, **effect size ($\rho$ and $r$) is the primary metric of practical association** rather than statistical significance.

### Inter-Feature Multicollinearity (|$\rho| \ge 0.50$)

| Feature A | Feature B | Spearman Correlation ($\rho$) | Signal Cluster |
|---|---|---:|---|
| `bm25_rank` | `bm25_reciprocal_rank` | -1.0000 | Modality Representation |
| `dense_rank` | `dense_reciprocal_rank` | -1.0000 | Modality Representation |
| `bm25_score` | `bm25_rank` | -0.8902 | Modality Representation |
| `bm25_score` | `bm25_reciprocal_rank` | +0.8902 | Modality Representation |
| `dense_score` | `dense_rank` | -0.8787 | Modality Representation |
| `dense_score` | `dense_reciprocal_rank` | +0.8787 | Modality Representation |
| `dense_rank` | `modality_preference` | -0.7725 | Cross-Modality Interaction |
| `dense_reciprocal_rank` | `modality_preference` | +0.7725 | Cross-Modality Interaction |
| `dense_score` | `modality_preference` | +0.7723 | Cross-Modality Interaction |
| `bm25_rank` | `modality_preference` | +0.7699 | Cross-Modality Interaction |
| `bm25_reciprocal_rank` | `modality_preference` | -0.7699 | Cross-Modality Interaction |
| `bm25_score` | `modality_preference` | -0.7649 | Cross-Modality Interaction |
| `query_token_count` | `length_ratio` | -0.7485 | Document / Query Length |
| `jaccard_similarity` | `length_ratio` | -0.6780 | Lexical Alignment |
| `retrieved_by_both` | `rank_discrepancy` | -0.6492 | Cross-Modality Interaction |
| `rrf_score` | `retrieved_by_both` | +0.6492 | Cross-Modality Interaction |
| `doc_token_count` | `length_ratio` | +0.6336 | Document / Query Length |
| `jaccard_similarity` | `query_token_count` | +0.6138 | Lexical Alignment |
| `token_overlap_ratio` | `jaccard_similarity` | +0.6100 | Lexical Alignment |
| `bm25_rank` | `token_overlap_ratio` | -0.5333 | Lexical Alignment |
| `bm25_reciprocal_rank` | `token_overlap_ratio` | +0.5333 | Lexical Alignment |
| `dense_rank` | `rrf_score` | -0.5173 | Cross-Modality Interaction |
| `dense_reciprocal_rank` | `rrf_score` | +0.5173 | Cross-Modality Interaction |
| `bm25_score` | `token_overlap_ratio` | +0.5172 | Lexical Alignment |
| `bm25_rank` | `rrf_score` | -0.5146 | Cross-Modality Interaction |
| `bm25_reciprocal_rank` | `rrf_score` | +0.5146 | Cross-Modality Interaction |
| `bm25_score` | `jaccard_similarity` | +0.5101 | Lexical Alignment |

---

## 4. What Happened?

### 1. Modest but Positive Association for Fusion and Dense Signals

The highest individual monotonic associations in this experiment were observed in `retrieved_by_both` ($\rho = +0.2111$), `rrf_score` ($\rho = +0.1758$), and `dense_rank` ($\rho = -0.1758$).

These values represent positive but modest individual associations ($\rho \approx 0.16 - 0.21$), confirming that single retrieval signals contain measurable relevance information but leave substantial unexplained variance for downstream re-ranking models.

### 2. Monotonic vs. Linear Discrepancies ($\rho$ vs. $r$)

Rank-based features (e.g. `dense_rank`, `bm25_rank`) exhibit higher absolute Spearman rank correlations than Pearson linear correlations. Because retrieval ranks follow a long-tailed non-linear distribution, non-parametric tree models (GBDT / LightGBM) can split on these rank thresholds directly without requiring manual normalization.

### 3. Consensus Signal (`retrieved_by_both`)

The binary agreement flag `retrieved_by_both` shows a correlation of $\rho = +0.2111$, providing a discrete indicator that separates dual-retrieved candidates from single-modality candidates.

### 4. Constant Behavior of `exact_query_match`

> **`exact_query_match` was constant at zero across all 25,663 candidate pairs, so it provides no discriminative information for this experiment. Its usefulness on other datasets remains untested.**

This occurs because SciFact test queries are complete claims (e.g. *"0-dimensional nanocarriers fail to enhance delivery..."*), whereas scientific abstracts use formal, varied phrasing rather than exact query verbatim strings.

---

## 5. Complementarity / Collinearity Diagnostics

### Representation Clusters & Redundancy Mapping

The inter-feature correlation matrix reveals four distinct information clusters:

1. **Dense Representation Cluster**: `dense_score`, `dense_rank`, and `dense_reciprocal_rank` share strong internal correlations ($|\rho| > 0.87$). They represent alternative scalings of the same underlying semantic vector similarity.
2. **Lexical Representation Cluster**: `bm25_score`, `bm25_rank`, `bm25_reciprocal_rank`, `token_overlap_ratio`, and `jaccard_similarity` form a cluster ($|\rho| > 0.51$) tracking word-level keyword matches.
3. **Length Cluster**: `query_token_count`, `doc_token_count`, and `length_ratio` correlate weakly with relevance labels ($|\rho| < 0.02$), showing that document length alone does not directly indicate relevance in this dataset.
4. **Interaction Signals**: `rank_discrepancy` and `modality_preference` capture disagreement between modalities, providing non-redundant interaction information.

---

## 6. Methodological Deep-Dive

### Why Correlation is Not Feature Importance

It is critical to distinguish between **feature correlation** and **model feature importance**:
- **Correlation** evaluates each feature in isolation against the label ($X_j \leftrightarrow y$). It does not account for feature interactions or redundant representations.
- **Feature Importance** (e.g. LightGBM Gain / SHAP) evaluates each feature's marginal contribution **in the presence of all other features**.

Our multicollinearity analysis shows that while `dense_reciprocal_rank` and `dense_score` both correlate with relevance, a trained decision tree may only need one of them to achieve optimal split gain. This diagnostics experiment establishes the baseline hypothesis for our subsequent **feature ablation** experiments.

---

## 7. Key Findings

### Finding 1 — Dense & Fusion Signals Show Highest Monotonic Associations
> `retrieved_by_both` ($\rho = +0.2111$) and `rrf_score` ($\rho = +0.1758$) demonstrate the highest individual monotonic associations with relevance on SciFact.

### Finding 2 — Secondary Lexical Signals with Partial Redundancy
> Lexical features (`bm25_reciprocal_rank` $\rho = +0.1543$, `token_overlap_ratio` $\rho = +0.1165$) provide moderate correlation, but exhibit partial redundancy with BM25 scores ($\rho = 0.517$) and Jaccard similarity ($\rho = 0.610$).

### Finding 3 — Length Features Show Weak Direct Association with Relevance
> `doc_token_count` ($\rho = -0.0119$) and `query_token_count` ($\rho = +0.0016$) show near-zero direct correlation with relevance.

### Finding 4 — Redundancy Within Modality Clusters
> Our 16 features contain substantial redundancy within several signal groups, with reciprocal ranks ($1/r$), raw ranks ($r$), and raw scores within the same modality exhibiting $|\rho| > 0.87$.

---

## 8. What This Means for System Architecture

For Stage-2 LightGBM LambdaMART ranking (`RLB-331`):
- Retrieval provenance, dense ranking, fusion, and BM25 signals contain measurable relevance information, while several representations of the same retrieval signal are highly redundant.
- Decision trees will naturally utilize non-linear rank splits and consensus flags (`retrieved_by_both`) to re-score candidates.
- Following initial LTR training, a systematic **feature ablation sweep** should test whether compact 6–8 feature subsets match full 16-feature performance for lower latency.

```text
CandidatePool (K=50) ──► FeatureExtractor (16 Features) ──► LTRDataset
                                                                │
         ┌──────────────────────────────────────────────────────┴──────────────────────────────────────┐
         ▼                                                                                            ▼
Correlated Signal Clusters:                                                                 Context Priors & Disagreement:
 • Fusion: rrf_score, retrieved_by_both                                                       • rank_discrepancy, modality_preference
 • Dense: dense_score, dense_rank, dense_reciprocal_rank                                      • query/doc length, length_ratio
 • Lexical: bm25_score, bm25_rank, overlap, jaccard                                           
         │                                                                                            │
         └──────────────────────────────────► LightGBM LambdaMART ◄───────────────────────────────────┘
                                                       │
                                                       ▼
                                            Optimal Re-Ranked Evidence
```
