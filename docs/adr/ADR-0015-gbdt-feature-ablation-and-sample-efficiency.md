# ADR-0015: GBDT Feature Ablation and Sample Efficiency Envelope

**Status**: Accepted  
**Deciders**: RetrievLab Team  
**Date**: 2026-09-26  

---

## Context

Across Experiments 025–029, tabular LambdaMART (`LightGBM`) demonstrated superior ranking accuracy and latency over neural cross-encoders across three distinct BEIR benchmarks (SciFact, NFCorpus, FiQA). However, to transition tabular re-ranking from an experimental prototype to production deployment, several operational constraints and structural dependencies required formal empirical boundaries:

1. **Feature Dependency**: Which feature group acts as the foundational load-bearer? Is the GBDT model reliant on lexical keyword signals, dense semantic similarities, ordinal ranks, or surface metadata?
2. **Feature Pruning Budget**: Can surface text metadata (token counts, character counts, length ratios) be omitted to eliminate text parsing overhead in the online serving path?
3. **Data Efficiency & Cold-Start**: What is the minimum training dataset size ($N_{\text{queries}}$) required before tabular LTR becomes viable? At what sample size does performance plateau?

---

## Decision

1. **Mandate Dense Semantic Signals as Primary Load-Bearers**:
   - Feature representations for second-stage tabular ranking must always include dense similarity signals (`dense_score`, `dense_rank`, `dense_recip_rank`). Dense features cannot be omitted; lexical-only tabular ranking suffers a severe -10.0% ranking quality collapse.
2. **Standardize on Dual Score-and-Rank Feature Representations**:
   - Tabular feature extractors must supply both uncalibrated continuous similarity scores (`dense_score`, `bm25_score`, z-scores) and ordinal rank positions (`recip_rank`, `rank_diff`). Scores provide granular margin separation, while ranks provide invariance to query score scale drift.
3. **Approve Pruning of Surface Text Metadata in High-Throughput Pipelines**:
   - Document length, query length, and character/token ratio features may be safely excluded from the online feature pipeline. Ablating all metadata resulted in less than a 1.5% drop in nDCG@10 while eliminating string length tokenization in the critical serving path.
4. **Establish $N=25$ Cold-Start Minimum and $N=200$ Production Annotation Target**:
   - **Hard Constraint**: Never deploy a trained LambdaMART model with fewer than 25 annotated queries ($N < 25$). Models trained on $N=10$ queries suffer severe underfitting and trail unsupervised baselines.
   - **Annotation Budget**: For new domain adaptations, human annotation can be capped at **200 queries**, which captures **96.9%** of full in-domain ranking performance and surpasses single-stage dense retrieval. Collecting additional queries beyond 200 yields sharply diminishing returns.

---

## Empirical Rationale ([`Exp 030`](file:///e:/Downloads/RetrievLab/results/sprint_3/exp030_lightgbm_stress_testing.md))

Benchmarked on BEIR SciFact ($N_{\text{test}}=300$ queries, Union $K_{\text{cand}}=50$ candidate pool):
- **Feature Ablation Impact**:
  - Full Feature Suite (16 features): `nDCG@10 = 0.7617`
  - Lexical-Only (dense ablated): `nDCG@10 = 0.6854` (**-0.0763**, -10.02% relative loss)
  - Dense-Only (lexical ablated): `nDCG@10 = 0.7226` (**-0.0391**, -5.13% relative loss)
  - Rank-Only (scores ablated): `nDCG@10 = 0.7368` (**-0.0249**, -3.27% relative loss)
  - Score-Only (ranks ablated): `nDCG@10 = 0.7459` (**-0.0158**, -2.07% relative loss)
  - No-Metadata (text lengths ablated): `nDCG@10 = 0.7507` (**-0.0110**, -1.44% relative loss)
- **Sample Complexity Envelope**:
  - $N=10$: `nDCG@10 = 0.5938` (Catastrophic failure; trails BM25 `0.6646` by -0.0708)
  - $N=25$: `nDCG@10 = 0.6767` (Viability point; beats single-stage BM25)
  - $N=200$: `nDCG@10 = 0.7380` (Production saturation knee; beats single-stage Dense `0.7203` and Hybrid RRF `0.7150`)
  - $N=809$: `nDCG@10 = 0.7617` (Full convergence; +0.0237 lift over $N=200$ for $4\times$ data)

---

## Consequences

### Positive
- Clarifies the minimal viable feature set required for production GBDT serving (6–12 core features).
- Eliminates expensive document text parsing during online re-ranking without measurable degradation.
- Provides precise quantitative ROI guidelines for human data annotation budgets (stop at 200 queries).

### Negative / Trade-offs
- Tabular GBDT re-ranking cannot function as a standalone lexical enhancer; it strictly requires a functional dense embedding retriever in Stage 1 to achieve competitive ranking quality.
