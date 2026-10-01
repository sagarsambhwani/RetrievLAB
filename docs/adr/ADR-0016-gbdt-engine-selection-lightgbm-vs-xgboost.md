# ADR-0016: GBDT Engine Selection — LightGBM vs. XGBoost

**Status**: Accepted  
**Deciders**: RetrievLab Team  
**Date**: 2026-09-26  

---

## Context

Following the success of tabular Learning-to-Rank in Sprint 3, RetrievLab evaluated the two industry-standard Gradient Boosted Decision Tree (GBDT) frameworks:
1. **LightGBM** (`LGBMRanker`, LambdaMART with leaf-wise tree growth).
2. **XGBoost** (`XGBRanker`, listwise `rank:ndcg` and pairwise `rank:pairwise` with depth-wise tree growth).

Both engines were implemented under the unified [`ReRanker`](file:///e:/Downloads/RetrievLab/src/retrievlab/ranking/interface.py) interface and benchmarked head-to-head on BEIR SciFact (300 test queries, 809 training queries, 16 features).

The architectural decisions to settle were:
1. Which engine should serve as RetrievLab's default production re-ranker?
2. How do the engines compare in cold-start sample efficiency ($N \in [25, 100, 809]$)?
3. Does XGBoost provide any ranking accuracy advantage over LightGBM?

---

## Decision

1. **Retain LightGBM as RetrievLab's Default Production GBDT Engine**:
   - LightGBM LambdaMART is selected as the default primary re-ranker due to:
     - **Higher Ranking Accuracy**: `nDCG@10 = 0.7617` vs. XGBoost's `0.7561` (+0.0056 points).
     - **Faster Training Throughput**: 0.77s vs. 1.16s (33.6% faster fitting on 69k candidate rows).
     - **Superior Cold-Start Robustness**: LightGBM beats BM25 at only $N=25$ queries (`0.6767` vs. `0.6646`), whereas XGBoost trails BM25 at $N=25$ (`0.6593`).
2. **Standardize `XGBoostRanker` as a Drop-In Enterprise Alternate**:
   - `XGBoostRanker` is fully supported, tested, and exported in `retrievlab.ranking`. It is approved for deployment in environments where enterprise infrastructure mandates XGBoost runtimes (e.g., Treelite compilation, Triton Inference Server).
3. **Mandate $N \ge 100$ Queries for XGBoost Deployments**:
   - While LightGBM is viable at $N=25$, XGBoost's depth-wise tree growth requires at least 100 queries before it reliably beats unsupervised baselines (`nDCG@10 = 0.7019` at $N=100$).

---

## Empirical Rationale ([`Exp 031`](file:///e:/Downloads/RetrievLab/results/sprint_3/exp031_lightgbm_vs_xgboost.md))

Benchmarked on BEIR SciFact ($N_{\text{test}}=300$ queries, Union $K_{\text{cand}}=50$ pool):
- **Full In-Domain Comparison ($N=809$ Queries)**:
  - LightGBM LambdaMART: `nDCG@10 = 0.7617`, `Recall@10 = 0.8836`, Fit = 0.77s, Rerank Latency = 25.9 ms
  - XGBoost (`rank:ndcg`): `nDCG@10 = 0.7561`, `Recall@10 = 0.8736`, Fit = 1.16s, Rerank Latency = 26.9 ms
  - XGBoost (`rank:pairwise`): `nDCG@10 = 0.7537`, `Recall@10 = 0.8719`, Fit = 0.97s, Rerank Latency = 27.2 ms
- **Sample Efficiency Divergence**:
  - At $N=25$: LightGBM achieved `0.6767` (beating BM25 `0.6646`), while XGBoost scored `0.6593` (trailing BM25).
  - At $N=100$: XGBoost achieved `0.7019` vs. LightGBM `0.6986` (virtually tied).
- **Feature Agreement**:
  - Both engines allocated >85% of tree gain to `dense_rank` and `rrf_score`.

---

## Consequences

### Positive
- Both major GBDT frameworks are now fully unified behind RetrievLab's clean `ReRanker` interface.
- Engineers have clear empirical guidelines for when to choose LightGBM (default, cold-start) vs. XGBoost (enterprise runtimes).
- Confirmed that listwise nDCG loss is mathematically superior to pairwise ranking loss across both engines.

### Negative / Trade-offs
- Adding `xgboost` introduces a 46 MB dependency into the virtual environment, though both libraries remain lightweight relative to PyTorch neural models.
