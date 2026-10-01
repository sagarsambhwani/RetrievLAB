# ADR-0012: Cascaded Coarse-to-Fine Re-Ranking Architecture

**Status**: Accepted  
**Deciders**: RetrievLab Team  
**Date**: 2026-09-24  

---

## Context

In multi-stage information retrieval systems, candidate generation (Stage 1) produces a candidate pool ($K_{\text{cand}} \in [50, 100]$) to maximize recall. However, re-ranking this candidate pool in Stage 2 presents a fundamental trade-off:

1. **Neural Cross-Encoders (`ms-marco-MiniLM-L-6-v2`)**:
   Compute full sequence token-level cross-attention (`[CLS] Query [SEP] Chunk [SEP]`). While semantically expressive, latency scales linearly with candidate depth ($O(N \cdot L^2)$), requiring $450\text{ ms}+$ on CPU for 50+ candidates. Furthermore, when exposed to large candidate pools in specialized domains, off-the-shelf cross-encoders are susceptible to false-positive lexical and semantic distractors.
2. **Tabular GBDTs (`LightGBMRanker` / LambdaMART)**:
   Evaluate multi-signal tabular features (BM25 scores, dense cosine similarities, rank discrepancies, token overlaps) in under $1\text{ ms}$, providing rapid coarse sorting and distractor elimination.

The architectural challenge is: **How can RetrievLab achieve the fine-grained semantic discrimination of neural cross-attention without paying the quadratic compute and latency penalty on 50–100 candidates, while mitigating out-of-domain distractor confusion?**

---

## Decision

1. Implement **`CascadedReRanker`** in `src/retrievlab/ranking/cascade.py`, implementing the polymorphic `ReRanker` interface:
   ```python
   class CascadedReRanker(ReRanker):
       def __init__(
           self,
           filter_ranker: ReRanker,
           refiner_ranker: ReRanker,
           intermediate_k: int = 15,
       ) -> None:
           self.filter_ranker = filter_ranker
           self.refiner_ranker = refiner_ranker
           self.intermediate_k = intermediate_k
   ```
2. **Execution Flow**:
   - **Step 1 (Coarse Filter)**: `filter_ranker.rerank(query, candidates, top_k=intermediate_k)` scores all $K_{\text{cand}}$ candidates using tabular LightGBM and retains only the top $K_{\text{inter}}$ items.
   - **Step 2 (Fine Refiner)**: `refiner_ranker.rerank(query, filtered_chunks, top_k=top_k)` evaluates only those $K_{\text{inter}}$ items using neural cross-attention, returning the final ranked list.
3. Export `CascadedReRanker` in `src/retrievlab/ranking/__init__.py`.
4. Validate performance against standalone baselines in **Experiment 027**.

---

## Architectural Principles & Rationale

1. **Distractor Pruning**:
   Off-the-shelf neural cross-encoders trained on broad web text (MS MARCO) frequently assign high scores to irrelevant technical passages that share peripheral vocabulary. Pre-filtering with LightGBM prunes 70–85% of irrelevant candidate chunks, presenting the neural refiner with a cleaner, higher-precision candidate set.
2. **Linear Latency Acceleration**:
   Because Transformer self-attention latency scales linearly with candidate count, reducing candidate depth from 50 to 10 cuts neural forward passes by **$80\%$**, accelerating re-ranking latency by over **$5\times$**.
3. **Pluggable Polymorphic Composition**:
   Because `CascadedReRanker` implements the standard `ReRanker` contract, calling applications interact with it identically to any single-stage or two-stage ranker. It can chain any arbitrary filter (e.g. fast linear ranker, LightGBM, SPLADE) with any refiner (e.g. MiniLM, DeBERTa, ColBERT).

---

## Empirical Validation ([`Exp 027`](file:///e:/Downloads/RetrievLab/results/sprint_3/exp027_cascaded_reranking.md))

Benchmarked on the BEIR SciFact dataset ($N=300$ test queries, 5,183 scientific abstracts) using candidate pools of depth $K_{\text{cand}}=50$:

| Architecture | Intermediate Cutoff ($K_{\text{inter}}$) | nDCG@10 | MRR | Re-rank Latency | Speedup vs Pure CE |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Two-Stage Pure Cross-Encoder** | — (scores all 50) | 0.6838 | 0.6522 | 459.64 ms | 1.00x |
| **Cascade (LightGBM $\to$ Cross-Encoder)** | $K=30$ | 0.6953 | 0.6606 | 187.70 ms | 2.45x |
| **Cascade (LightGBM $\to$ Cross-Encoder)** | $K=20$ | 0.6988 | 0.6602 | 133.97 ms | 3.43x |
| **Cascade (LightGBM $\to$ Cross-Encoder)** | $K=15$ | 0.7050 | 0.6640 | 107.10 ms | 4.29x |
| **Cascade (LightGBM $\to$ Cross-Encoder)** | **$K=10$** | **0.7194** | **0.6726** | **80.24 ms** | **5.73x** |
| **Two-Stage Pure LightGBM** | — (scores all 50) | 0.7617 | 0.7257 | 26.50 ms | 17.34x |

### Key Discoveries:
1. **Quality Gain via Distractor Pruning**: Cascading ($50 \to 10$) **increased Cross-Encoder nDCG@10 by +0.0355** (`0.6838` $\to$ `0.7194`) by removing negative distractors.
2. **Speedup**: Cut re-ranking latency from $459.64\text{ ms}$ down to $80.24\text{ ms}$ (**$5.73\times$ faster**).
3. **In-Domain GBDT Dominance**: In specialized domains with available training data, in-domain trained LightGBM LambdaMART establishes the highest individual benchmark (`0.7617` nDCG@10).

---

## Consequences

### Positive
- Bridges the gap between high-recall candidate generation ($K=50..100$) and compute-constrained neural re-ranking.
- Prevents thermal and memory exhaustion on resource-constrained serving infrastructure by bounding neural sequence evaluations.
- Lays the operational foundation for **Adaptive LTR** (Pillar I), where queries can conditionally exit after the coarse filter or proceed to the neural refiner.

### Negative
- Requires maintaining two models in serving pipelines (feature extraction pipeline + neural tokenizer/model weights).
- If the coarse filter erroneously discards a relevant candidate at position $K > K_{\text{inter}}$, the fine refiner has no opportunity to recover it.
