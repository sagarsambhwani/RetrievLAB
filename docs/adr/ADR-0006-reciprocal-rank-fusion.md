# ADR-0006: Reciprocal Rank Fusion (RRF) Architecture & Score Normalization Invariance

**Status**: Accepted  
**Deciders**: RetrievLab Team  
**Date**: 2026-08-23  

---

## Context
In Sprint 1, we observed that lexical retrieval (BM25) and dense retrieval (FastEmbed BGE) exhibit orthogonal error patterns:
- BM25 excels at exact keyword matching, code identifiers, and rare tokens, but fails when queries use paraphrases without lexical overlap.
- Dense retrieval excels at semantic similarity and synonym matching, but can miss exact keyword constraints.

Combining both signals into a hybrid candidate generation pipeline requires a rank aggregation or score fusion strategy. However:
1. BM25 produces unbounded positive relevance scores ($[0, \infty)$) dependent on document length and term frequency saturation ($k_1, b$).
2. Dense cosine similarity produces scores in $[-1.0, 1.0]$ (or $[0, 1.0]$ for typical text pairs).
3. Directly summing raw scores or applying linear interpolation requires complex score calibration/normalization (e.g. min-max scaling or z-score), which is sensitive to corpus size, outlier scores, and query-dependent distribution shifts.

## Decision
Adopt **Reciprocal Rank Fusion (RRF)** (Cormack, Clarke, and Büttcher, 2009) as the foundational unsupervised rank aggregation algorithm for RetrievLab's hybrid retrieval system.

Implement RRF in `src/retrievlab/ranking/fusion.py` as a standalone, configurable rank aggregation component that accepts multiple ranked lists and outputs unified `SearchResult` objects.

### Mathematical Formulation
For a set of $M$ ranked candidate lists and a chunk $d$:

$$\text{RRF\_Score}(d) = \sum_{m \in M} \frac{w_m}{k + r_m(d)}$$

where:
- $r_m(d)$ is the 1-based rank position of chunk $d$ in the $m$-th ranking list.
- $k$ is the smoothing constant (default: $k = 60$).
- $w_m$ is an optional weight for the $m$-th retrieval system (default: $1.0$).
- If chunk $d$ does not appear in ranking list $m$, its reciprocal rank contribution for that list is $0.0$.

## Architectural Assumptions & Invariants (Build vs. Use)

1. **Scale Invariance**: RRF operates purely on rank positions ($1, 2, \dots$), making it completely invariant to the underlying score distributions, scales, and score bounds of individual retrievers.
2. **Smoothing Parameter $k$**: The parameter $k$ prevents top-ranked items from dominating the fused score excessively. With $k=60$:
   - Rank 1 contributes $\frac{1}{61} \approx 0.01639$
   - Rank 2 contributes $\frac{1}{62} \approx 0.01613$
   - Rank 10 contributes $\frac{1}{70} \approx 0.01429$
   - Rank 100 contributes $\frac{1}{160} \approx 0.00625$
3. **Multi-System Generalization ($N \ge 2$)**: The algorithm generically aggregates $N$ distinct retrievers (e.g., BM25 + Dense + SPLADE + Graph).
4. **Deterministic Tie-Breaking**: When multiple items obtain identical RRF scores, tie-breaking is deterministic (sorted by chunk ID), guaranteeing reproducible benchmarks.

## Consequences

### Positive
- Zero calibration needed across different scoring functions.
- Highly resilient to outlier scores and noisy individual rankings.
- Fully decoupled from specific retriever classes, conforming to *Rule 1 (One algorithm, one implementation)* and *Rule 2 (Configurable behavior)*.
- High performance ($O(N \cdot K)$ time complexity where $K$ is the candidate pool size).

### Negative / Tradeoffs
- Discards raw score confidence deltas (e.g., whether rank 1 scored $0.99$ or $0.51$).
- In later phases (Sprint 3 / learned ranking), feature-based rankers (e.g., LightGBM / LambdaMART) will be evaluated against RRF baselines to capture fine-grained score margins.
