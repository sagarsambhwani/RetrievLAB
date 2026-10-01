# ADR-0011: Two-Stage Re-Ranking: Tabular LambdaMART vs. Neural Cross-Attention

**Status**: Accepted  
**Deciders**: RetrievLab Team  
**Date**: 2026-09-17  

---

## Context

In modern multi-stage search systems, candidate pools generated in Stage 1 ($K_{\text{cand}} \approx 50 \text{ to } 100$) must be re-ordered by relevance in Stage 2. RetrievLab evaluated two fundamentally different re-ranking paradigms:

1. **Neural Cross-Attention (Cross-Encoder)**:
   Passes concatenated query-document pairs (`[CLS] Query [SEP] Document [SEP]`) through multi-layer bidirectional transformer encoders (e.g. `ms-marco-MiniLM-L-6-v2`). Every query token attends directly to every document token ($O(L^2)$ cross-attention), providing maximum semantic expressiveness but requiring intensive floating-point tensor arithmetic.
2. **Tabular Gradient-Boosted Decision Trees (LightGBM LambdaMART)**:
   Extracts multi-signal numerical features (lexical scores, dense similarities, ordinal ranks, length ratios) and evaluates shallow decision tree ensembles trained via pairwise LambdaMART loss. Inference consists of CPU integer comparisons and scalar additions with zero matrix multiplications.

The core architectural question is: **Which re-ranking paradigm provides the optimal balance of ranking accuracy, serving latency, hardware efficiency, and domain robustness?**

---

## Decision

1. Define the unified **`ReRanker`** abstract base class in `src/retrievlab/ranking/interface.py`:
   ```python
   class ReRanker(ABC):
       @abstractmethod
       def rerank(
           self,
           query: str,
           candidates: Union[CandidatePool, Sequence[Union[Chunk, Candidate]]],
           top_k: int | None = None,
       ) -> list[SearchResult]:
           pass
   ```
2. Implement **`CrossEncoderReRanker`** in `src/retrievlab/ranking/cross_encoder.py` leveraging SentenceTransformers with GPU micro-batching (`batch_size=64`, `max_length=256`).
3. Implement **`LightGBMRanker`** in `src/retrievlab/ranking/lightgbm.py` leveraging LightGBM's native `objective="lambdarank"` and `eval_metric="ndcg"`.
4. **Adopt `LightGBMRanker` as the primary production-grade re-ranker** in RetrievLab, retaining `CrossEncoderReRanker` as a scientific benchmark baseline.

---

## Architectural Principles & Implementation Details

1. **Pluggable Polymorphic Interface**:
   Both `CrossEncoderReRanker` and `LightGBMRanker` implement the identical `ReRanker` signature and accept either a `CandidatePool` or raw `Sequence[Chunk | Candidate]`. Calling pipelines can swap between neural cross-attention and tabular trees without changing evaluation harnesses.
2. **Hardware Separation & Resource Safety**:
   * Neural models require dedicated GPU VRAM (routed to NVIDIA GeForce GTX 1650) to protect our tight 8 GB System RAM boundary.
   * Tabular GBDT models execute entirely on CPU OpenMP multithreading, consuming under 50 MB of RAM and zero GPU VRAM.
3. **Training Efficiency**:
   While fine-tuning transformer cross-encoders requires hours of GPU backpropagation and millions of parameters, LightGBM fits 100 trees over 69,079 candidate pairs in **1.21 seconds on CPU**.

---

## Validation & Experimental Findings ([`exp024`](file:///e:/Downloads/RetrievLab/results/sprint_3/exp024_cross_encoder_reranking.md) & [`exp025`](file:///e:/Downloads/RetrievLab/results/sprint_3/exp025_twostage_reranking_benchmark.md))

Both systems were benchmarked on BEIR SciFact ($N=300$ test queries, 5,183 documents) against single-stage baselines:

| Paradigm | System | Recall@5 | nDCG@5 | Latency (Pure Re-Rank) | Hardware |
|:---|:---|---:|---:|---:|:---|
| **Single-Stage Baseline** | Hybrid RRF (1:2) | 0.7777 | 0.6927 | — | CPU |
| **Neural Two-Stage** | Union + Cross-Encoder (`MiniLM`) | 0.7452 | 0.6626 | 455.03 ms | GPU (GTX 1650) |
| **Tabular GBDT Two-Stage** | Union + LightGBM LambdaMART | **0.8168** | **0.7388** | **25.66 ms** | CPU |

### Key Empirical Takeaways
1. **Accuracy Dominance**: LightGBM outperformed the Cross-Encoder by **+0.0762 nDCG@5** (`0.7388` vs `0.6626`) and **+7.2%p Recall@5** (`0.8168` vs `0.7452`).
2. **17.7× Latency Advantage**: Pure LightGBM re-ranking executed in **25.66 ms/query** on CPU, compared to **455.03 ms/query** for batch GPU cross-attention.
3. **Domain Adaptability**: Pre-trained web cross-encoders (`ms-marco`) suffered from domain mismatch when evaluated on scientific abstracts. Because LightGBM trains in seconds, it can be trained directly in-domain on target queries, completely avoiding out-of-domain degradation.

---

## Consequences

### Positive
* Delivers state-of-the-art ranking performance (`nDCG@5 = 0.7388`) at sub-30ms re-ranking latency on commodity CPU hardware.
* Re-ranking consumes negligible system RAM (< 50 MB) and leaves 100% of GPU VRAM free for vector embeddings or LLM generation.
* Decoupled `ReRanker` abstraction enables easy benchmarking of future rankers (e.g. CatBoost, ColBERT, or LLM listwise rankers).

### Tradeoffs / Considerations
* LightGBM requires offline training data (labeled queries with candidate pools); unlike zero-shot Cross-Encoders, it cannot be deployed out-of-the-box without training annotations.
* Feature extraction must run online before tree traversal (~24 ms overhead to extract 16 features across 85 candidates).
