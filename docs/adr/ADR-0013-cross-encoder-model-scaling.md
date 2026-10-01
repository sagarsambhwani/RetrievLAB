# ADR-0013: Cross-Encoder Model Scaling Limits & Task Incongruence

**Status**: Accepted  
**Deciders**: RetrievLab Team  
**Date**: 2026-09-25  

---

## Context

In Experiment 025, our initial neural re-ranking baseline (`cross-encoder/ms-marco-MiniLM-L-6-v2`) severely degraded ranking quality on BEIR SciFact (`nDCG@10 = 0.6838`), trailing both unsupervised single-stage retrievers (FAISS Dense: `0.7203`) and tabular GBDT re-ranking (`0.7617`).

A prominent hypothesis in modern retrieval research posits that **model scale and broader pre-training** can overcome out-of-domain degradation. Specifically:
- `ms-marco-MiniLM-L-6-v2` has only 22M parameters and was trained solely on Bing web search clicks.
- `BAAI/bge-reranker-base` has 110M parameters (5× capacity, 12 layers) and was trained on diverse multilingual, academic, and web retrieval benchmarks.

The architectural question was: **Does upgrading candidate pool re-ranking from a 22M MiniLM model to a 110M BGE model resolve the neural ranking deficit on specialized scientific literature, and justify the accompanying latency penalty?**

---

## Decision

1. **Reject Zero-Shot General-Purpose Cross-Encoders as Primary Re-Rankers on Specialized Claim Verification Tasks**:
   - Zero-shot neural cross-encoders trained on general QA and semantic similarity must not be deployed as default re-rankers on claim verification or contradiction-sensitive benchmarks without domain fine-tuning.
2. **Standardize LightGBM GBDT Re-Ranking as the Default High-Throughput Stage-2 Ranker**:
   - For technical enterprise and scientific corpora, retain tabular LightGBM (`0.7617` nDCG@10, `21.7 ms` latency) as the primary Stage-2 ranker over candidate pools.
3. **Preserve Pluggable Model Selection in `CrossEncoderReRanker`**:
   - Maintain the `model_name` parameter in `CrossEncoderReRanker` to allow seamless loading of domain-adapted models (e.g., bio-rerankers) without architectural alterations.

---

## Architectural Rationale & Failure Diagnosis

Empirical validation in **Experiment 028** revealed that scaling to 110M parameters produced **negative scaling**:
- `nDCG@10` dropped from `0.6838` (MiniLM) to `0.6463` (BGE-base), a relative decline of **-5.49%**.
- `MRR` dropped from `0.6522` to `0.6100` (**-6.47%**).
- Pure re-ranking latency increased from `44.9 ms` to `4822.2 ms` (**107× latency inflation**).

### Root Causes
1. **Semantic Entailment vs. Contradiction Evidence**:
   Scientific claim verification requires retrieving abstracts that either support *or refute* a declarative claim. General-purpose cross-encoders penalize contradictory phrasing, falsely treating refuting evidence as non-relevant.
2. **Distractor Sensitivity in Uncalibrated Attention**:
   In candidate pools of 80+ abstracts, non-relevant abstracts share dense biomedical terminology. Higher model capacity without task supervision caused BGE-base to attend intensely to superficial token overlaps in distractors, demoting true evidence.
3. **Loss of Multi-Retriever Provenance**:
   Neural cross-encoders score `(query, document)` in isolation, ignoring whether BM25 or FAISS retrieved the document at rank 1 or rank 49. LightGBM exploits this concordance signal directly via `dense_rank`, `rrf_score`, and `rank_discrepancy`.

---

## Consequences

### Positive
- Prevents wasteful allocation of GPU memory and latency budget to large zero-shot models that actively harm retrieval accuracy.
- Confirms the architectural strength of RetrievLab's tabular multi-signal feature extraction pipeline.

### Negative / Trade-offs
- Overcoming the ranking ceiling on SciFact will require domain-specific fine-tuning (e.g. training a Cross-Encoder on SciFact/FEVER scientific triplets) or late-interaction architectures (ColBERT), rather than simply scaling pre-trained checkpoints.
