# 🗓️ RetrievLab Sprint 3

**Sprint:** Two-Stage Retrieval & Learning-to-Rank (LTR) — From Candidate Generation to Neural Re-Ranking  
**Duration:** 1–2 Weeks  
**Status:** ⚪ Planned  

---

# 🎯 Sprint Vision

Sprint 1 established foundational baselines (BM25, Dense) and evaluation harnesses.  
Sprint 2 introduced hybrid rank fusion (RRF), failure diagnostics, and accelerated vector indexing (FAISS).

Sprint 3 evolves RetrievLab into a production-grade **Two-Stage Retrieval System**:
1. **Stage 1 (Candidate Generation)**: Fast, high-recall candidate generation ($K_{\text{cand}} \in [50, 200]$) using BM25, Dense (FAISS), and Hybrid retrieval.
2. **Stage 2 (Re-Ranking & Scoring)**: High-precision scoring using multi-signal feature extraction, Cross-Encoders, and Learning-to-Rank (LightGBM / GBDT).
3. **Multi-Domain External Benchmarks**: Scaling beyond toy datasets by integrating BEIR benchmarks (`SciFact`, `NFCorpus`, `FiQA`) to evaluate out-of-domain generalization.
4. **Latency & Throughput Profiling**: Measuring p50/p95/p99 latency, query throughput (QPS), and candidate depth trade-offs.

---

# Success Criteria

- ✅ BEIR benchmark loader & automated dataset ingestion (`SciFact`, `NFCorpus`, `FiQA`)
- ✅ Graded relevance metrics (`nDCG@K`, `MAP@K`, `Hit@K`)
- ✅ Unified `CandidateGenerator` and `CandidatePool` abstraction
- ✅ Candidate depth ($K_{\text{cand}}$) vs. Recall curve analysis
- ✅ Multi-signal feature extraction engine (Lexical, Semantic, Fusion, Query difficulty, Score divergence)
- ✅ Cross-Encoder neural re-ranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- ✅ Gradient-Boosted Decision Tree (LightGBM) pairwise Learning-to-Rank model
- ✅ Candidate latency, QPS, and throughput profiling across candidate depths
- ✅ Sprint 3 research report comparing Two-Stage Retrieval against single-stage baselines

---

# 🏛️ Epic 1 — Multi-Domain Benchmarks & Dataset Scaling (BEIR Integration)

**Goal**

Integrate standardized, multi-domain evaluation datasets from the BEIR benchmark suite to measure out-of-domain generalization and evaluate retrieval on multi-thousand document corpora.

---

## 🎟️ RLB-300 — BEIR Benchmark Loader & Dataset Adapter

### Description
Implement an automated loader and caching layer for BEIR datasets, supporting standardized document schemas, queries, and graded relevance assessments (`qrels`).

### Tasks
- [ ] Implement `BEIRLoader` in `retrievlab.ingestion.beir` (download, parse, cache)
- [ ] Support `SciFact` (scientific claim verification, ~5K docs)
- [ ] Support `NFCorpus` (biomedical search, ~3.6K docs)
- [ ] Support `FiQA` (financial question answering, ~57K docs)
- [ ] Map BEIR corpora into standard `Chunk` and `BenchmarkCase` schemas

---

## 🎟️ RLB-301 — Out-of-Domain Generalization Benchmark

### Description
Evaluate baseline BM25, FAISS Dense, and Hybrid (RRF) retrievers across all three BEIR domains.

### Research Question
> How do lexical, dense, and hybrid retrieval strategies generalize when tested on specialized out-of-domain corpora (Biomedical vs Scientific vs Financial)?

---

# 🏛️ Epic 2 — Deep Candidate Generation & Candidate Pool Profiling

**Goal**

Formalize the first-stage retrieval phase as a dedicated `CandidateGenerator` producing candidate pools, and quantify the relationship between candidate depth ($K_{\text{cand}}$), recall ceiling, and execution latency.

---

## 🎟️ RLB-310 — Candidate Generator & Candidate Pool Abstraction

### Description
Design and implement `CandidateGenerator` and `CandidatePool` in `retrievlab.selection.candidate`.

### Tasks
- [ ] Define `CandidateGenerator` abstract base class
- [ ] Define `CandidatePool` data structure (holding candidate chunks, initial retrieval ranks, and per-retriever raw scores)
- [ ] Implement `MultiRetrieverCandidateGenerator` (unioning candidates from BM25 + FAISS Dense)

---

## 🎟️ RLB-311 — Candidate Pool Depth vs. Recall Ceiling Analysis

### Description
Analyze candidate recall as candidate depth sweeps from $K_{\text{cand}} \in [10, 20, 50, 100, 200, 500]$.

### Research Question
> What candidate depth ($K_{\text{cand}}$) is required to achieve $\ge 98\%$ recall ceiling before passing candidates to the re-ranking stage?

---

## 🎟️ RLB-312 — Candidate Generation Latency & Throughput Profiling

### Description
Benchmark query latency (p50, p95, p99) and QPS across candidate pool sizes ($K_{\text{cand}}$) to identify stage 1 latency bottlenecks.

---

# 🏛️ Epic 3 — Multi-Signal Feature Extraction Engine

**Goal**

Build a modular feature extraction pipeline that computes lexical, dense, rank, and statistical signals for any `(query, chunk)` pair in a candidate pool.

---

## 🎟️ RLB-320 — Multi-Signal Feature Extraction Engine

### Description
Implement `FeatureExtractor` in `retrievlab.features.extractor`.

### Feature Suite
1. **Lexical Signals**: BM25 score, raw term frequency, exact match boolean, token overlap ratio.
2. **Dense Signals**: FAISS cosine similarity score, dense ranking position.
3. **Fusion Signals**: RRF score, rank reciprocal ($1 / r$).
4. **Query & Document Statistics**: Query token length, chunk token length, document length ratio, query term entropy.
5. **Cross-Modality Divergence**: Discrepancy between BM25 rank and Dense rank ($|r_{\text{lex}} - r_{\text{dense}}|$).

---

## 🎟️ RLB-321 — Feature Importance & Signal Correlation Diagnostics

### Description
Analyze the Pearson/Spearman correlation of each extracted feature against true query-chunk relevance labels.

---

# 🏛️ Epic 4 — Learning-to-Rank (LTR) & Cross-Encoder Re-Ranking

**Goal**

Implement second-stage scoring models to re-rank the candidate pool and maximize top-K precision ($P@K$) and $n\text{DCG}@K$.

---

## 🎟️ RLB-330 — Cross-Encoder Neural Re-Ranker

### Description
Implement `CrossEncoderReRanker` in `retrievlab.ranking.cross_encoder` wrapping `sentence-transformers` (`cross-encoder/ms-marco-MiniLM-L-6-v2`).

### Tasks
- [ ] Full sequence interaction scoring `(query, text)`
- [ ] Batch inference optimization
- [ ] Top-K re-ranking

---

## 🎟️ RLB-331 — Gradient-Boosted Decision Tree (LightGBM) Pairwise Ranker

### Description
Train and evaluate a LightGBM LambdaMART / Pairwise LTR model using the extracted feature matrix.

### Tasks
- [ ] Implement `LightGBMRanker` in `retrievlab.ranking.lightgbm`
- [ ] Train on benchmark train/dev splits with `lambdarank` objective
- [ ] Evaluate re-ranking quality and inference latency vs Cross-Encoder

---

## 🎟️ RLB-332 — End-to-End Two-Stage Retrieval Benchmark

### Description
Execute an end-to-end comparative experiment evaluating:
1. Single-stage BM25
2. Single-stage Dense (FAISS)
3. Single-stage Hybrid (RRF)
4. Two-stage: Hybrid Candidate Generator ($K=100$) + LightGBM Re-Ranker
5. Two-stage: Hybrid Candidate Generator ($K=50$) + Cross-Encoder Re-Ranker

---

# 🏛️ Epic 5 — Advanced Graded Relevance Metrics & Sprint Summary

**Goal**

Expand the evaluation suite with graded relevance metrics and produce the comprehensive Sprint 3 Research Report.

---

## 🎟️ RLB-340 — Graded Relevance Metrics (`nDCG@K`, `MAP@K`, `Hit@K`)

### Description
Implement graded ranking metrics in `retrievlab.evaluation.metrics`:
- [ ] `ndcg_at_k(retrieved_results, qrels, k=10)` (Normalized Discounted Cumulative Gain)
- [ ] `map_at_k(retrieved_results, qrels, k=10)` (Mean Average Precision)
- [ ] `hit_at_k(retrieved_results, qrels, k=10)`

---

## 🎟️ RLB-341 — Sprint 3 Research Report

### Deliverable
```text
results/
    sprint_3_report.md
```
Comprehensive report synthesizing:
- Multi-domain BEIR benchmark results
- Candidate pool depth vs recall curves
- Feature importance analysis
- Stage 2 re-ranking quality ($n\text{DCG}@10$) vs Stage 1 baselines
- End-to-end latency and throughput trade-off matrix
