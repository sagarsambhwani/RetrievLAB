# Evaluation Roadmap

> Goal: Build a simple, extensible evaluation framework for retrieval systems.

---

# Philosophy

Evaluation exists to answer one question:

> **How good is a retrieval pipeline?**

A retrieval pipeline is only useful if its performance can be measured and compared.

RetrievLab aims to evaluate retrieval algorithms independently of embedding providers, vector databases, or user interfaces.

---

# Guiding Principles

- Start with the simplest possible evaluation.
- Build from first principles.
- Evaluation should be independent of the retrieval implementation.
- Every experiment should be reproducible.
- Optimize for understanding before benchmark performance.

---

# Phase 1 — Foundation

## Objective

Evaluate a single retriever against a handcrafted benchmark.

### Tasks

- [ ] Design benchmark data model
- [ ] Design benchmark case
- [ ] Create a small benchmark
- [ ] Implement Recall@K
- [ ] Validate Recall@K using Dense Retrieval

---

# Phase 2 — Core Metrics

Implement classical Information Retrieval metrics.

- [ ] Precision@K
- [ ] Recall@K
- [ ] Hit Rate
- [ ] MRR
- [ ] MAP
- [ ] nDCG

Each metric should implement a common Evaluator interface.

---

# Phase 3 — Benchmark Support

Support benchmark datasets.

Initially:

- [ ] Handcrafted benchmark
- [ ] Local benchmark loading

Later:

- [ ] BEIR
- [ ] MS MARCO
- [ ] MIRACL

---

# Phase 4 — Experiment Framework

Every experiment should answer one research question.

Example:

Question

> Does BM25 outperform Dense Retrieval?

Pipeline

Dense Retriever

Benchmark

Simple Benchmark

Metrics

Recall@10

MRR

Expected Result

Dense Retrieval should retrieve semantic matches better.

---

# Phase 5 — Benchmark Comparison

Compare multiple retrieval pipelines.

Example

Dense

↓

Recall@10

↓

MRR

↓

Latency

--------------------------------

BM25

↓

Recall@10

↓

MRR

↓

Latency

--------------------------------

Hybrid

↓

Recall@10

↓

MRR

↓

Latency

---

# Phase 6 — Feature Evaluation

As new retrieval signals are added, evaluate them individually.

Examples

- Dense Similarity
- BM25 Score
- Metadata Match
- Heading Match
- Cross Encoder Score

The objective is to determine which signals contribute most to ranking quality.

---

# Phase 7 — Learning-to-Rank

Evaluate ranking models.

Examples

- LambdaMART
- LightGBM Ranker
- XGBoost Ranker

Metrics remain identical.

Only the ranking model changes.

---

# Phase 8 — Adaptive Learning-to-Rank

Final objective.

Pipeline

Documents

↓

Retriever

↓

Candidate Generation

↓

Feature Engineering

↓

Learning-to-Rank

↓

Adaptive Ranker

↓

Evaluation

Research Questions

- Does Adaptive LTR outperform Dense Retrieval?
- Does Adaptive LTR outperform Hybrid Retrieval?
- Which features are most important?
- Which retrieval strategy performs best for different query types?

---

# Open Design Questions

These are intentionally postponed.

## Benchmark Design

- Chunk-level relevance?
- Document-level relevance?
- Passage-level relevance?

## Stable Identifiers

- UUID
- Deterministic IDs
- Content Hashes

## Dataset Support

- Local benchmarks
- Public benchmarks
- Synthetic benchmarks

These decisions should be made only when required.

---

# Current Focus

Only implement:

- Benchmark data model
- Recall@K
- Dense Retrieval evaluation

Nothing else.

Everything above this point exists only as a roadmap.