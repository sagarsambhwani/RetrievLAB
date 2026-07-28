# 🗓️ RetrievLab Sprint Plan
**Sprint:** Sprint 1 — Retrieval Foundations  
**Duration:** 1 Week  
**Status:** 🚧 In Progress

---

# 🎯 Sprint Goal

By the end of this sprint, RetrievLab should be capable of **scientifically comparing multiple retrieval strategies** on a benchmark dataset.

This milestone establishes the foundation for every future retrieval experiment including Hybrid Retrieval, Reranking, Learning-to-Rank, and Agentic Retrieval.

---

# Success Criteria

By the end of the sprint, the project should support:

- ✅ Dense Retrieval
- ✅ BM25 Retrieval
- ✅ Standard Retriever Interface
- ✅ Benchmark Dataset
- ✅ Evaluation Metrics
- ✅ Experiment Runner
- ✅ Benchmark Report

The goal is not to build the most advanced retriever.

The goal is to build the **first complete retrieval experimentation pipeline.**

---

# 🏛️ Epic 1 — Retrieval Foundations

**Goal**

Finish implementing and validating the two baseline retrieval methods that every future experiment will compare against.

---

## 🎟️ RLB-001 — Finalize BM25 Retriever

**Priority:** High

### Description

Complete the BM25 implementation and ensure it behaves consistently across different query types.

### Tasks

- [x] Tokenizer
- [x] Index Builder
- [x] Term Frequencies
- [x] Inverse Document Frequency
- [x] BM25 Scoring
- [x] Retrieval Logic
- [x] Unit Tests
- [x] Edge Case Testing
- [x] Performance Review
- [x] Code Cleanup

### Acceptance Criteria

- BM25 returns deterministic rankings.
- Unknown terms return zero score.
- Empty corpora are handled gracefully.
- All public methods include documentation.
- Retrieval quality matches expected intuition.

### Estimated Effort

**2 Hours**

### Current Limitations: 
The baseline BM25 implementation uses simple tokenization without stemming, lemmatization, or stopword removal. These preprocessing techniques will be evaluated in future experiments to improve lexical matching.

---

## 🎟️ RLB-002 — Review Dense Retriever

**Priority:** High

### Description

Review the existing Dense Retriever implementation to ensure it follows the same engineering standards as BM25.

### Tasks

- [x] Review embedding generation
- [x] Verify cosine similarity implementation
- [x] Improve documentation
- [x] Verify interface consistency
- [x] Add tests

### Acceptance Criteria

- ✅ Retrieval quality is verified.
- ✅ Documentation is complete.
- ✅ Matches Retriever interface.

### Estimated Effort

**2 Hours**

### Completed Work

- Added comprehensive class and method docstrings (Google style)
- Renamed `similarity()` to `_similarity()` for proper encapsulation
- Added validation for embedding dimensions with descriptive errors
- Added error handling for missing embeddings
- Created 8 unit tests covering all edge cases and interface compliance
- All tests passing

---

# 🏛️ Epic 2 — Benchmark Infrastructure

**Goal**

Introduce a reproducible benchmark format that every retrieval algorithm can evaluate against.

---

## 🎟️ RLB-010 — Design Benchmark Schema

**Priority:** High

### Description

Define a simple benchmark format containing queries and relevant chunks.

### Example

```yaml
query: "What is FastAPI?"
relevant_chunks:
  - fastapi.md:1
```

### Tasks

- [x] Create benchmark schema
- [x] Validate schema
- [x] Create example benchmark

### Deliverable

```
benchmarks/
    simple.json
```

### Estimated Effort

**1 Hour**

### Change in Decision

We will use JSON instead of YAML for benchmarks because JSON benchmark structures were already present in the codebase and are easier to work with.

### Deliverable Changed

```json
{
    "query": "What is FastAPI?",
    "relevant_chunks": [
        "fastapi.md:1"
    ]
}
```

we will use the same 

Schema used:

Benchmark Case

- query: str
- relevant_chunk_ids: list[str]

Benchmark

- cases: list[BenchmarkCase]  ->  list of BenchmarkCase objects

---

## 🎟️ RLB-011 — Implement Benchmark Loader

**Priority:** High

### Description

Implement a loader that converts benchmark files into Python objects.

### Tasks

- [x] Load JSON benchmark
- [x] Validate required fields
- [x] Return Benchmark object

### Acceptance Criteria

Benchmark files load without manual parsing.

### Estimated Effort

**2 Hours**

### Delivered

we added json schema for benchmarks

we added example benchmark in benchmarks/simple.json

we added benchmark loader that converts benchmark files into Python objects

now we can load benchmarks using load_benchmark function

---

## 🎟️ RLB-012 — Benchmark Models

**Priority:** Medium

### Description

Create strongly typed benchmark models.

### Models

- Benchmark
- QueryExample
- GroundTruth

### Estimated Effort

**1 Hour**

### Schema Implemented

Benchmark Case

- query: str
- relevant_chunk_ids: list[str]

Benchmark

- cases: list[BenchmarkCase]  ->  list of BenchmarkCase objects

---

# 🏛️ Epic 3 — Evaluation Framework

**Goal**

Measure retrieval quality instead of relying on intuition.

---

## 🎟️ RLB-020 — Recall@K

**Priority:** High

### Tasks

- [ ] Recall@1
- [ ] Recall@3
- [ ] Recall@5
- [ ] Recall@10

### Acceptance Criteria

Correctly computes recall over benchmark queries.

### Estimated Effort

**2 Hours**

---

## 🎟️ RLB-021 — Mean Reciprocal Rank (MRR)

**Priority:** High

### Description

Measure ranking quality by considering the first relevant retrieved chunk.

### Tasks

- [ ] Reciprocal Rank
- [ ] Mean Reciprocal Rank

### Estimated Effort

**2 Hours**

---

## 🎟️ RLB-022 — Precision@K

**Priority:** Medium

### Tasks

- [ ] Precision@1
- [ ] Precision@5
- [ ] Precision@10

### Estimated Effort

**2 Hours**

---

## 🎟️ RLB-023 — Evaluation Report

**Priority:** Medium

### Description

Generate a simple evaluation summary after running benchmarks.

### Example Output

| Retriever | Recall@5 | MRR | Precision@5 |
|------------|---------:|----:|------------:|
| Dense | 0.82 | 0.71 | 0.76 |
| BM25 | 0.79 | 0.74 | 0.72 |

### Estimated Effort

**2 Hours**

---

# 🏛️ Epic 4 — Retrieval Experiments

**Goal**

Turn RetrievLab into a research playground rather than just another retrieval library.

---

## 🎟️ RLB-030 — BM25 vs Dense

**Priority:** High

### Description

Run both retrievers against the benchmark dataset.

### Questions to Answer

- Which retriever performs better?
- Which query types favor BM25?
- Which query types favor Dense Retrieval?

### Deliverable

Experiment report.

### Estimated Effort

**2 Hours**

---

## 🎟️ RLB-031 — Lexical Query Study

**Priority:** Medium

### Example Queries

- docker compose
- dependency injection
- async await
- python decorators

### Goal

Understand when BM25 outperforms Dense Retrieval.

### Estimated Effort

**1 Hour**

---

## 🎟️ RLB-032 — Semantic Query Study

**Priority:** Medium

### Example Queries

- modern web framework
- asynchronous python api
- backend development framework

### Goal

Understand when Dense Retrieval wins.

### Estimated Effort

**1 Hour**

---

## 🎟️ RLB-033 — Experiment Report

**Priority:** Medium

### Description

Summarize experiment findings.

### Questions

- What worked?
- What failed?
- What surprised us?
- What should be improved next?

### Deliverable

Markdown report.

### Estimated Effort

**1 Hour**

---

# 🏛️ Epic 5 — Project Cleanup

**Goal**

Improve maintainability before introducing additional retrieval techniques.

---

## 🎟️ RLB-040 — Code Cleanup

### Tasks

- [ ] Remove dead code
- [ ] Improve variable naming
- [ ] Remove duplication

---

## 🎟️ RLB-041 — Documentation

### Tasks

- [ ] Update README
- [ ] Add architecture overview
- [ ] Document project structure

---

## 🎟️ RLB-042 — API Documentation

### Tasks

- [ ] Review public docstrings
- [ ] Verify typing
- [ ] Improve examples

---

# 📈 Sprint Progress

| Epic | Status |
|-------|--------|
| Retrieval Foundations | ✅ Complete |
| Benchmark Infrastructure | ✅ Complete |
| Evaluation Framework | ⚪ Planned |
| Experiments | ⚪ Planned |
| Cleanup | ⚪ Planned |

---

# 🚀 Stretch Goals (Only If Time Permits)

These stories should only begin once every sprint goal has been completed.

- [ ] Reciprocal Rank Fusion (RRF)
- [ ] Better Tokenizer
- [ ] SentenceTransformer Embeddings
- [ ] FAISS Index
- [ ] BM25+
- [ ] Query Expansion

---

# 📌 Sprint Deliverable

By the end of this sprint, RetrievLab should support the following workflow:

```text
Documents
     │
     ▼
Document Loader
     │
     ▼
Chunker
     │
     ▼
Embedding Model
     │
     ├──────────────┐
     ▼              │
Dense Retriever     │
                    │
BM25 Retriever      │
     └──────┬───────┘
            ▼
    Benchmark Runner
            ▼
     Evaluation Metrics
            ▼
     Experiment Report
```

---

# 📝 Sprint Retrospective (To Complete at End of Sprint)

## What went well?

-

## What didn't go well?

-

## What did we learn?

-

## Technical debt introduced

-

## Next sprint focus

-
