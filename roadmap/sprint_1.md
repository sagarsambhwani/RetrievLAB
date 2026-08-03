# 🗓️ RetrievLab Sprint Plan
**Sprint:** Sprint 1 — Retrieval Foundations  
**Duration:** 1 Week  
**Status:** ✅ Complete

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

- [x] Recall@1
- [x] Recall@3
- [x] Recall@5
- [x] Recall@10

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

- [x] Reciprocal Rank
- [x] Mean Reciprocal Rank

### Estimated Effort

**2 Hours**

---

## 🎟️ RLB-022 — Precision@K

**Priority:** Medium

### Tasks

- [x] Precision@1
- [x] Precision@5
- [x] Precision@10

### Estimated Effort

**2 Hours**

---

## 🎟️ RLB-023 — Evaluation Report

**Priority:** Medium

### Description

Generate a simple evaluation summary after running benchmarks.

### Tasks

- [x] Evaluation report models (`RetrieverEvaluationResult`, `EvaluationReport`)
- [x] Markdown table formatting (`to_markdown`)
- [x] Retriever evaluator (`evaluate_retriever`)

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

- [x] Run both retrievers against benchmark dataset
- [x] Identify query types favoring BM25 vs Dense

### Deliverable

Experiment report deliverable generated at `results/sprint_1_experiment_report.md`.

---

## 🎟️ RLB-031 — Lexical Query Study

**Priority:** Medium

### Tasks

- [x] Benchmark lexical query cases (`Pydantic and Starlette`, `Uvicorn deployment`, `Kubernetes container orchestration`, `async await syntax`)
- [x] Analyze exact keyword matching performance

---

## 🎟️ RLB-032 — Semantic Query Study

**Priority:** Medium

### Tasks

- [x] Benchmark semantic query cases (`modern high performance web framework`, `isolated containerized runtime environment`, `object oriented procedural and functional scripting language`)
- [x] Analyze vector similarity performance on conceptual queries

---

## 🎟️ RLB-033 — Experiment Report

**Priority:** Medium

### Tasks

- [x] Summarize overall aggregate metrics
- [x] Document lexical vs semantic study findings
- [x] Write comprehensive report deliverable (`results/sprint_1_experiment_report.md`)

---

# 🏛️ Epic 5 — Project Cleanup

**Goal**

Improve maintainability before introducing additional retrieval techniques.

---

## 🎟️ RLB-040 — Code Cleanup

### Tasks

- [x] Remove dead code
- [x] Improve variable naming
- [x] Remove duplication

---

## 🎟️ RLB-041 — Documentation

### Tasks

- [x] Prepare README update draft for approval
- [x] Add architecture overview
- [x] Document project structure

---

## 🎟️ RLB-042 — API Documentation

### Tasks

- [x] Review public docstrings
- [x] Verify typing
- [x] Improve examples

---

# 📈 Sprint Progress

| Epic | Status |
|-------|--------|
| Retrieval Foundations | ✅ Complete |
| Benchmark Infrastructure | ✅ Complete |
| Evaluation Framework | ✅ Complete |
| Experiments | ✅ Complete |
| Cleanup | ✅ Complete |

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
As of now we have 

- A solid foundation for retrieval evaluation

- a simple pipeline for evaluation

- Designed a architecture for extensibility

- Worked with two different retrieval algorithms

- Built a working pipeline for evaluation report

## What didn't go well?

- Bm25 is not completed yet because it requires stemming and stopword removal 

## What did we learn?

- Understood benchmarks, metrics and evaluation
- Understood system and interface design

## Technical debt introduced

- Baseline BM25 tokenization lacks stemming/stopwords.
- Dense retriever uses linear search; needs FAISS/ANN index for scaling.

## Next sprint focus

- Implement Stemming & Stopword filtering for BM25.
- Implement Hybrid Retrieval with Reciprocal Rank Fusion (RRF).
- Integrate FAISS vector index for Dense retrieval scaling.
