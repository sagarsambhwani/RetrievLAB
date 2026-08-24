# 🗓️ RetrievLab Sprint 2

**Sprint:** Retrieval Evolution — From Baselines to Hybrid Systems  
**Duration:** 1 Week  
**Status:** ⚪ Planned

---

# 🎯 Sprint Vision

Sprint 1 established reliable retrieval baselines and a reproducible evaluation framework.

Sprint 2 focuses on **improving retrieval quality through experimentation**.

Rather than introducing new infrastructure for its own sake, every implementation in this sprint should answer a retrieval question through measurable experiments.

By the end of this sprint, RetrievLab should not only support hybrid retrieval, but also explain **why** hybrid retrieval works, **where** it works, and **when** it fails.

---

# Success Criteria

- ✅ BM25 preprocessing pipeline (stemming & stopword filtering)
- ✅ Configurable BM25 parameters
- ✅ Reciprocal Rank Fusion (Hybrid Retrieval)
- ✅ Hybrid retrieval evaluation
- ✅ Retrieval failure analysis
- ✅ Sprint research report documenting findings

---

# 🏛️ Epic 1 — Lexical Retrieval Evolution & Preprocessing Architecture

**Goal**

Establish a unified, multi-level preprocessing architecture (`retrievlab.preprocessing`) that allows interchangeable tokenization strategies (Word-level, Character/N-Gram-level, Subword-level, and Composable Pipelines) to be injected into lexical retrievers without altering algorithm implementations (adhering to `design_principles.md`).

---

## 🎟️ RLB-200 — Preprocessing Architecture & Abstract Interface

### Description

Design and implement the core `BaseTokenizer` abstraction layer to decouple tokenization from retrieval algorithms.

### Deliverables

- [ ] `BaseTokenizer` abstract base class in `retrievlab.preprocessing.interface`
- [ ] Unified `tokenize(text: str) -> list[str]` contract
- [ ] Type definitions and docstring standardizations

---

## 🎟️ RLB-201 — Multi-Level Tokenizer Suite

### Description

Implement tokenizers operating across different structural levels:

### Tokenizer Levels

- [ ] **Word-Level** (`BasicWordTokenizer`, `RegexTokenizer`, `StopwordTokenizer`, `StemmedTokenizer`)
- [ ] **N-Gram Level** (`CharNGramTokenizer`, `WordNGramTokenizer`)
- [ ] **Subword Level** (`SubwordTokenizer` — BPE / Tiktoken / HuggingFace adapter)
- [ ] **Pipeline Level** (`PipelineTokenizer` — composable normalization, tokenization, filtering, and stemming)

### Research Question

> How does tokenization granularity (Word vs. Char N-Gram vs. Subword vs. Stemming) impact lexical recall and index size?

---

## 🎟️ RLB-202 — Configurable Retriever Tokenizer Injection

### Description

Update `BM25Retriever` to accept any `BaseTokenizer` via dependency injection, keeping the BM25 scoring algorithm untouched (`design_principles.md` Rule 2).

### Tasks

- [ ] Expose `tokenizer` parameter in `BM25Retriever.__init__`
- [ ] Inject tokenizer into index building and query retrieval phases
- [ ] Maintain backward compatibility with default `BasicWordTokenizer`

---

## 🎟️ RLB-203 — BM25 Parameter Exploration

### Description

Expose BM25 parameters as configurable settings.

### Tasks

- [ ] Configurable k₁
- [ ] Configurable b
- [ ] Benchmark multiple parameter combinations

### Research Question

> Which parameter configuration performs best for our benchmark corpus?

---

## 🎟️ RLB-204 — Multi-Level Tokenization Benchmark & Lexical Evolution Report

### Description

Evaluate BM25 across all tokenization levels and parameter configurations against the Sprint 1 baseline.

### Deliverable

Comparative report covering:
- Baseline (Basic Word Tokenizer)
- + Stopword Filtering
- + Porter Stemming
- Character $n$-grams (3-gram to 5-gram)
- Subword / BPE tokenization
- Parameter tuning ($k_1, b$)

---

# 🏛️ Epic 2 — Hybrid Retrieval

**Goal**

Combine lexical and semantic retrieval into a stronger candidate generator.

---

## 🎟️ RLB-210 — Reciprocal Rank Fusion

### Description

Implement Reciprocal Rank Fusion as the first hybrid retrieval strategy.

### Tasks

- [x] Generic RRF implementation
- [x] Support multiple retrievers
- [x] Configurable fusion constant
- [x] Unit tests

### Research Question

> Can rank fusion outperform individual retrievers?

---

## 🎟️ RLB-211 — Hybrid Retriever

### Description

Implement a Hybrid Retriever using BM25 + Dense retrieval.

### Tasks

- [x] Execute both retrievers
- [x] Fuse rankings
- [x] Return unified SearchResults

### Acceptance Criteria

HybridRetriever implements the Retriever interface.

---

## 🎟️ RLB-212 — Hybrid Failure Analysis

### Description

Compare BM25, Dense and Hybrid on every benchmark query.

Automatically detect:

- [ ] BM25 failures recovered by Hybrid
- [ ] Dense failures recovered by Hybrid
- [ ] Queries where Hybrid performs worse
- [ ] Ranking disagreements

### Deliverable

Query-level comparison report.

---

# 🏛️ Epic 3 — Retrieval Research

**Goal**

Turn experiments into reproducible research.

---

## 🎟️ RLB-220 — Experiment Framework Expansion

### Tasks

- [ ] Standard experiment template
- [ ] Automatic metric collection
- [ ] Markdown report generation
- [ ] Experiment metadata

---

## 🎟️ RLB-221 — Sprint 2 Research Report

### Description

Produce a report answering:

- Did stemming improve BM25?
- Did stopword removal help?
- Did parameter tuning matter?
- Did Hybrid outperform BM25?
- Did Hybrid outperform Dense?
- Which queries changed ranking?
- Which failures remain unsolved?

Deliverable:

```
results/
    sprint_2_report.md
```

---

# 🏛️ Epic 4 — Performance Foundations

**Goal**

Prepare RetrievLab for larger datasets without changing retrieval behavior.

---

## 🎟️ RLB-230 — FAISS Integration

### Description

Introduce FAISS as an interchangeable dense retrieval backend.

### Tasks

- [ ] Build FAISS index
- [ ] Validate retrieval equivalence
- [ ] Compare retrieval latency
- [ ] Benchmark against brute-force search

### Research Question

> Can we improve scalability while preserving retrieval quality?

---

# 📈 Sprint Philosophy

Every implementation should answer a measurable question.

Every feature should end with an experiment.

Every experiment should produce a report.

Every report should improve our understanding of retrieval systems.

RetrievLab is not only a retrieval library.

It is a laboratory for retrieval research.
