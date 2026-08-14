# 🔍 Repository Research & Architecture Plan

**Goal**: Analyze mature retrieval and evaluation frameworks to identify clean design patterns, avoid premature generalizations, and document our design choices through Architecture Decision Records (ADRs).

This phase prioritizes architectural design study before writing code, ensuring RetrievLab is structured as an extensible laboratory rather than a rigid pipeline.

---

## 🗺️ Scope of Research

We will study **three representative repositories** that address our immediate goals:

| Category | Repository | Study Focus |
|:---|:---|:---|
| **Retrieval Framework** | [Haystack](https://github.com/deepset-ai/haystack) | Pipeline graphs, component registration, and evaluation abstractions. |
| **LLM RAG Orchestration** | [LlamaIndex](https://github.com/run-llama/llama_index) | Data nodes, index abstractions, and query engines. |
| **Retrieval Benchmarks** | [BEIR](https://github.com/beir-cellar/beir) | Standardized query formats, relevance mappings, and dataset splits. |

---

## 🏛️ Structured Review Template

Each repository review must be documented under `docs/research/` (e.g., `docs/research/01-haystack.md`) using the following structure:

```markdown
# Repository Review: [Repository Name]

## Purpose
[High-level goal of the project]

## Folder Structure
[How code components are organized]

## Core Abstractions
- **[Component Name 1]**: [Role/Responsibility]
- **[Component Name 2]**: [Role/Responsibility]

## Key Design Patterns Used
- [e.g., Strategy Pattern for Retrievers, registry decorator for pipelines]

## Strengths & Weaknesses
- **Strengths**: [What it does exceptionally well]
- **Weaknesses**: [Anti-patterns or complexities to avoid]

## 💡 Lessons for RetrievLab
- **What to Adopt**: [1-3 specific concepts we should build]
- **What to Avoid**: [1-2 complexities or premature generalizations to ignore]
```

---

## 📝 Deliverables & File Structure

This phase will populate the following hierarchy in our `docs/` folder:

```text
docs/
├── adr/                         # Architecture Decision Records
│   ├── ADR-0001-preamble-chunking.md
│   ├── ADR-0002-tokenizer-abstraction-and-stemming.md
│   ├── ADR-0003-benchmark-schema.md
│   ├── ADR-0004-default-embedding-model-and-assumptions.md
│   └── ADR-0005-benchmark-suite-evolution-simple-vs-simple2.md
├── patterns/                    # Documented design patterns in RetrievLab
│   ├── Strategy-Pattern.md
│   └── Registry-Pattern.md
├── research/                    # Repository analysis reports
│   ├── 01-haystack.md
│   ├── 02-llamaindex.md
│   └── 03-beir.md
└── repository_research_plan.md  # This document
```

---

## 📅 Roadmap & Milestones

### **Milestone 1: Repository Reviews**
*   Conduct analysis and write reports for:
    *   `docs/research/01-haystack.md`
    *   `docs/research/02-llamaindex.md`
    *   `docs/research/03-beir.md`

### **Milestone 2: Architecture Decision Records (ADRs)**
*   Draft ADRs to settle critical interface designs:
    *   **ADR 0001**: Retriever interface design (Index ownership vs runtime parameters).
    *   **ADR 0002**: Result schema standardization (`SearchResult` vs raw dict).
    *   **ADR 0003**: Benchmark data format selection.

### **Milestone 3: Patterns Setup**
*   Establish pattern templates in `docs/patterns/` describing where we will use Strategy and Registry patterns in RetrievLab.
