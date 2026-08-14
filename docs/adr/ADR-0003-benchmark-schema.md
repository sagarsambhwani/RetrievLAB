# ADR-0003: Benchmark Schema & Loader Design

**Status**: Accepted  
**Deciders**: RetrievLab Team  
**Date**: 2026-08-14  

---

## Context
RetrievLab requires reproducible, automated evaluation against standardized question-passage pairs to compare retrieval quality across algorithms and configurations. We needed a benchmark representation that is easy to author manually, fast to parse programmatically, and simple to validate.

## Decision
Use structured JSON format with Pydantic-validated models (`BenchmarkCase` and `Benchmark`) for all local benchmark suites:

```json
{
  "name": "Benchmark Name",
  "version": "1.0",
  "description": "Benchmark description...",
  "cases": [
    {
      "query": "What is FastAPI?",
      "relevant_chunk_ids": ["fastapi.md:1"],
      "query_type": "lexical"
    }
  ]
}
```

## Consequences
### Positive
- Strict Pydantic validation guarantees schema adherence at load time.
- JSON format is natively supported across all Python environments with zero external dependencies.
- Extensible schema supports query categorizations (`lexical`, `semantic`, `hybrid`, `edge_case`).

### Negative / Tradeoffs
- JSON lacks comment support for handcrafted annotations (which can be mitigated via top-level `description` and `notes` metadata fields).
