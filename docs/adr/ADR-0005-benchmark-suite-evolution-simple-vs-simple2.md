# ADR-0005: Benchmark Suite Evolution (simple.json vs. simple2.json)

**Status**: Accepted  
**Deciders**: RetrievLab Team  
**Date**: 2026-08-14  

---

## Context
In Sprint 1, RetrievLab introduced `data/benchmarks/simple.json` containing 14 standard conversational question queries (*"What is FastAPI?"*, *"How do I install FastAPI?"*, *"What is Docker?"*).

While `simple.json` served as an effective initial sanity check, it lacked intentional query stratification. To evaluate where sparse lexical models (BM25) and dense semantic models fail, experiments required distinct query categories:
1. **Pure Lexical Queries**: Exact keywords, package names, and technical terms (testing term specificity).
2. **Pure Semantic Paraphrases**: Conceptual queries with zero keyword overlap with the corpus (testing semantic generalization).

## Decision
1. **Preserve `data/benchmarks/simple.json` (14 Cases)**:
   - Kept as a lightweight benchmark for unit testing (`tests/test_benchmark.py`) and fast local validation.
2. **Introduce `data/benchmarks/simple2.json` (22 Cases) as Evaluation Standard**:
   - Designed as a complete superset containing:
     - **Cases 1–14**: The original 14 baseline queries from `simple.json`.
     - **Cases 15–18 (Lexical Subset)**: Exact keyword queries (`"Pydantic and Starlette"`, `"Uvicorn deployment"`, `"Kubernetes container orchestration"`, `"async await syntax"`).
     - **Cases 19–22 (Semantic Subset)**: Semantic paraphrases (`"modern high performance web framework"`, `"isolated containerized runtime environment"`, `"object oriented procedural and functional scripting language"`, `"asynchronous background execution"`).
3. **Adopt `simple2.json` for Comprehensive Sprint Runners**:
   - Experiment runners (`run_sprint1_experiments.py`, Sprint 2 evaluators) target `simple2.json` to compute **Overall Metrics**, **Lexical Study Metrics**, and **Semantic Study Metrics** in a single pass.

## Consequences

### Positive
- Unit tests remain fast and lightweight using `simple.json`.
- Comprehensive experiments can report both overall aggregate metrics and stratified query-category breakdowns from a single dataset file.
- Clean backward compatibility with early experiment logs and reports.

### Negative / Tradeoffs
- Multiple benchmark files require explicit documentation to avoid ambiguity regarding which dataset was used in a given report.

## Related ADRs & Files
- [ADR-0003: Benchmark Schema & Loader Design](file:///e:/Downloads/RetrievLab/docs/adr/ADR-0003-benchmark-schema.md)
- [`data/benchmarks/simple.json`](file:///e:/Downloads/RetrievLab/data/benchmarks/simple.json)
- [`data/benchmarks/simple2.json`](file:///e:/Downloads/RetrievLab/data/benchmarks/simple2.json)
