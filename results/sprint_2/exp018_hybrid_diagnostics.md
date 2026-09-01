# Experiment 018 Report: Hybrid Retrieval Failure Diagnostics & Outcome Taxonomy

**Date**: 2026-09-01  
**Status**: Completed  
**Benchmark Suite**: `data/benchmarks/simple2.json` (22 test cases)  
**Corpus**: `data/raw/` (4 documents, 9 chunks)  
**Evaluated Systems**: `BM25Retriever` ($k_1=1.5, b=0.75$), `DenseRetriever` (`bge-small-en-v1.5`), `HybridRetriever` (RRF weights $w=[1.0, 2.0], k=60$)  

---

## 1. Executive Summary

This experiment validates RetrievLab's automated diagnostic and failure analysis tooling ([`src/retrievlab/evaluation/diagnostics.py`](file:///e:/Downloads/RetrievLab/src/retrievlab/evaluation/diagnostics.py)).

### Key Findings
1. **Automated Recovery Detection:** The diagnostic engine successfully identified **Query 6** (*"How can FastAPI be deployed?"*) as an isolated `DENSE_WIN_HYBRID_RECOVERED` case. BM25 completely missed the passage (Recall=0.0) due to vocabulary mismatch, Dense retrieved it at rank #5, and Hybrid successfully pulled it into the Top-5 search results.
2. **Zero Degradations:** Across all 22 queries, there were **0 cases** where Hybrid degraded results relative to the individual baselines (`hybrid_degradation = 0`).
3. **High Concordance:** **21 out of 22 queries (95.45%)** were `joint_hit` cases where both lexical and semantic channels agreed.

---

# Hybrid Failure Analysis & Recovery Diagnosis

## 1. Outcome Distribution Summary

- **Total Benchmark Queries:** 22
- **Joint Hits (Both BM25 & Dense Succeeded):** 21
- **Dense Wins Recovered by Hybrid:** 1
- **BM25 Wins Recovered by Hybrid:** 0
- **Hybrid Degradations (Recall Loss):** 0
- **Joint Misses (All Failed):** 0

## 2. Query-by-Query Diagnostic Breakdown

| # | Query | BM25 Rank | Dense Rank | Hybrid Rank | Category | Status |
| :---: | :--- | :---: | :---: | :---: | :--- | :---: |
| 1 | What is FastAPI? | #1 | #1 | #1 | joint_hit | [OK] |
| 2 | How do I install FastAPI? | #1 | #1 | #1 | joint_hit | [OK] |
| 3 | What defines API endpoints in FastAPI? | #1 | #2 | #2 | joint_hit | [OK] |
| 4 | Does FastAPI support dependency injection? | #1 | #1 | #1 | joint_hit | [OK] |
| 5 | What are background tasks in FastAPI? | #1 | #1 | #1 | joint_hit | [OK] |
| 6 | How can FastAPI be deployed? | - | #5 | #5 | dense_win_hybrid_recovered | [RECOVERED] |
| 7 | What is Docker? | #1 | #1 | #1 | joint_hit | [OK] |
| 8 | What are the key features of Docker? | #1 | #1 | #1 | joint_hit | [OK] |
| 9 | What is Python? | #1 | #1 | #1 | joint_hit | [OK] |
| 10 | What are the key features of Python? | #1 | #1 | #1 | joint_hit | [OK] |
| 11 | Does Python support multiple programming paradigms... | #1 | #1 | #1 | joint_hit | [OK] |
| 12 | Does FastAPI support async and await? | #1 | #1 | #1 | joint_hit | [OK] |
| 13 | What frameworks and libraries is FastAPI built on? | #1 | #1 | #1 | joint_hit | [OK] |
| 14 | What are the key features of FastAPI? | #2 | #1 | #1 | joint_hit | [OK] |
| 15 | Pydantic and Starlette | #1 | #1 | #1 | joint_hit | [OK] |
| 16 | Uvicorn deployment | #1 | #1 | #1 | joint_hit | [OK] |
| 17 | Kubernetes container orchestration | #1 | #1 | #1 | joint_hit | [OK] |
| 18 | async await syntax | #1 | #3 | #1 | joint_hit | [OK] |
| 19 | modern high performance web framework | #1 | #1 | #1 | joint_hit | [OK] |
| 20 | isolated containerized runtime environment | #1 | #1 | #1 | joint_hit | [OK] |
| 21 | object oriented procedural and functional scriptin... | #1 | #1 | #1 | joint_hit | [OK] |
| 22 | asynchronous background execution | #1 | #1 | #1 | joint_hit | [OK] |
