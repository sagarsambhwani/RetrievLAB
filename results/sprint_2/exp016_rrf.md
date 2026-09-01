# Experiment 016 Report: Reciprocal Rank Fusion (RRF) Parameter Study & Hybrid Evaluation

**Date**: 2026-08-23  
**Status**: Completed  
**Benchmark Suite**: `data/benchmarks/simple2.json` (22 test cases)  
**Corpus**: `data/raw/` (4 document, 9 chunks)  

---

## 1. Executive Summary

This experiment evaluates **Reciprocal Rank Fusion (RRF)** (RLB-210) as a hybrid retrieval strategy combining lexical matching (BM25 with `BasicWordTokenizer`) and semantic vector retrieval (FastEmbed `BAAI/bge-small-en-v1.5`).

We measure all three standard IR evaluation metrics: **Recall@5**, **Precision@5**, and **Mean Reciprocal Rank (MRR)**.

### Key Findings
1. **MRR Boost over Baselines**: Standard Balanced RRF ($k=60$) achieves **0.9545 MRR**, improving over both standalone BM25 (0.9318 MRR) and standalone Dense (0.9106 MRR) by reinforcing top-ranked relevant hits across both modalities.
2. **Smoothing Parameter Stability ($k$)**: Metrics remain perfectly stable across $k \in [0, 100]$, confirming Cormack et al. (2009) observations regarding RRF's parameter insensitivity.
3. **Dense-Biased Weighting (1:2)**: Weighting Dense embeddings higher ($w=[1.0, 2.0]$) attains **1.0000 Recall@5** and **0.9409 MRR**, successfully retaining Dense retrieval's 100% recall while boosting MRR over the Dense baseline (0.9106 -> 0.9409).

---

## 2. Comparative Performance Matrix

| Strategy / Configuration | k | Weights (BM25:Dense) | Recall@5 | Precision@5 | MRR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BM25 Baseline (Lexical)** | - | - | 0.9545 | 0.2182 | 0.9318 |
| **Dense Baseline (Semantic)** | - | - | 1.0000 | 0.2273 | 0.9106 |
| **RRF (k=0, Pure Reciprocal)** | 0 | [1.0, 1.0] | 0.9545 | 0.2182 | 0.9545 |
| **RRF (k=10)** | 10 | [1.0, 1.0] | 0.9545 | 0.2182 | 0.9545 |
| **RRF (k=20)** | 20 | [1.0, 1.0] | 0.9545 | 0.2182 | 0.9545 |
| **RRF (k=40)** | 40 | [1.0, 1.0] | 0.9545 | 0.2182 | 0.9545 |
| **RRF (k=60, Standard Baseline)** | 60 | [1.0, 1.0] | 0.9545 | 0.2182 | 0.9545 |
| **RRF (k=100)** | 100 | [1.0, 1.0] | 0.9545 | 0.2182 | 0.9545 |

---

## 3. Weighted RRF Performance

| Configuration | k | Weights (BM25:Dense) | Recall@5 | Precision@5 | MRR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RRF Balanced** | 60 | [1.0, 1.0] | 0.9545 | 0.2182 | 0.9545 |
| **RRF BM25-Biased (2:1)** | 60 | [2.0, 1.0] | 0.9545 | 0.2182 | 0.9545 |
| **RRF BM25-Heavy (3:1)** | 60 | [3.0, 1.0] | 0.9545 | 0.2182 | 0.9545 |
| **RRF Dense-Biased (1:2)** | 60 | [1.0, 2.0] | 1.0000 | 0.2273 | 0.9409 |
| **RRF Dense-Heavy (1:3)** | 60 | [1.0, 3.0] | 1.0000 | 0.2273 | 0.9182 |

---

## 4. Query-Level Complementarity Breakdown

| # | Query | Expected Chunks | BM25 (R/P/MRR) | Dense (R/P/MRR) | RRF k=60 (R/P/MRR) | Synergy / Outcome |
| :-: | :--- | :--- | :-: | :-: | :-: | :--- |
| 1 | What is FastAPI? | `big_fastapi.md:1,fastapi.md:1` | 1.00/0.40/1.00 | 1.00/0.40/1.00 | 1.00/0.40/1.00 | Robust Rank |
| 2 | How do I install FastAPI? | `big_fastapi.md:2` | 1.00/0.20/1.00 | 1.00/0.20/1.00 | 1.00/0.20/1.00 | Robust Rank |
| 3 | What defines API endpoints in FastAPI? | `big_fastapi.md:3` | 1.00/0.20/1.00 | 1.00/0.20/0.50 | 1.00/0.20/1.00 | Robust Rank |
| 4 | Does FastAPI support dependency injection? | `big_fastapi.md:4` | 1.00/0.20/1.00 | 1.00/0.20/1.00 | 1.00/0.20/1.00 | Robust Rank |
| 5 | What are background tasks in FastAPI? | `big_fastapi.md:5` | 1.00/0.20/1.00 | 1.00/0.20/1.00 | 1.00/0.20/1.00 | Robust Rank |
| 6 | How can FastAPI be deployed? | `big_fastapi.md:6` | 0.00/0.00/0.00 | 1.00/0.20/0.20 | 0.00/0.00/0.00 | Degradation |
| 7 | What is Docker? | `docker.md:1` | 1.00/0.20/1.00 | 1.00/0.20/1.00 | 1.00/0.20/1.00 | Robust Rank |
| 8 | What are the key features of Docker? | `docker.md:1` | 1.00/0.20/1.00 | 1.00/0.20/1.00 | 1.00/0.20/1.00 | Robust Rank |
| 9 | What is Python? | `python.md:1` | 1.00/0.20/1.00 | 1.00/0.20/1.00 | 1.00/0.20/1.00 | Robust Rank |
| 10 | What are the key features of Python? | `python.md:1` | 1.00/0.20/1.00 | 1.00/0.20/1.00 | 1.00/0.20/1.00 | Robust Rank |
| 11 | Does Python support multiple programming paradigms? | `python.md:1` | 1.00/0.20/1.00 | 1.00/0.20/1.00 | 1.00/0.20/1.00 | Robust Rank |
| 12 | Does FastAPI support async and await? | `fastapi.md:1` | 1.00/0.20/1.00 | 1.00/0.20/1.00 | 1.00/0.20/1.00 | Robust Rank |
| 13 | What frameworks and libraries is FastAPI built on? | `fastapi.md:1` | 1.00/0.20/1.00 | 1.00/0.20/1.00 | 1.00/0.20/1.00 | Robust Rank |
| 14 | What are the key features of FastAPI? | `fastapi.md:1` | 1.00/0.20/0.50 | 1.00/0.20/1.00 | 1.00/0.20/1.00 | Robust Rank |
| 15 | Pydantic and Starlette | `fastapi.md:1` | 1.00/0.20/1.00 | 1.00/0.20/1.00 | 1.00/0.20/1.00 | Robust Rank |
| 16 | Uvicorn deployment | `big_fastapi.md:6` | 1.00/0.20/1.00 | 1.00/0.20/1.00 | 1.00/0.20/1.00 | Robust Rank |
| 17 | Kubernetes container orchestration | `docker.md:1` | 1.00/0.20/1.00 | 1.00/0.20/1.00 | 1.00/0.20/1.00 | Robust Rank |
| 18 | async await syntax | `fastapi.md:1` | 1.00/0.20/1.00 | 1.00/0.20/0.33 | 1.00/0.20/1.00 | Robust Rank |
| 19 | modern high performance web framework | `fastapi.md:1,big_fastapi.md:1` | 1.00/0.40/1.00 | 1.00/0.40/1.00 | 1.00/0.40/1.00 | Robust Rank |
| 20 | isolated containerized runtime environment | `docker.md:1` | 1.00/0.20/1.00 | 1.00/0.20/1.00 | 1.00/0.20/1.00 | Robust Rank |
| 21 | object oriented procedural and functional scripting language | `python.md:1` | 1.00/0.20/1.00 | 1.00/0.20/1.00 | 1.00/0.20/1.00 | Robust Rank |
| 22 | asynchronous background execution | `big_fastapi.md:5,fastapi.md:1` | 1.00/0.40/1.00 | 1.00/0.40/1.00 | 1.00/0.40/1.00 | Robust Rank |

---

## 5. Conclusions & Next Steps
- **Hypothesis H005 Confirmed**: RRF maintains high parameter stability across $k \in [20, 60]$ and successfully fuses dense semantic signals with lexical keyword signals.
- **Sprint 2 Roadmap**: Proceed to **RLB-211 (HybridRetriever implementation)** and **RLB-212 (Query-Level Failure Taxonomy)**.
