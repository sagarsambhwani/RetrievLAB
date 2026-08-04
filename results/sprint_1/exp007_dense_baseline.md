# Experiment Report: exp007 — Dense Vector Baseline Retrieval Evaluation

**Experiment ID:** `exp007`  
**Retriever:** `DenseRetriever` (`retrievlab.retrieval.dense.DenseRetriever`)  
**Embedding Model:** `FastEmbedClient` (`BAAI/bge-small-en-v1.5`)  
**Date:** 2026-08-04  
**Status:** Completed  

---

## 1. Objective

Evaluate the baseline Dense Vector retrieval engine using `FastEmbedClient` embeddings across the chunked corpus and standard benchmark dataset. The goal is to measure semantic matching capability (Recall@5, Precision@5, MRR) and compare against lexical BM25 retrieval.

---

## 2. Experimental Setup

- **Corpus:** `data/raw/` (4 markdown documents, 9 chunks)
- **Embedding Provider:** `FastEmbedClient` generating 384-dimensional dense vectors
- **Similarity Metric:** Cosine similarity over normalized dense vectors
- **Benchmarks Evaluated:**
  - `data/benchmarks/simple.json` (14 query cases)
  - `data/benchmarks/simple2.json` (22 query cases)

---

## 3. Benchmark Results

### 3.1 Aggregate Metrics

| Benchmark Dataset | Query Count | Recall@5 | Precision@5 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| `simple.json` | 14 | 1.0000 | 0.2143 | 0.9071 |
| `simple2.json` | 22 | 1.0000 | 0.2273 | 0.9106 |

### 3.2 Query-by-Query Retrieval Details (`simple.json`)

| Query | Expected Chunk IDs | Top-1 Retrieved Chunk ID | Recall@5 | MRR |
| :--- | :--- | :--- | :---: | :---: |
| What is FastAPI? | `big_fastapi.md:1`, `fastapi.md:1` | `big_fastapi.md:1` | 1.00 | 1.00 |
| How do I install FastAPI? | `big_fastapi.md:2` | `big_fastapi.md:2` | 1.00 | 1.00 |
| What defines API endpoints in FastAPI? | `big_fastapi.md:3` | `fastapi.md:1` | 1.00 | 0.50 |
| Does FastAPI support dependency injection? | `big_fastapi.md:4` | `big_fastapi.md:4` | 1.00 | 1.00 |
| What are background tasks in FastAPI? | `big_fastapi.md:5` | `big_fastapi.md:5` | 1.00 | 1.00 |
| How can FastAPI be deployed? | `big_fastapi.md:6` | `big_fastapi.md:4` | 1.00 | 0.20 |
| What is Docker? | `docker.md:1` | `docker.md:1` | 1.00 | 1.00 |
| What are the key features of Docker? | `docker.md:1` | `docker.md:1` | 1.00 | 1.00 |
| What is Python? | `python.md:1` | `python.md:1` | 1.00 | 1.00 |
| What are the key features of Python? | `python.md:1` | `python.md:1` | 1.00 | 1.00 |
| Does Python support multiple programming paradigms? | `python.md:1` | `python.md:1` | 1.00 | 1.00 |
| Does FastAPI support async and await? | `fastapi.md:1` | `fastapi.md:1` | 1.00 | 1.00 |
| What frameworks and libraries is FastAPI built on? | `fastapi.md:1` | `fastapi.md:1` | 1.00 | 1.00 |
| What are the key features of FastAPI? | `fastapi.md:1` | `fastapi.md:1` | 1.00 | 1.00 |

---

## 4. Key Findings & Observations

- **100% Recall@5:** Dense retrieval retrieved at least one relevant document in top-5 for 100% of benchmark queries across both benchmark sets (`Recall@5 = 1.0000`).
- **Semantic Generalization:** Unlike BM25 which returned 0 hits for morphological variations (e.g. `deployed` vs `deployment`), Dense retrieval successfully captured semantic intent (`MRR = 0.20` on deployment query vs `0.00` in BM25).
- **Fine-Grained Distinction Challenge:** On generic or overlapping queries (e.g. `What defines API endpoints in FastAPI?`), vector embeddings rank closely related chunks (`fastapi.md:1` vs `big_fastapi.md:3`) near the top, leading to minor MRR reductions compared to exact keyword matching.
