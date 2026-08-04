# Experiment Report: exp006 — BM25 Baseline Retrieval Evaluation

**Experiment ID:** `exp006`  
**Retriever:** `BM25Retriever` (`retrievlab.retrieval.bm25.BM25Retriever`)  
**Date:** 2026-08-04  
**Status:** Completed  

---

## 1. Objective

Evaluate the baseline Okapi BM25 lexical retrieval engine across the chunked corpus and standard benchmark dataset. The goal is to establish baseline lexical retrieval metrics (Recall@5, Precision@5, MRR) and verify edge case handling (unknown vocabulary, empty queries).

---

## 2. Experimental Setup

- **Corpus:** `data/raw/` (4 markdown documents)
- **Chunker:** `MarkdownChunker` producing 9 chunks
- **Average Chunk Length:** 30.78 tokens
- **BM25 Parameters:** $k_1 = 1.5$, $b = 0.75$
- **Benchmarks Evaluated:**
  - `data/benchmarks/simple.json` (14 query cases)
  - `data/benchmarks/simple2.json` (22 query cases)

---

## 3. Token & Scoring Inspection

Term frequency ($df$) and Inverse Document Frequency ($IDF$) values computed across the corpus:

| Token | Document Frequency ($df$) | IDF Score |
| :--- | :---: | :---: |
| `fastapi` | 2 | 1.005 |
| `python` | 1 | 1.479 |
| `docker` | 1 | 1.479 |
| `framework` | 2 | 1.005 |

---

## 4. Benchmark Results

### 4.1 Aggregate Metrics

| Benchmark Dataset | Query Count | Recall@5 | Precision@5 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| `simple.json` | 14 | 0.9286 | 0.2000 | 0.8929 |
| `simple2.json` | 22 | 0.9545 | 0.2182 | 0.9318 |

### 4.2 Query-by-Query Breakdown (`simple.json`)

| Query | Expected | Top-1 Retrieved (Score) | Recall@5 | MRR |
| :--- | :--- | :--- | :---: | :---: |
| What is FastAPI? | `big_fastapi.md:1`, `fastapi.md:1` | `big_fastapi.md:1` (2.83) | 1.00 | 1.00 |
| How do I install FastAPI? | `big_fastapi.md:2` | `big_fastapi.md:2` (4.83) | 1.00 | 1.00 |
| What defines API endpoints in FastAPI? | `big_fastapi.md:3` | `big_fastapi.md:3` (5.94) | 1.00 | 1.00 |
| Does FastAPI support dependency injection? | `big_fastapi.md:4` | `big_fastapi.md:4` (8.41) | 1.00 | 1.00 |
| What are background tasks in FastAPI? | `big_fastapi.md:5` | `big_fastapi.md:5` (8.35) | 1.00 | 1.00 |
| How can FastAPI be deployed? | `big_fastapi.md:6` | `big_fastapi.md:1` (1.56) | 0.00 | 0.00 |
| What is Docker? | `docker.md:1` | `docker.md:1` (3.43) | 1.00 | 1.00 |
| What are the key features of Docker? | `docker.md:1` | `docker.md:1` (6.58) | 1.00 | 1.00 |
| What is Python? | `python.md:1` | `python.md:1` (2.84) | 1.00 | 1.00 |
| What are the key features of Python? | `python.md:1` | `python.md:1` (4.40) | 1.00 | 1.00 |
| Does Python support multiple programming paradigms? | `python.md:1` | `python.md:1` (7.33) | 1.00 | 1.00 |
| Does FastAPI support async and await? | `fastapi.md:1` | `fastapi.md:1` (5.65) | 1.00 | 1.00 |
| What frameworks and libraries is FastAPI built on? | `fastapi.md:1` | `fastapi.md:1` (7.93) | 1.00 | 1.00 |
| What are the key features of FastAPI? | `fastapi.md:1` | `docker.md:1` (3.70) | 1.00 | 0.50 |

---

## 5. Edge Case Verification

1. **Unknown Query Terms (`'supercalifragilisticexpialidocious query'`):**
   - Results count: 3
   - Top scores: `0.0000` (Handled gracefully without errors).
2. **Empty Query (`''`):**
   - Results count: 3
   - Top scores: `0.0000` (Handled gracefully without errors).

---

## 6. Key Findings & Observations

- **Exact Match Strengths:** BM25 achieves perfect MRR (1.00) on queries containing exact technical keywords present in the document.
- **Tokenizer Limitations:** Because the baseline tokenizer performs basic whitespace splitting and lowercasing without stemming (e.g. `deployed` vs `deployment`), queries with morphological variations fail (e.g., `How can FastAPI be deployed?` yields 0.00 MRR).
