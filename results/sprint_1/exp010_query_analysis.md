# Experiment Report: exp010 — Query Breakdown & Noteworthy Case Analysis

**Experiment ID:** `exp010`  
**Focus:** Lexical vs. Semantic Query Study & Divergence Analysis  
**Date:** 2026-08-04  
**Status:** Completed  

---

## 1. Objective

Analyze performance differences between `BM25Retriever` and `DenseRetriever` across query categories (Lexical vs. Semantic) and extract "Noteworthy Queries" where retrieval models diverge or fail.

---

## 2. Query Sub-Category Benchmarks (`simple2.json`)

### 2.1 Lexical Queries Study (Exact Keyword Matches)
Queries: `"Pydantic and Starlette"`, `"Uvicorn deployment"`, `"Kubernetes container orchestration"`, `"async await syntax"`, `"What is FastAPI?"`, `"Does FastAPI support dependency injection?"`

| Retriever | Recall@5 | MRR | Precision@5 |
| :--- | :---: | :---: | :---: |
| **BM25 (Lexical)** | **1.0000** | **1.0000** | 0.2333 |
| **Dense (Lexical)** | **1.0000** | 0.8889 | 0.2333 |

**Key Takeaway:** BM25 achieves a perfect MRR of $1.0000$ on exact keyword queries. Dense retrieval slightly misranks exact code/library combinations like `"Pydantic and Starlette"`.

---

### 2.2 Semantic Queries Study (Abstract Concepts & Synonyms)
Queries: `"modern high performance web framework"`, `"isolated containerized runtime environment"`, `"object oriented procedural and functional scripting language"`, `"asynchronous background execution"`

| Retriever | Recall@5 | MRR | Precision@5 |
| :--- | :---: | :---: | :---: |
| **BM25 (Semantic)** | 1.0000 | 1.0000 | 0.3000 |
| **Dense (Semantic)** | 1.0000 | 1.0000 | 0.3000 |

---

## 3. Detailed Query Comparison Table (`simple2.json`)

| Query | BM25 Reciprocal Rank | Dense Reciprocal Rank | Delta ($|\Delta|$) | Winner | Notes / Category |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `What is FastAPI?` | 1.00 | 1.00 | 0.00 | **Tie** | Lexical |
| `How do I install FastAPI?` | 1.00 | 1.00 | 0.00 | **Tie** | Lexical |
| `What defines API endpoints in FastAPI?` | 1.00 | 0.50 | 0.50 | **BM25** | Lexical — Dense placed overview chunk 1st |
| `Does FastAPI support dependency injection?` | 1.00 | 1.00 | 0.00 | **Tie** | Lexical |
| `What are background tasks in FastAPI?` | 1.00 | 1.00 | 0.00 | **Tie** | Lexical |
| `How can FastAPI be deployed?` | 0.00 | 0.20 | 0.20 | **Dense** | BM25 Zero-Hit (tokenizer missing stemming) |
| `What is Docker?` | 1.00 | 1.00 | 0.00 | **Tie** | Lexical |
| `What are the key features of Docker?` | 1.00 | 1.00 | 0.00 | **Tie** | Lexical |
| `What is Python?` | 1.00 | 1.00 | 0.00 | **Tie** | Lexical |
| `What are the key features of Python?` | 1.00 | 1.00 | 0.00 | **Tie** | Lexical |
| `Does Python support multiple programming paradigms?` | 1.00 | 1.00 | 0.00 | **Tie** | Lexical |
| `Does FastAPI support async and await?` | 1.00 | 1.00 | 0.00 | **Tie** | Lexical |
| `What frameworks and libraries is FastAPI built on?` | 1.00 | 1.00 | 0.00 | **Tie** | Lexical |
| `What are the key features of FastAPI?` | 0.50 | 1.00 | 0.50 | **Dense** | Dense placed specific overview chunk 1st |
| `Pydantic and Starlette` | 1.00 | 0.33 | 0.67 | **BM25** | BM25 exact term match |
| `Uvicorn deployment` | 1.00 | 1.00 | 0.00 | **Tie** | Lexical |
| `Kubernetes container orchestration` | 1.00 | 1.00 | 0.00 | **Tie** | Lexical |
| `async await syntax` | 1.00 | 1.00 | 0.00 | **Tie** | Lexical |
| `modern high performance web framework` | 1.00 | 1.00 | 0.00 | **Tie** | Semantic |
| `isolated containerized runtime environment` | 1.00 | 1.00 | 0.00 | **Tie** | Semantic |
| `object oriented procedural and functional scripting language` | 1.00 | 1.00 | 0.00 | **Tie** | Semantic |
| `asynchronous background execution` | 1.00 | 1.00 | 0.00 | **Tie** | Semantic |

---

## 4. Key Takeaways for Model Selection

1. **BM25 Weakness:** BM25 fails completely ($RR=0.00$) when words in the query use a different grammatical form than words in the document (e.g. `deployed` vs `deployment`).
2. **Dense Weakness:** Dense embeddings sometimes over-generalize on specific library names (`Pydantic and Starlette`), giving $RR=0.33$ while BM25 achieves $1.00$.
3. **Hybrid Motivation:** These complementary failure modes strongly justify **Hybrid Retrieval (RRF)** for Sprint 2.
