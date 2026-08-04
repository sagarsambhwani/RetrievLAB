# Experiment Report: exp008 — Evaluation Framework & Comparative Study

**Experiment ID:** `exp008`  
**Components:** `EvaluationReport`, `evaluate_retriever`, `load_benchmark`  
**Date:** 2026-08-04  
**Status:** Completed  

---

## 1. Objective

Validate the comparative evaluation framework (`EvaluationReport`) by running automated benchmark evaluations across both baseline retrieval paradigms (`BM25Retriever` vs `DenseRetriever`).

---

## 2. Framework Overview

The `EvaluationReport` class standardizes metric reporting across different retrieval strategies by collecting:
- **Recall@K ($K=1, 3, 5, 10$):** Fraction of ground truth relevant chunks retrieved in top $K$.
- **Precision@K ($K=1, 5, 10$):** Proportion of top $K$ retrieved chunks that are relevant.
- **Mean Reciprocal Rank (MRR):** Multi-query average of reciprocal rank ($\frac{1}{\text{rank of first relevant item}}$).

---

## 3. Comparative Evaluation Summary

### 3.1 Benchmark 1: `data/benchmarks/simple.json` (14 Cases)

| Retriever | Recall@5 | MRR | Precision@5 |
| :--- | :---: | :---: | :---: |
| **BM25** | 0.9286 | 0.8929 | 0.2000 |
| **Dense** | **1.0000** | **0.9071** | **0.2143** |

### 3.2 Benchmark 2: `data/benchmarks/simple2.json` (22 Cases — RLB-030 Suite)

| Retriever | Recall@5 | MRR | Precision@5 |
| :--- | :---: | :---: | :---: |
| **BM25** | 0.9545 | **0.9318** | 0.2182 |
| **Dense** | **1.0000** | 0.9106 | **0.2273** |

---

## 4. Architectural & Metric Findings

1. **Recall Advantage:** `DenseRetriever` maintains perfect $1.0000$ Recall@5 across both benchmark sets, eliminating zero-hit failures caused by missing exact keywords.
2. **Top-1 Precision Advantage:** `BM25Retriever` slightly leads in MRR on exact keyword benchmarks (`0.9318` vs `0.9106`) because exact term matching places the exact section at rank 1 when query terms match chunk text directly.
3. **Precision@5 Context:** Precision values reflect the sparse ground-truth nature of the benchmark (most queries have 1 relevant chunk, leading to maximum theoretical Precision@5 of $0.20$ to $0.25$).
