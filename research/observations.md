# 🔬 Research Observations & Empirical Insights

This document captures high-level empirical observations and research insights discovered across experimentation sprints.

---

## Sprint 1 Observations: Dense vs. BM25 Baselines

### 1. Complementary Strengths of Sparse vs. Dense
- On the `simple2.json` benchmark (8 queries across 4 documents / 13 chunks):
  - **Dense Retriever (`bge-small-en-v1.5`)**: Achieved **100% Recall@5** and **MRR 0.9375**. It excels at understanding semantic intent (e.g., *"modern high performance web framework"* successfully ranks FastAPI top-1).
  - **BM25 Retriever**: Achieved **75% Recall@5** and **MRR 0.7500**. It performed with 100% precision on exact keyword and technical identifier queries (e.g., *"Kubernetes container orchestration"* and *"Pydantic and Starlette"*), but failed when queries were purely conceptual paraphrases.
  - **Conclusion**: A naive single-retriever strategy leaves significant recall on the table. Hybrid candidate generation (BM25 + Dense) is mathematically positioned to achieve strict superiority over either individual baseline.

### 2. Lexical Normalization is Essential for Sparse Retrieval
- Without stemming or lemmatization, BM25 treats different morphological forms of the same root word as disjoint vocabulary tokens.
- Injecting a configurable tokenizer pipeline allows exploring the trade-off between Recall gains from stemming and Precision penalties from over-stemming.

### 3. Metric Alignment
- In retrieval experiments, `Recall@K` alone does not tell the full story. A method can have 100% Recall@5 while ranking the true answer at position 5 (hurting downstream LLM generation quality).
- Combining `Recall@K` with **Mean Reciprocal Rank (MRR)** and **Precision@K** is required to assess both retrieval coverage and ranking confidence.
