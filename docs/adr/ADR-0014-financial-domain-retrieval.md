# ADR-0014: Financial Domain Retrieval & Cross-Domain Ranking Transfer

**Status**: Accepted  
**Deciders**: RetrievLab Team  
**Date**: 2026-09-25  

---

## Context

Prior to Sprint 3, RetrievLab’s evaluation suite focused exclusively on biomedical literature:
- **SciFact**: Formal academic claim verification (synthetic scientific statements $\to$ peer-reviewed abstracts).
- **NFCorpus**: Layperson medical question answering (dietary topics $\to$ clinical guidelines).

To validate our retrieval and re-ranking architecture on authentic commercial and financial search, we expanded to **BEIR FiQA** (Financial Opinion QA: 57,638 documents, 648 user investment queries).

The architectural questions were:
1. What is the performance gap between lexical (BM25) and dense (FAISS) retrieval on commercial financial questions?
2. Does a tabular GBDT ranker trained on medical/scientific data transfer zero-shot to financial search?
3. How does candidate pool composition impact second-stage neural re-ranking?

---

## Decision

1. **Adopt Dense-First Primary Retrieval for Financial Question Answering**:
   - Because financial queries suffer from acute vocabulary mismatch (colloquial user terms vs. formal statutory/banking terms), FAISS FlatIP dense retrieval (`bge-small-en-v1.5`) must be deployed as the primary single-stage engine (`nDCG@10 = 0.3848`, `Latency = 41.9 ms`).
2. **Prioritize Conversational Source Domains for Tabular GBDT Transfer**:
   - When transferring tabular ranking policies zero-shot across domains without retraining, models trained on conversational QA syntax (NFCorpus) must be preferred over models trained on declarative claim verification (SciFact).
3. **Mandate Provenance Filtering on Multi-Retriever Candidate Pools**:
   - Multi-retriever candidate pools must not be routed directly to standalone neural cross-encoders without provenance gating (`dense_rank`, `rrf_score`) when one component retriever exhibits low precision ($< 0.25$ nDCG).

---

## Empirical Rationale ([`Exp 029`](file:///e:/Downloads/RetrievLab/results/sprint_3/exp029_fiqa_financial_benchmark.md))

Benchmarked on BEIR FiQA ($N=648$ queries, 57,638 documents):
- **Dense vs. Lexical Gap**: FAISS Dense scored `nDCG@10 = 0.3848` vs. BM25 `0.2346` (**+64.0% relative improvement**, $7.5\times$ lower latency).
- **Tabular Transfer Advantage**: Zero-shot LightGBM trained on conversational health queries achieved `0.3679` nDCG@10, outperforming the model trained on formal claims (`0.3465`, +6.2% relative) and edging out the neural Cross-Encoder (`0.3661`) while running in **19.5 ms**.
- **Candidate Pool Ceiling**: A Union $K=50$ pool established a **66.59% recall ceiling**, confirming that over 20 percentage points of relevant documents remain unharvested in single-stage top-10 retrieval.

---

## Consequences

### Positive
- Establishes a verified commercial baseline on 57,000+ financial documents.
- Demonstrates that GPU-accelerated SentenceTransformers and FAISS FlatIP maintain sub-50ms latency at 57k scale.
- Proves tabular rankers trained on conversational queries generalize across completely disjoint domains (Health $\to$ Finance).

### Negative / Trade-offs
- Pure BM25 search over 57k documents exhibits noticeable latency degradation (313 ms on CPU); production deployment of BM25 at scale requires inverted index optimizations or candidate sharding.
