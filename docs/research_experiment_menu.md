# 🧪 RetrievLab — Master Experiment Menu & Research Roadmap

This document serves as RetrievLab's comprehensive research menu and experimental taxonomy. It organizes all prospective retrieval, ranking, and optimization explorations into 9 research pillars across 5 strategic buckets, leading systematically toward **Adaptive Learning-to-Rank (LTR)**.

> **Guiding Principle:** *Don't execute 84 experiments at once. Use this as your structured experiment menu to explore hypotheses without losing the main research direction.*

```text
Candidate Recall (Stage 1)
      ↓
Different Datasets (Multi-Domain)
      ↓
Different Retrieval Models (Sparse / Dense / Late Interaction)
      ↓
Different Pipeline Architectures (Two-Stage / Cascades)
      ↓
Failure Analysis (Query Diagnostics)
      ↓
Adaptive LTR (Dynamic Routing)
```

---

# 🏛️ The 5 Strategic Buckets

| Bucket | Pillars Covered | Core Focus |
|:---|:---|:---|
| 🟢 **1. Retrieval Quality** | Pillar B (13–23) | Maximizing Stage-1 candidate pool recall before re-ranking. |
| 🔵 **2. Representation** | Pillar A (1–12) | Understanding how corpus ingestion, chunking, and tokenization affect retrieval. |
| 🟣 **3. Ranking** | Pillar D (31–42) | Determining what feature signals and scoring models optimize Stage 2. |
| 🟠 **4. Generalization & Failures** | Pillars F, G, H (53–74) | Multi-domain benchmarking, cross-domain transfer penalties, and error taxonomy. |
| 🔴 **5. Adaptive LTR** | Pillar I (75–84) | The final destination: query difficulty routing, confidence estimation, and dynamic compute allocation. |

---

# 📋 The 84 Experiment Options

## Pillar A: Improve Ingestion & Preprocessing (Representation)
*Research Question: How much does corpus representation itself affect retrieval?*

- [ ] **1. Chunk size sweep:** 128 / 256 / 512 tokens.
- [ ] **2. Chunk overlap:** 0% / 10% / 20% sliding window overlap.
- [ ] **3. Sentence vs. paragraph chunking:** Semantic sentence splitting vs structural paragraph boundaries.
- [x] **4. Heading-aware vs. plain chunks:** Markdown heading hierarchy prepending (`retrievlab.chunking.markdown`).
- [ ] **5. Document-level vs. chunk-level retrieval:** Flat chunking vs document parent-child rollup.
- [ ] **6. Title + body vs. body only:** Prepending metadata titles into document embedding text.
- [ ] **7. Metadata enrichment:** Tagging chunks with inferred entities, dates, or summary clauses.
- [ ] **8. Query normalization:** Punctuation stripping, case folding, and whitespace canonicalization.
- [ ] **9. Document normalization:** Unicode NFKC normalization and HTML tag stripping.
- [x] **10. Stopword / stemming experiments:** Benchmarking word tokenizers, Snowball stemmers, and stopword filters (`Exp 001–004`).
- [ ] **11. Query expansion:** Synonym injection, pseudo-relevance feedback (PRF / RM3), and HyDE.
- [ ] **12. Duplicate / near-duplicate handling:** MinHash / SimHash deduplication of overlapping passages.

---

## Pillar B: Improve Stage-1 Recall (Retrieval Quality)
*Research Question: How do we maximize candidate recall before ranking?*

- [x] **13. Candidate-depth sweep:** Sweeping $K_{\text{cand}} \in [10, 20, 50, 100, 150, 200, 300, 500]$ to quantify recall ceiling vs pool size.
- [ ] **14. BM25 parameter sweep:** Grid searching $k_1 \in [0.8, 2.0]$ and $b \in [0.3, 0.9]$ per domain.
- [ ] **15. Dense model comparison:** Comparing `bge-small-en-v1.5`, `all-MiniLM-L6-v2`, and `e5-small-v2`.
- [ ] **16. Dense embedding dimension comparison:** 384-d vs 768-d vs Matryoshka dimension truncation.
- [x] **17. BM25 + Dense union:** Merging top-$K$ lexical and dense results (`MultiRetrieverCandidateGenerator`).
- [x] **18. Different RRF weights:** Comparing 1:1 vs 1:2 vs 2:1 BM25-to-Dense weighting in fusion (`Exp 021`).
- [x] **19. Different RRF constants:** Sweeping $k_{\text{rrf}} \in [10, 30, 60, 100]$.
- [ ] **20. More than two retrievers:** 3-way fusion combining BM25 + Dense + Sparse/Entity retriever.
- [ ] **21. Learned sparse retrieval:** SPLADE term expansion into inverted index vocabulary.
- [ ] **22. ColBERT / late interaction:** Multi-vector token representation with MaxSim operator.
- [ ] **23. Query expansion + retrieval:** Evaluating candidate recall when expanding colloquial queries into medical/technical terms.

---

## Pillar C: Try Different Embedding & Retrieval Models
*Research Question: Is our current dense retriever actually the limiting factor?*

- [ ] **24. BGE variants:** `bge-small-en-v1.5` vs `bge-base-en-v1.5` vs `bge-large-en-v1.5`.
- [ ] **25. E5 variants:** `e5-small-v2` vs `multilingual-e5`.
- [ ] **26. MiniLM variants:** `all-MiniLM-L6-v2` vs `all-MiniLM-L12-v2`.
- [ ] **27. Domain-specific embeddings:** PubMedBERT / BioLinkBERT for biomedical search.
- [ ] **28. SPLADE:** Neural sparse expansion vs standard BM25.
- [ ] **29. ColBERT:** Fast late-interaction index using PLAID or linear token dot-product.
- [x] **30. Different FAISS indexes:** Exact `IndexFlatIP` vs approximate `IndexHNSWFlat` / `IndexIVFFlat`.

---

## Pillar D: Improve & Explore Re-Ranking (Ranking)
*Research Question: What information actually makes a good ranking decision?*

- [x] **31. Cross-Encoder model comparison:** Evaluating MS MARCO MiniLM cross-encoders (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
- [x] **32. Small vs. large Cross-Encoder:** MiniLM-L6 (22M params) vs BGE-Reranker-Base (110M params) (`Exp 028`).
- [ ] **33. Domain-specific Cross-Encoder:** Bio-Cross-Encoder fine-tuned on scientific claim verification.
- [x] **34. LightGBM hyperparameter sweep:** Tree depth, learning rate, num_leaves ($15, 31, 63$), and min_child_samples.
- [ ] **35. Feature ablation:** Systematically stripping feature groups (Lexical vs Dense vs Fusion vs Metadata).
- [ ] **36. Feature pruning:** Dropping zero-gain features (`retrieved_by_both`) to reduce extraction latency.
- [ ] **37. Rank-only LightGBM:** Training strictly on ordinal rank positions ($1/r_{\text{bm25}}, 1/r_{\text{dense}}, \text{rrf}$) without raw similarity scores.
- [ ] **38. Score-only LightGBM:** Training strictly on uncalibrated raw scores ($\text{bm25\_score}, \text{dense\_score}$).
- [x] **39. Rank + score LightGBM:** Hybrid feature space combining raw scores, reciprocal ranks, and rank divergence (`Exp 025`).
- [ ] **40. Query-difficulty features:** Query token entropy, query length, character length ratio, and punctuation presence.
- [x] **41. Candidate-pool features:** Rank discrepancy ($|r_{\text{lex}} - r_{\text{dense}}|$), modality preference, and candidate pool position.
- [x] **42. Retriever-agreement features:** Boolean co-occurrence and rank concordance across Stage-1 retrievers.

---

## Pillar E: Explore Different Pipeline Architectures
*Research Question: Which pipeline architecture gives the best quality/latency trade-off?*

- [ ] **43. BM25 → reranker:** Pure lexical candidate generation followed by LightGBM / Cross-Encoder.
- [ ] **44. Dense → reranker:** Pure semantic candidate generation followed by LightGBM / Cross-Encoder.
- [x] **45. BM25 + Dense → RRF:** Single-stage heuristic fusion baseline without machine learning (`Exp 021`).
- [x] **46. BM25 + Dense → LightGBM:** Multi-retriever union candidates scored by tabular LambdaMART (`Exp 025`).
- [x] **47. BM25 + Dense → Cross-Encoder:** Multi-retriever union candidates scored by full cross-attention (`Exp 025`).
- [ ] **48. Large candidate pool → LightGBM:** $K=200$ pool filtered down to top 10 using tabular GBDT.
- [ ] **49. Large candidate pool → Cross-Encoder:** $K=200$ pool passed directly to GPU cross-attention.
- [ ] **50. Multi-retriever → LightGBM:** 3-way candidate pool (BM25 + Dense + Sparse) feeding LightGBM.
- [x] **51. LightGBM → Cross-Encoder cascade:** Union $K=50 \xrightarrow{\text{LightGBM}} \text{Top 10..30} \xrightarrow{\text{Cross-Encoder}} \text{Final Top 10}$ (`Exp 027`).
- [ ] **52. Retriever → adaptive reranker:** Routing easy queries to LightGBM and hard queries to Cross-Encoder.

---

## Pillar F: Explore Domains
*Research Question: What generalizes and what is domain-specific?*

- [x] **53. SciFact:** Scientific claim verification (~5k abstracts, formal academic queries).
- [x] **54. NFCorpus:** Nutrition & medical QA (~3.6k abstracts, layperson conversational queries).
- [x] **55. FiQA:** Financial opinion QA (~57k documents, technical financial terminology) (`Exp 029`).
- [ ] **56. SCIDOCS:** Scientific paper citations and co-readership graphs.
- [ ] **57. HotpotQA:** Multi-hop reasoning and multi-document synthesis.
- [ ] **58. Robust04 / TREC:** News articles with long traditional keyword queries.

---

## Pillar G: Cross-Domain Research
*Research Question: What parts of the learned ranking policy transfer?*

- [x] **59. SciFact → NFCorpus:** Evaluating zero-shot transfer of scientific GBDT ranker to layperson medical search (`Exp 026`).
- [x] **60. SciFact → FiQA:** Evaluating transfer from scientific verification to financial QA (`Exp 029`).
- [ ] **61. NFCorpus → SciFact:** Reverse-transferring a layperson-trained model to academic literature.
- [ ] **62. Combined-domain training:** Pooling training sets from SciFact + NFCorpus to train a generalized tabular ranker.
- [x] **63. Zero-shot vs. in-domain LTR:** Quantifying the performance recovery from 1 second of in-domain CPU retraining.
- [x] **64. Feature distribution shift:** Measuring statistical divergence in feature values across corpora.
- [x] **65. Feature importance shift:** Tracking split-gain re-weighting between formal syntax and keyword queries (`Exp 026`).
- [x] **66. Domain-transfer penalty:** Formally computing $\Delta_{\text{transfer}} = \text{In-Domain} - \text{Zero-Shot}$.

---

## Pillar H: Retrieval Failure Analysis
*Research Question: Why does the system fail?*

- [x] **67. BM25-only successes:** Queries where exact lexical match succeeds while dense embeddings suffer semantic drift.
- [x] **68. Dense-only successes:** Queries where semantic paraphrase succeeds while BM25 suffers vocabulary mismatch.
- [x] **69. Both succeed (Joint Hits):** Concordant queries where fusion boosts confidence.
- [x] **70. Both fail (Joint Misses):** Queries where neither baseline captures relevant items in top 5 (27.2% on NFCorpus).
- [x] **71. Candidate missing vs. ranking failure:** Separating Stage-1 truncation (recall ceiling) from Stage-2 ordering errors.
- [ ] **72. Query difficulty analysis:** Quantifying correlation between query characteristics and retrieval failure.
- [x] **73. Query-level winner/loser analysis:** Tracking per-query win/loss deltas across systems (`Exp 025 & Exp 026`).
- [ ] **74. Failure taxonomy across datasets:** Comparing failure mode proportions between SciFact, NFCorpus, and FiQA.

---

## Pillar I: Adaptive LTR — The Eventual Destination
*Research Question: Can retrieval dynamically adapt its compute and ranking policy per query?*

```text
User Query
    ↓
Compute Cheap Signals (Length, Token Overlap, Rank Discrepancy, BM25/Dense Score Spread)
    ↓
Query Difficulty / Uncertainty Classifier
    ↓
┌─────────────────────────────────────────────────┐
│                                                 │
▼                                                 ▼
[Easy Query: High Confidence]       [Hard Query: High Disagreement]
│                                   │
Fast Path (LightGBM ~1ms)           Full Cascade / Cross-Encoder (~40ms)
│                                   │
▼                                   ▼
Top 5 Results                       Top 5 Results
```

- [ ] **75. Query difficulty estimator:** Predicting query difficulty prior to second-stage scoring.
- [ ] **76. Retriever confidence estimator:** Quantifying top-1 score margin and score distribution skewness.
- [ ] **77. BM25/Dense disagreement signal:** Using $|r_{\text{lex}} - r_{\text{dense}}|$ as an online trigger for reranking depth.
- [ ] **78. Candidate-pool quality prediction:** Predicting whether candidate recall ceiling is high enough to justify deep reranking.
- [ ] **79. Adaptive candidate depth:** Dynamically setting $K_{\text{cand}} \in [20, 200]$ based on query difficulty.
- [ ] **80. Adaptive retriever selection:** Invoking only BM25 for keyword queries, and activating Dense only when needed.
- [ ] **81. Adaptive LightGBM vs. Cross-Encoder:** Routing queries between tabular trees and neural cross-attention.
- [ ] **82. Cheap → expensive reranking cascade:** Scoring top 50 with LightGBM, and passing only top 10 to Cross-Encoder.
- [ ] **83. Adaptive compute budget:** Enforcing latency SLA boundaries (e.g. max 50 ms/query) with dynamic early-exit.
- [ ] **84. Learn when additional retrieval is worthwhile:** Reinforcement learning or bandit policy deciding whether to trigger query expansion.
