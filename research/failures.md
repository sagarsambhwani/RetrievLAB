# 🛑 Retrieval Failure Log & Taxonomy

This log systematically records failure modes observed during retrieval benchmarks to guide feature engineering and adaptive ranking policies.

---

## Retrieval Failure Taxonomy

| Failure Mode | Category | Root Cause | Example Scenario | Primary Mitigation |
| :--- | :--- | :--- | :--- | :--- |
| **Vocabulary Mismatch** | Lexical (BM25) | Synonyms or conceptual rephrasings with zero lexical overlap | Query: *"containerized environment"*, Passage: *"Docker isolation"* | Dense embeddings / Hybrid retrieval |
| **Semantic Drift** | Dense Vector | Embedding model maps generic semantic similarity but misses specific technical constraints | Query: *"async await syntax in Python 3.5"*, Passage: *"general history of Python"* | BM25 lexical term matching / Cross-encoder reranker |
| **Over-Stemming Collision** | Preprocessing | Stemmer aggressively reduces unrelated words to identical root | Query: *"university admissions"*, Passage: *"universe expansion"* ($\rightarrow$ `univers`) | Moderate/light stemming (Snowball/WordNet lemma) |
| **Under-Stemming Miss** | Preprocessing | Inflected term fails exact match | Query: *"retrieved"*, Passage: *"information retrieval"* | Configurable stemming tokenizer |
| **Ranking Inversion** | Hybrid / Ranker | Relevant chunk retrieved in top-K but ranked below lower-quality matches | High BM25 score on boilerplate tokens overrides dense match | Reciprocal Rank Fusion / Learned LTR weights |

---

## Empirical Benchmark Failure Cases

### Case 001 — BM25 Vocabulary Mismatch on Semantic Paraphrase
- **Benchmark**: `simple2.json` (Query: *"isolated containerized runtime environment"*)
- **Retriever**: `BM25Retriever` (Baseline Basic Tokenizer)
- **Observed Behavior**: BM25 scored target chunk `docker.md:1` with 0.0 because the passage text used terms *"container"*, *"packaged"*, *"isolated"*, but not the exact phrase combinations.
- **Dense Result**: `DenseRetriever` scored `docker.md:1` at Rank 1 (Score: 0.7329).
- **Status**: Documented for Hybrid RRF evaluation in Sprint 2.

---

### Case 002 — Lexical Inflection Miss without Stemming
- **Benchmark**: Lexical variation query *"queries for indexed documents"*
- **Retriever**: `BM25Retriever` (Unstemmed)
- **Observed Behavior**: Missed passage containing *"querying an index"* due to morphological suffix variance (`queries` $\neq$ `querying`, `indexed` $\neq$ `index`).
- **Status**: Mitigated by `StemmedTokenizer` in Sprint 2.

---

### Case 003 — BM25 Vocabulary Miss on Deployment Query Recovered by Hybrid
- **Benchmark**: `simple2.json` (Query 6: *"How can FastAPI be deployed?"*)
- **Target Chunk**: `big_fastapi.md:6` (Passage covers containerization, production servers, and cloud scaling)
- **BM25 Result**: Recall@5 = 0.00 (missed completely due to vocabulary mismatch on deployment terminology).
- **Dense Result**: Recall@5 = 1.00, Rank = 5, MRR = 0.20.
- **Hybrid Result (`HybridRetriever`, RRF 1:2)**: **Recovered into Top-5!**
- **Outcome Category**: `DENSE_WIN_HYBRID_RECOVERED`
- **Mitigation / Research Finding**: Validates hypothesis H005; multi-channel rank fusion successfully recovers single-channel misses with 0 degradations across the dataset.

