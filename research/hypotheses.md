# 🔬 Research Hypotheses Registry

This registry tracks formal hypotheses tested within RetrievLab experiments.

---

## Active & Validated Hypotheses

### H001 — Heading-Aware Markdown Chunking
- **Status**: ✅ Validated (Sprint 1 Baseline)
- **Hypothesis**: Chunking documents along Markdown heading boundaries (`#` to `######`) preserves topical coherence better than fixed-character window chunking.
- **Metrics**: `Recall@K`, `MRR`
- **Result**: Confirmed. Preserving heading metadata keeps semantic context intact for both dense embeddings and lexical matching.

---

### H002 — Lexical vs. Dense Orthogonality
- **Status**: ✅ Validated (Sprint 1 Report)
- **Hypothesis**: Dense retrieval and BM25 exhibit orthogonal error patterns: BM25 excels at exact keyword/code token matches, while Dense retrieval excels at semantic paraphrases.
- **Metrics**: `Recall@5`, `Precision@5`, Query Failure Breakdown
- **Result**: Confirmed on `simple2.json` benchmark (Dense Recall@5: 1.0, BM25 Recall@5: 0.75). BM25 fails when queries use synonymous phrasing without lexical overlap.

---

### H003 — Morphological Normalization in BM25 (Stemming vs. Exact Match)
- **Status**: 🟡 Testing in Sprint 2 (RLB-204 / Exp 012)
- **Hypothesis**: Applying moderate rule-based stemming (Snowball/Porter) will improve BM25 `Recall@K` on inflected queries by mapping word variations to common roots, with minimal penalty on `Precision@K`.
- **Baseline**: `BM25(BasicWordTokenizer)`
- **Comparison**: `BM25(StemmedTokenizer)` vs `BM25(LemmatizedTokenizer)` vs `BM25(Lancaster)`
- **Metrics**: `Recall@5`, `Precision@5`, `MRR`, Vocabulary Size

---

### H004 — Stopword Filtering in Small vs Large Corpora
- **Status**: 🟡 Testing in Sprint 2 (RLB-204 / Exp 011)
- **Hypothesis**: Removing high-frequency function words (stopwords) prevents non-informative term matches from polluting BM25 IDF scores and reduces index memory without hurting Recall.
- **Baseline**: `BM25(BasicWordTokenizer)`
- **Metrics**: `Recall@5`, `MRR`, Index Posting List Length

---

### H005 — Reciprocal Rank Fusion (RRF) Parameter Stability
- **Status**: ⚪ Planned for Sprint 2 (RLB-210)
- **Hypothesis**: Fusing Dense and BM25 rankings via Reciprocal Rank Fusion with smoothing parameter $k \in [20, 60]$ achieves higher `MRR` and `Recall@5` than either standalone retriever.
- **Formula**: $\text{RRF}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$
- **Metrics**: `Recall@5`, `Precision@5`, `MRR`

---

### H006 — Adaptive Runtime Ranking over Heuristic Fusion
- **Status**: ⚪ Long-Term Roadmap (Sprint 3 / Phase 8)
- **Hypothesis**: A learned ranker (LambdaMART / LightGBM) trained on query-passage features (BM25 score, dense cosine similarity, heading depth, query coverage) consistently outperforms fixed linear/RRF weights across diverse query intents.
- **Metrics**: `nDCG@10`, `MRR`, `Recall@10`