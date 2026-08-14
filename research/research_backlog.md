# 📋 Research Backlog

This backlog tracks exploratory topics and feature experiments across the retrieval pipeline.

---

## 1. Chunking Strategies
- [x] Heading-aware Markdown chunking (`retrievlab.chunking.markdown`)
- [ ] Fixed-size with sliding window overlap (`retrievlab.chunking.fixed`)
- [ ] Recursive character splitting (`retrievlab.chunking.recursive`)
- [ ] Semantic chunking (embedding similarity distance between sentences)
- [ ] Code-aware syntax chunking (AST / function boundary extraction)

## 2. Preprocessing & Normalization
- [x] Basic word tokenizer
- [x] Regex pattern tokenizer
- [x] Stopword filtering tokenizer
- [x] Stemming tokenizer (`py_rust_stemmers` / Snowball)
- [ ] Character $n$-gram tokenizer (3-gram to 5-gram)
- [ ] Subword tokenizer adapter (BPE / WordPiece)
- [ ] Composable pipeline tokenizer (`PipelineTokenizer`)

## 3. Retrieval Paradigms
- [x] Dense vector retrieval (Cosine / Dot product linear search)
- [x] BM25 lexical retrieval engine
- [ ] Reciprocal Rank Fusion (RRF) Hybrid retrieval
- [ ] Convex linear score combination ($w \cdot \text{Dense} + (1-w) \cdot \text{BM25}$)
- [ ] Dense retrieval with FAISS HNSW / IVF index
- [ ] Multi-query expansion & retrieval

## 4. Feature Engineering & Ranking Signals
- [ ] Dense similarity score
- [ ] BM25 score & term coverage ratio
- [ ] Heading relevance & depth hierarchy penalty
- [ ] Chunk position & document freshness
- [ ] Cross-encoder reranking score

## 5. Learning-to-Rank (LTR) & Adaptive Ranking
- [ ] Feature extraction pipeline
- [ ] LambdaMART / LightGBM ranker training harness
- [ ] Query intent routing (Lexical vs Semantic vs Technical query classifier)
- [ ] Adaptive ranker policy