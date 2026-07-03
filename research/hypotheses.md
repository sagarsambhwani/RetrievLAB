# H001 - Markdown-aware chunking

Status: Baseline

Hypothesis:
Chunking by Markdown headings preserves semantic coherence better than fixed-size chunking.

Baseline:
Fixed-size chunks (500 characters)

Metric:
Recall@K
MRR
nDCG

Status:
Not tested

# H002 - Adaptive Runtime Ranking

Status: Planned

Hypothesis:
Runtime ranking using multiple retrieval features outperforms cosine similarity alone.

Features:
- Dense similarity
- BM25
- Heading relevance
- Chunk position
- Neighbor consistency

Metric:
Recall@K
MRR
Answer correctness
Latency