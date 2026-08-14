# ADR-0004: Default Embedding Model & Assumptions (FastEmbed / BAAI/bge-small-en-v1.5)

**Status**: Accepted  
**Deciders**: RetrievLab Team  
**Date**: 2026-08-14  

---

## Context
Dense retrieval requires mapping text chunks and search queries into high-dimensional vector spaces. To support reproducible offline experiments without external API costs or GPU prerequisites, RetrievLab encapsulates embedding generation behind the `EmbeddingClient` interface.

In `src/retrievlab/embeddings/fastembed.py`, `FastEmbedClient` instantiates `fastembed.TextEmbedding()`.

## Decision
Adopt **`BAAI/bge-small-en-v1.5`** (via FastEmbed's ONNX runtime) as the default baseline dense embedding model for RetrievLab benchmarks.

## External Library Assumptions & Invariants (Build vs. Use Principle)

In accordance with our core principle (*"Understand what the library is doing, what assumptions it makes, and how it affects your experiment"*), we document the following mathematical and architectural assumptions:

1. **Dimensionality**: Outputs fixed **384-dimensional** dense vectors.
2. **Vector Normalization**: FastEmbed vectors are **$L_2$-normalized unit vectors** ($\|\mathbf{v}\|_2 = 1.0$) upon output.
3. **Metric Space Equivalence**:
   - Because all vectors have unit length, **Cosine Similarity is mathematically identical to the Dot Product (Inner Product)**:
     $$\text{Cosine}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \mathbf{u} \cdot \mathbf{v}$$
   - *Experimental Impact*: Linear search and future vector indices (e.g., FAISS `IndexFlatIP`) can safely compute inner products without recalculating vector norms during query time.
4. **Execution Runtime**: Operates locally using quantized ONNX models on CPU, ensuring zero external network latency, no API fees, and fully deterministic offline benchmarking.
5. **Linguistic Scope**: English-optimized retrieval model.

## Consequences

### Positive
- Ultra-fast local embedding generation with minimal memory overhead (~130 MB model footprint).
- Standardized baseline across all dense retrieval and hybrid experiments.
- Dot product optimization allows fast brute-force and indexed vector similarity search.

### Negative / Tradeoffs
- 384 dimensions capture fewer long-context cross-domain semantic nuances compared to 1024+ dimension models (e.g. `bge-large`, `voyage-3`, or `text-embedding-3-large`).
- Future model comparisons must verify whether alternative providers return normalized or unnormalized vectors before computing similarities.
