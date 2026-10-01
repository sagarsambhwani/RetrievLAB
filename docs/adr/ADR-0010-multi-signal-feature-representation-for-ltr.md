# ADR-0010: Multi-Signal Feature Representation for Learning-to-Rank

**Status**: Accepted  
**Deciders**: RetrievLab Team  
**Date**: 2026-09-17  

---

## Context

Following the establishment of candidate pool generation ([ADR-0009](file:///e:/Downloads/RetrievLab/docs/adr/ADR-0009-candidate-pool-generation-architecture.md)), second-stage Learning-to-Rank (LTR) algorithms require numerical feature representations for each (query, candidate) pair. 

In traditional search engines, reliance on a single retrieval score (e.g. pure BM25 or pure dense vector similarity) creates known failure modes:
1. **Uncalibrated Raw Scores**: BM25 scores are unbounded and scale-dependent on query token length and corpus document frequency statistics; cosine similarity scores occupy a narrow distribution ($[0.6, 0.95]$) that does not calibrate linearly across different queries.
2. **Missing Complementary Signals**: Lexical overlap, document length penalties, rank discrepancy between modalities, and heuristic fusion scores (RRF) provide orthogonal signals that can be non-linearly combined to improve ranking accuracy.

To support tabular GBDT models (such as LightGBM LambdaMART), RetrievLab required a standardized, deterministic feature extraction architecture that translates `(Query, Candidate)` pairs into structured tabular matrices with query-level grouping metadata.

---

## Decision

Implement a modular feature extraction subsystem under `src/retrievlab/features/`:
1. Implement **`FeatureExtractor`** in `src/retrievlab/features/extractor.py` computing a fixed suite of **16 deterministic signals** across five orthogonal categories:
   * **Retrieval Scores**: `bm25_score`, `dense_score`
   * **Retrieval Ranks**: `bm25_rank`, `dense_rank`, `bm25_reciprocal_rank`, `dense_reciprocal_rank`
   * **Heuristic Fusion**: `rrf_score`, `retrieved_by_both`
   * **Lexical & Length Signals**: `exact_query_match`, `token_overlap_ratio`, `jaccard_similarity`, `query_token_count`, `doc_token_count`, `length_ratio`
   * **Modality Divergence**: `rank_discrepancy`, `modality_preference`
2. Implement **`LTRDataset`** in `src/retrievlab/features/dataset.py` as a structured dataclass containing feature matrices ($X \in \mathbb{R}^{N \times 16}$), relevance labels ($y \in \mathbb{Z}^N$), and query group sizes (`group_sizes: list[int]`).
3. Implement **`build_ltr_dataset`** to assemble `LTRDataset` instances from candidate pools and ground-truth benchmark cases.

---

## Architectural Principles & Implementation Details

1. **Rank-First Representation**:
   To address the lack of cross-query score calibration, `FeatureExtractor` extracts both raw retrieval scores and ordinal rank positions (`dense_rank`, `bm25_rank`). Ordinal ranks are scale-invariant across queries ($1, 2, 3 \dots$).
2. **Deterministic & Self-Contained**:
   `FeatureExtractor` relies strictly on candidate provenance and lightweight string tokenization, requiring no external neural models or GPU calls during extraction. Extraction for 85 candidates executes in under **25 milliseconds** on a single CPU core.
3. **Query-Grouped Dataset Schema (`group_sizes`)**:
   Pairwise and listwise ranking algorithms (e.g. LambdaMART) require knowledge of query boundaries to compute pairwise document swaps. `LTRDataset` explicitly enforces:
   $$\sum_{q=1}^{Q} \text{group\_size}_q = N_{\text{candidates}}$$
   guaranteeing mathematical safety during tree construction.

---

## Validation & Experimental Findings

1. **Collinearity & Signal Diagnostics ([`exp023_feature_correlation_diagnostics.md`](file:///e:/Downloads/RetrievLab/results/sprint_3/exp023_feature_correlation_diagnostics.md))**:
   Evaluated across 25,663 candidate pairs from 300 BEIR SciFact queries:
   * Spearman rank correlation confirmed that dense signals (`dense_rank`, $r_s = -0.428$; `dense_score`, $r_s = +0.426$) exhibited the strongest individual association with relevance.
   * Hierarchical clustering identified four clean signal families: Dense Proximity, Lexical Overlap, Document Length, and Modality Divergence.
2. **Feature Split Gain in LambdaMART ([`exp025_twostage_reranking_benchmark.md`](file:///e:/Downloads/RetrievLab/results/sprint_3/exp025_twostage_reranking_benchmark.md))**:
   When training LightGBM on 69,079 pairs, tree split gain was heavily dominated by ordinal ranks:
   * `dense_rank`: **57.7% of total gain** (57,514.25)
   * `rrf_score`: **25.3% of total gain** (25,187.16)
   * `doc_token_count`: **4.8% of total gain** (4,791.07)
   Over **83% of total model gain** was derived from rank-based features rather than uncalibrated raw float scores.

---

## Consequences

### Positive
* Standardizes feature extraction into a deterministic, reusable component.
* Provides native compatibility with modern GBDT ranking libraries via query group metadata.
* Ordinal rank features effectively eliminate cross-query calibration drift.

### Tradeoffs / Considerations
* The 16-feature suite is fixed; incorporating dense embeddings directly (e.g. 384-dimensional vector features) would dramatically expand tabular dimensionality and tree training time.
* Token overlap and Jaccard similarity features rely on simple whitespace/alphanumeric regex tokenization, which does not capture morphological stemming or semantic synonymy.
