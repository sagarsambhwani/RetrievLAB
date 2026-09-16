"""
Multi-signal feature extractor for candidates in a CandidatePool.

Extracts lexical, dense, fusion, length, and cross-modality divergence signals
for (query, candidate) pairs, outputting deterministic feature dictionaries
or 2D numpy feature matrices for downstream Learning-to-Rank (LTR).
"""

import re
import numpy as np
from retrievlab.selection.candidate import Candidate, CandidatePool


class FeatureExtractor:
    """Extracts a fixed suite of 16 multi-signal features for candidate chunks.

    Design Principle: FeatureExtractor computes objective signals and metadata.
    It never makes ranking decisions or applies thresholds.
    """

    FEATURE_NAMES: list[str] = [
        "bm25_score",
        "dense_score",
        "bm25_rank",
        "dense_rank",
        "bm25_reciprocal_rank",
        "dense_reciprocal_rank",
        "rrf_score",
        "retrieved_by_both",
        "exact_query_match",
        "token_overlap_ratio",
        "jaccard_similarity",
        "query_token_count",
        "doc_token_count",
        "length_ratio",
        "rank_discrepancy",
        "modality_preference",
    ]

    def __init__(
        self,
        missing_rank: int = 1000,
        rrf_k: int = 60,
    ) -> None:
        """Initialize FeatureExtractor.

        Args:
            missing_rank: Default 1-indexed rank assigned when a candidate was
                          not retrieved by a specific modality. Defaults to 1000.
            rrf_k: Smoothing constant k for RRF calculations. Defaults to 60.
        """
        self.missing_rank = missing_rank
        self.rrf_k = rrf_k
        self._word_regex = re.compile(r"\w+")

    def get_feature_names(self) -> list[str]:
        """Return the stable, ordered list of feature names.

        Returns:
            List of 16 string feature names.
        """
        return list(self.FEATURE_NAMES)

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase alphanumeric words."""
        if not text:
            return []
        return self._word_regex.findall(text.lower())

    def extract_candidate_features(
        self, query: str, candidate: Candidate
    ) -> dict[str, float]:
        """Extract all 16 features for a single (query, candidate) pair.

        Args:
            query: The search query string.
            candidate: A Candidate object containing chunk text and provenance.

        Returns:
            Dictionary mapping feature names to float values.
        """
        # 1. Retrieval scores
        bm25_score = float(candidate.retriever_scores.get("bm25", 0.0))
        dense_score = float(candidate.retriever_scores.get("dense", 0.0))

        # 2. Retrieval ranks
        has_bm25 = "bm25" in candidate.retriever_ranks
        has_dense = "dense" in candidate.retriever_ranks

        bm25_rank = float(candidate.retriever_ranks.get("bm25", self.missing_rank))
        dense_rank = float(candidate.retriever_ranks.get("dense", self.missing_rank))

        # 3. Reciprocal ranks
        bm25_rr = 1.0 / bm25_rank if has_bm25 else 0.0
        dense_rr = 1.0 / dense_rank if has_dense else 0.0

        # 4. RRF Score (sum of 1 / (k + r) for available modalities)
        rrf_score = 0.0
        if has_bm25:
            rrf_score += 1.0 / (self.rrf_k + bm25_rank)
        if has_dense:
            rrf_score += 1.0 / (self.rrf_k + dense_rank)

        # 5. Retrieved by both flag
        retrieved_by_both = 1.0 if (has_bm25 and has_dense) else 0.0

        # 6. Lexical string & token features
        doc_text = candidate.chunk.text or ""
        clean_query = query.strip().lower()
        exact_query_match = (
            1.0 if (clean_query and clean_query in doc_text.lower()) else 0.0
        )

        q_tokens = self._tokenize(query)
        d_tokens = self._tokenize(doc_text)

        q_token_set = set(q_tokens)
        d_token_set = set(d_tokens)

        q_count = len(q_tokens)
        d_count = len(d_tokens)

        if q_token_set:
            overlap = len(q_token_set & d_token_set)
            union = len(q_token_set | d_token_set)
            token_overlap_ratio = overlap / len(q_token_set)
            jaccard_similarity = overlap / union if union > 0 else 0.0
        else:
            token_overlap_ratio = 0.0
            jaccard_similarity = 0.0

        length_ratio = float(d_count) / float(max(q_count, 1))

        # 7. Cross-modality divergence & preference
        rank_discrepancy = abs(bm25_rank - dense_rank)

        sum_rr = bm25_rr + dense_rr
        if sum_rr > 0.0:
            modality_preference = dense_rr / sum_rr
        else:
            modality_preference = 0.5

        return {
            "bm25_score": bm25_score,
            "dense_score": dense_score,
            "bm25_rank": bm25_rank,
            "dense_rank": dense_rank,
            "bm25_reciprocal_rank": bm25_rr,
            "dense_reciprocal_rank": dense_rr,
            "rrf_score": rrf_score,
            "retrieved_by_both": retrieved_by_both,
            "exact_query_match": exact_query_match,
            "token_overlap_ratio": token_overlap_ratio,
            "jaccard_similarity": jaccard_similarity,
            "query_token_count": float(q_count),
            "doc_token_count": float(d_count),
            "length_ratio": length_ratio,
            "rank_discrepancy": rank_discrepancy,
            "modality_preference": modality_preference,
        }

    def extract_pool_features(
        self, pool: CandidatePool
    ) -> list[dict[str, float]]:
        """Extract feature dictionaries for all candidates in a CandidatePool.

        Args:
            pool: The CandidatePool to extract features from.

        Returns:
            List of feature dictionaries, one per candidate in the pool.
        """
        return [
            self.extract_candidate_features(pool.query, cand)
            for cand in pool.candidates
        ]

    def extract_matrix(self, pool: CandidatePool) -> np.ndarray:
        """Extract a 2D float64 numpy feature matrix for a CandidatePool.

        Args:
            pool: The CandidatePool to extract features from.

        Returns:
            2D numpy array of shape (len(pool), len(FEATURE_NAMES)).
        """
        if not pool.candidates:
            return np.empty((0, len(self.FEATURE_NAMES)), dtype=np.float64)

        names = self.FEATURE_NAMES
        rows: list[list[float]] = []
        for cand in pool.candidates:
            feat_dict = self.extract_candidate_features(pool.query, cand)
            rows.append([feat_dict[name] for name in names])

        return np.array(rows, dtype=np.float64)
