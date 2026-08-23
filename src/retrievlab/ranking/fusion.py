"""Reciprocal Rank Fusion (RRF) rank aggregation algorithm."""
from __future__ import annotations

from retrievlab.models import Chunk, SearchResult


class ReciprocalRankFusion:
    """Reciprocal Rank Fusion (RRF) rank aggregation.

    RRF is an unsupervised rank aggregation method that combines ranked lists
    from multiple retrieval systems (e.g., BM25 and Dense vector search) using
    reciprocal rank scoring:

        RRF(d) = sum_{m in M} (w_m / (k + r_m(d)))

    where r_m(d) is the 1-based rank position of chunk d in ranking list m,
    k is the smoothing / fusion constant, and w_m is an optional retriever weight.
    """

    def __init__(
        self,
        k: int = 60,
        weights: list[float] | None = None,
    ) -> None:
        """Initialize ReciprocalRankFusion.

        Args:
            k: Smoothing constant added to rank in the denominator. Must be >= 0.
               Default is 60 (standard Cormack et al. 2009 constant).
            weights: Optional list of retriever weights (w_m). If provided,
                     each weight must be non-negative.

        Raises:
            ValueError: If k < 0 or any weight in weights < 0.
        """
        if k < 0:
            raise ValueError(f"k must be non-negative (>= 0), got {k}")
        if weights is not None:
            for idx, w in enumerate(weights):
                if w < 0:
                    raise ValueError(f"Weight at index {idx} must be non-negative (>= 0), got {w}")

        self.k = k
        self.weights = weights

    def fuse(
        self,
        rankings: list[list[SearchResult]],
        weights: list[float] | None = None,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Fuse multiple ranked lists into a single consolidated ranking.

        Args:
            rankings: List of ranked SearchResult lists from different retrievers.
            weights: Optional weights overriding instance weights for this fusion call.
            top_k: Optional maximum number of top results to return. If None,
                   returns all unique fused chunks.

        Returns:
            Ranked list of SearchResult objects sorted by descending RRF score.

        Raises:
            ValueError: If top_k is <= 0 (when specified), or if weights length does
                        not match the number of ranking lists.
        """
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be a positive integer (> 0), got {top_k}")

        if not rankings:
            return []

        active_weights = weights if weights is not None else self.weights

        if active_weights is not None:
            if len(active_weights) != len(rankings):
                raise ValueError(
                    f"Number of weights ({len(active_weights)}) does not match "
                    f"number of ranking lists ({len(rankings)})"
                )
            for idx, w in enumerate(active_weights):
                if w < 0:
                    raise ValueError(f"Weight at index {idx} must be non-negative (>= 0), got {w}")
        else:
            active_weights = [1.0] * len(rankings)

        # Map chunk_id -> (Chunk, cumulative_rrf_score)
        fused_chunks: dict[str, Chunk] = {}
        fused_scores: dict[str, float] = {}

        for ranked_list, weight in zip(rankings, active_weights):
            if weight == 0.0:
                continue

            seen_in_list: set[str] = set()
            for rank_idx, result in enumerate(ranked_list):
                chunk = result.chunk
                cid = chunk.id
                # Handle duplicate chunk in same list: use first (best) occurrence
                if cid in seen_in_list:
                    continue
                seen_in_list.add(cid)

                rank = rank_idx + 1  # 1-based rank
                contribution = weight / (self.k + rank)

                if cid not in fused_chunks:
                    fused_chunks[cid] = chunk
                    fused_scores[cid] = 0.0

                fused_scores[cid] += contribution

        # Sort chunks by descending score with stable deterministic tie-breaking (by chunk id)
        sorted_chunk_ids = sorted(
            fused_scores.keys(),
            key=lambda cid: (-fused_scores[cid], cid),
        )

        if top_k is not None:
            sorted_chunk_ids = sorted_chunk_ids[:top_k]

        return [
            SearchResult(chunk=fused_chunks[cid], score=fused_scores[cid])
            for cid in sorted_chunk_ids
        ]


def reciprocal_rank_fusion(
    rankings: list[list[SearchResult]],
    k: int = 60,
    weights: list[float] | None = None,
    top_k: int | None = None,
) -> list[SearchResult]:
    """Convenience function to perform Reciprocal Rank Fusion on multiple ranked lists.

    Args:
        rankings: List of ranked SearchResult lists from different retrievers.
        k: Smoothing constant added to rank in the denominator. Defaults to 60.
        weights: Optional list of retriever weights (w_m).
        top_k: Optional maximum number of top results to return.

    Returns:
        Ranked list of SearchResult objects sorted by descending RRF score.
    """
    rrf = ReciprocalRankFusion(k=k, weights=weights)
    return rrf.fuse(rankings=rankings, top_k=top_k)
