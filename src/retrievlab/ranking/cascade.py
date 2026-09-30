"""
Cascaded coarse-to-fine re-ranking architecture.

Chains a high-throughput coarse filter (e.g. LightGBMRanker) with a fine-grained
neural refiner (e.g. CrossEncoderReRanker) to achieve state-of-the-art ranking
precision with reduced neural evaluation overhead.
"""

from __future__ import annotations

from typing import Sequence, Union

from retrievlab.models import Chunk, SearchResult
from retrievlab.ranking.interface import ReRanker
from retrievlab.selection.candidate import Candidate, CandidatePool


class CascadedReRanker(ReRanker):
    """Two-stage cascaded re-ranker combining coarse filtering with neural refinement.

    Flow:
    1. Filter Ranker: Evaluates all candidates in the CandidatePool (size K_cand)
       and prunes them to the top `intermediate_k` most promising items.
    2. Refiner Ranker: Evaluates only those `intermediate_k` candidates with expensive
       neural cross-attention, returning the final `top_k` ranked items.
    """

    def __init__(
        self,
        filter_ranker: ReRanker,
        refiner_ranker: ReRanker,
        intermediate_k: int = 15,
    ) -> None:
        """Initialize the CascadedReRanker.

        Args:
            filter_ranker: Fast coarse ranker (e.g., LightGBMRanker).
            refiner_ranker: High-precision fine ranker (e.g., CrossEncoderReRanker).
            intermediate_k: Number of candidates to retain from the filter stage
                            before passing to the refiner. Must be > 0.
        """
        if intermediate_k <= 0:
            raise ValueError(f"intermediate_k must be greater than 0, got {intermediate_k}")

        self.filter_ranker = filter_ranker
        self.refiner_ranker = refiner_ranker
        self.intermediate_k = intermediate_k

    def rerank(
        self,
        query: str,
        candidates: Union[CandidatePool, Sequence[Union[Chunk, Candidate]]],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Execute coarse-to-fine cascaded re-ranking.

        Args:
            query: The search query string.
            candidates: Candidate pool or list of candidate items.
            top_k: Maximum number of final ranked results to return.

        Returns:
            List of SearchResult objects sorted descending by refiner score.
        """
        # Step 1: Coarse filter stage (e.g., LightGBM)
        filtered_results = self.filter_ranker.rerank(
            query=query,
            candidates=candidates,
            top_k=self.intermediate_k,
        )

        if not filtered_results:
            return []

        # Extract chunks preserved by the filter stage
        filtered_chunks = [res.chunk for res in filtered_results]

        # Step 2: Fine neural refiner stage (e.g., Cross-Encoder)
        return self.refiner_ranker.rerank(
            query=query,
            candidates=filtered_chunks,
            top_k=top_k,
        )
