"""Hybrid retrieval orchestrator combining multiple retrieval backends."""
from __future__ import annotations

from collections.abc import Sequence

from retrievlab.models import Chunk, SearchResult
from retrievlab.ranking.fusion import ReciprocalRankFusion
from retrievlab.retrieval.interface import Retriever


class HybridRetriever(Retriever):
    """Hybrid retriever that orchestrates multiple retrieval backends and fuses results.

    Adheres to RetrievLab Design Principles (docs/design_principles.md Rule 1 & Rule 2)
    by accepting arbitrary sub-retrievers (e.g., BM25Retriever, DenseRetriever) and
    delegating rank fusion to a configurable ranking strategy (defaulting to ReciprocalRankFusion).
    """

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        fusion_strategy: ReciprocalRankFusion | None = None,
        weights: list[float] | None = None,
        candidate_k: int | None = None,
    ) -> None:
        """Initialize HybridRetriever.

        Args:
            retrievers: Non-empty sequence of Retriever implementations to query.
            fusion_strategy: Fusion strategy for rank aggregation. Defaults to
                             ReciprocalRankFusion(k=60).
            weights: Optional list of weights for each retriever. If specified,
                     length must match the number of retrievers.
            candidate_k: Number of candidate results to fetch from each retriever
                         before fusion. If None, dynamically set to max(top_k, 20).

        Raises:
            ValueError: If retrievers is empty, or if weights length does not match
                        retrievers length, or if candidate_k is <= 0.
        """
        if not retrievers:
            raise ValueError("retrievers sequence must not be empty.")

        if weights is not None and len(weights) != len(retrievers):
            raise ValueError(
                f"Number of weights ({len(weights)}) does not match "
                f"number of retrievers ({len(retrievers)})."
            )

        if candidate_k is not None and candidate_k <= 0:
            raise ValueError(f"candidate_k must be a positive integer (> 0), got {candidate_k}")

        self.retrievers = list(retrievers)
        self.fusion_strategy = fusion_strategy or ReciprocalRankFusion(k=60)
        self.weights = weights
        self.candidate_k = candidate_k

    def retrieve(
        self,
        query: str,
        top_k: int,
        chunks: list[Chunk],
    ) -> list[SearchResult]:
        """Retrieve the top-k relevant chunks by fusing results from all sub-retrievers.

        Args:
            query: Search query string.
            top_k: Number of final search results to return (must be > 0).
            chunks: Candidate pool of chunks to rank.

        Returns:
            Ranked list of SearchResult objects sorted by descending fused score.

        Raises:
            ValueError: If top_k <= 0.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer (> 0), got {top_k}")

        if not chunks:
            return []

        cand_k = self.candidate_k if self.candidate_k is not None else max(top_k, 20)

        # Execute each sub-retriever
        ranked_lists: list[list[SearchResult]] = [
            retriever.retrieve(query=query, top_k=cand_k, chunks=chunks)
            for retriever in self.retrievers
        ]

        # Fuse the ranked lists
        return self.fusion_strategy.fuse(
            rankings=ranked_lists,
            weights=self.weights,
            top_k=top_k,
        )
