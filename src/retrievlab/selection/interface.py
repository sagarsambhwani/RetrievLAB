"""
Abstract interface for candidate generators.
"""

from abc import ABC, abstractmethod
from retrievlab.models import Chunk
from retrievlab.selection.candidate import CandidatePool


class CandidateGenerator(ABC):
    """Abstract base class for first-stage candidate generation."""

    @abstractmethod
    def generate(
        self,
        query: str,
        top_k_per_retriever: int,
        chunks: list[Chunk],
    ) -> CandidatePool:
        """Generate a pool of candidate chunks for a query from underlying retriever(s).

        Args:
            query: The search query string.
            top_k_per_retriever: Number of top results to fetch from each retriever.
            chunks: Candidate corpus of chunks to retrieve from.

        Returns:
            A deduplicated CandidatePool containing Candidate objects with provenance.
        """
        pass
