"""
Abstract interface definition for second-stage re-rankers.
"""

from abc import ABC, abstractmethod
from typing import Sequence, Union

from retrievlab.models import Chunk, SearchResult
from retrievlab.selection.candidate import Candidate, CandidatePool


class ReRanker(ABC):
    """Abstract base class for second-stage re-ranking algorithms.

    Re-rankers accept a query and a set of candidate chunks (or a CandidatePool),
    compute high-precision scores for each candidate, and return an ordered
    list of SearchResults sorted descending by relevance score.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: Union[CandidatePool, Sequence[Union[Chunk, Candidate]]],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Re-rank candidates for a given search query.

        Args:
            query: The search query string.
            candidates: A CandidatePool or sequence of Chunk/Candidate objects.
            top_k: Optional maximum number of top results to return. If None,
                   returns all scored candidates sorted descending.

        Returns:
            List of SearchResult objects sorted descending by score.
        """
        pass
