"""
Interface for retrieval backends.
"""
from abc import ABC, abstractmethod
from retrievlab.models import Chunk, SearchResult

class Retriever(ABC):
    """Interface for retrieval backends."""
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int,
        chunks: list[Chunk],
    ) -> list[SearchResult]:
        """
        Retrieve relevant documents based on a query.

        Args:
            query (str): The query string to search for.
            top_k (int): The number of top results to return.
            chunks (list[Chunk]): Candidate chunks to rank.

        Returns:
            list[SearchResult]: Ranked search results sorted by score.
        """