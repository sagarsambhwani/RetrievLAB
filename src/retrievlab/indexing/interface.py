"""
Abstract interface for vector indexing backends.
"""

from abc import ABC, abstractmethod
from retrievlab.models import Chunk


class VectorIndex(ABC):
    """Abstract base class for vector search indexes.
    
    A VectorIndex maintains an internal vector representation of pre-embedded
    chunks and provides k-nearest-neighbor search.
    """

    @abstractmethod
    def build(self, chunks: list[Chunk]) -> None:
        """Build the vector index from a list of pre-embedded chunks.
        
        Clears any existing indexed vectors and initializes a fresh index.
        
        Args:
            chunks: List of Chunk objects with embedding vectors populated.
            
        Raises:
            ValueError: If any chunk is missing an embedding or chunks list is empty.
        """
        pass

    @abstractmethod
    def add(self, chunks: list[Chunk]) -> None:
        """Incrementally add chunks with embeddings to the existing index.
        
        Args:
            chunks: List of Chunk objects with embedding vectors populated.
            
        Raises:
            ValueError: If any chunk is missing an embedding or dimension mismatch occurs.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """Search the index for the top-k nearest neighbors to the query vector.
        
        Args:
            query_vector: Dense embedding vector for the query.
            top_k: Maximum number of nearest neighbors to return.
            
        Returns:
            List of (Chunk, similarity_score) tuples sorted in descending order of score.
            
        Raises:
            ValueError: If index is empty, query_vector dimension mismatches, or top_k <= 0.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Reset and empty the vector index and associated chunk metadata."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Return the number of vectors currently indexed."""
        pass
