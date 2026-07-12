"""
Interface for embedding backends.
"""

from abc import ABC, abstractmethod

class EmbeddingClient(ABC):
    """Interface for embedding backends."""
    @abstractmethod
    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of text strings.

        Args:
            texts (list[str]): The list of text strings to embed.

        Returns:
            list[list[float]]: The list of embedding vectors.
        """