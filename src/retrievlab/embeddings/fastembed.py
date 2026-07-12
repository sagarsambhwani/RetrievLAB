"""
A client for generating embeddings using the FastEmbed model.
"""

from retrievlab.embeddings.client import EmbeddingClient
from fastembed import TextEmbedding

class FastEmbedClient(EmbeddingClient):
    """
    A client for generating embeddings using the FastEmbed model.

    This class implements the EmbeddingClient interface and provides a method to generate embeddings
    for a list of text strings using the FastEmbed model.
    """
    def __init__(self):
        """
        Initialize the FastEmbedClient with the FastEmbed model.
        """
        self.model = TextEmbedding()

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of text strings using the FastEmbed model.

        Args:
            texts (list[str]): The list of text strings to embed.

        Returns:
            list[list[float]]: The list of embedding vectors.
        """
        # Implementation for generating embeddings using FastEmbed
        embeddings = list(self.model.embed(texts))
        embeddings = [vector.tolist() for vector in embeddings]
        return embeddings