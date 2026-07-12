"""
A class for embedding chunks of text using a specified embedding model.

"""

from retrievlab.models import Chunk
from retrievlab.embeddings.client import EmbeddingClient

class Embedder:
    """
    A class for embedding chunks of text using a specified embedding model.

    Attributes:
        model_name (str): The name of the embedding model to use.
    """
    def __init__(self, client: EmbeddingClient):
        """
        Initialize the Embedder with a specified embedding model.
        """
        self.embedding_client = client

    def embed(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Generate embeddings for a list of text chunks.

        Args:
            chunks (list[Chunk]): The chunks of text to embed.

        Returns:
            list[Chunk]: The chunks with embedded vectors.
        """
        # Implementation for generating embedding (e.g., using the initialized model)
        embeddings = self.embedding_client.get_embeddings([chunk.text for chunk in chunks])
        for chunk, embedding_vector in zip(chunks, embeddings):
            chunk.embedding = embedding_vector
        return chunks
    