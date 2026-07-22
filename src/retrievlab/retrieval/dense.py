"""
A module for performing dense retrieval using embeddings.
"""

from retrievlab.retrieval.interface import Retriever
from retrievlab.embeddings.client import EmbeddingClient
from retrievlab.models import Chunk, SearchResult


class DenseRetriever(Retriever):
    """Dense retriever using embeddings for semantic search.
    
    This retriever ranks chunks based on semantic similarity between the query
    embedding and chunk embeddings. It assumes all chunks are pre-embedded.
    """

    def __init__(self, client: EmbeddingClient) -> None:
        """Initialize the dense retriever.
        
        Args:
            client: An EmbeddingClient instance for generating query embeddings.
        """
        self.embedding_model = client

    def retrieve(
        self,
        query: str,
        top_k: int,
        chunks: list[Chunk],
    ) -> list[SearchResult]:
        """Retrieve the top-k most semantically similar chunks for a query.
        
        Args:
            query: User query string.
            top_k: Number of chunks to return.
            chunks: Candidate chunks to rank. All chunks must have embeddings.
        
        Returns:
            Ranked search results sorted by descending similarity score.
        
        Raises:
            ValueError: If any chunk is missing an embedding.
        """
        # Generate embedding for the query
        query_embedding = self.embedding_model.get_embeddings([query])[0]
        
        # Score and rank all chunks
        results = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(
                    f"Chunk '{chunk.id}' has no embedding. "
                    "Generate embeddings before retrieval."
                )
            
            score = self._similarity(query_embedding, chunk.embedding)
            results.append(SearchResult(chunk=chunk, score=score))

        # Sort chunks by similarity score and return the top k
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        search_results = sorted_results[:top_k]
        return search_results

    def _similarity(
        self,
        embedding1: list[float],
        embedding2: list[float],
    ) -> float:
        """Compute cosine similarity between two embeddings.
        
        Uses dot product, which equals cosine similarity when embeddings are
        unit-normalized (as they are from FastEmbed).
        
        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.
        
        Returns:
            Cosine similarity score (dot product of normalized vectors).
        
        Raises:
            ValueError: If embeddings have mismatched dimensions.
        """
        if len(embedding1) != len(embedding2):
            raise ValueError(
                f"Embedding dimension mismatch: {len(embedding1)} vs {len(embedding2)}"
            )
        
        return sum(a * b for a, b in zip(embedding1, embedding2))