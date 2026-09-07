"""
FAISS vector indexing and retriever implementation for fast dense retrieval.
"""

from typing import Sequence
import faiss
import numpy as np

from retrievlab.indexing.interface import VectorIndex
from retrievlab.retrieval.interface import Retriever
from retrievlab.embeddings.client import EmbeddingClient
from retrievlab.models import Chunk, SearchResult


class FAISSIndex(VectorIndex):
    """FAISS-backed vector search index using exact Inner Product (IndexFlatIP).
    
    When embeddings are L2-normalized, inner product is mathematically equivalent
    to cosine similarity:
        dot(u, v) = ||u|| ||v|| cos(theta) = cos(theta) (for ||u|| = ||v|| = 1)
    
    This index supports building from pre-embedded chunks, incremental additions,
    and k-nearest-neighbor search.
    """

    def __init__(self, dimension: int | None = None, normalize: bool = True) -> None:
        """Initialize FAISSIndex.
        
        Args:
            dimension: Optional embedding vector dimensionality. If None, inferred
                       automatically upon first build/add.
            normalize: Whether to defensively L2-normalize vectors prior to indexing
                       and search (enables exact cosine similarity).
        """
        self.dimension = dimension
        self.normalize = normalize
        self._index: faiss.IndexFlatIP | None = None
        self._id_to_chunk: dict[int, Chunk] = {}

        if self.dimension is not None:
            self._init_index(self.dimension)

    def _init_index(self, dimension: int) -> None:
        """Instantiate underlying FAISS IndexFlatIP.
        
        Args:
            dimension: Vector dimensionality.
        """
        if dimension <= 0:
            raise ValueError(f"Index dimension must be positive, got {dimension}")
        self.dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)

    def _prepare_vectors(self, embeddings: Sequence[Sequence[float]]) -> np.ndarray:
        """Convert float embeddings to float32 NumPy array with optional L2-normalization.
        
        Args:
            embeddings: Sequence of numeric embedding vectors.
            
        Returns:
            Contiguous float32 NumPy matrix suitable for FAISS.
        """
        matrix = np.array(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D embedding matrix, got shape {matrix.shape}")
        
        if self.normalize:
            # Defensive L2 normalization
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            # Avoid division by zero for zero vectors
            norms[norms == 0] = 1.0
            matrix = matrix / norms

        return np.ascontiguousarray(matrix, dtype=np.float32)

    def build(self, chunks: list[Chunk]) -> None:
        """Build a fresh FAISS index from pre-embedded chunks.
        
        Args:
            chunks: Non-empty list of Chunks with embedding vectors populated.
            
        Raises:
            ValueError: If chunks is empty or any chunk lacks an embedding.
        """
        if not chunks:
            raise ValueError("Cannot build FAISS index with an empty chunks list.")

        self.clear()
        self.add(chunks)

    def add(self, chunks: list[Chunk]) -> None:
        """Incrementally add pre-embedded chunks to the index.
        
        Args:
            chunks: List of Chunks to add. All chunks must have valid embeddings.
            
        Raises:
            ValueError: If any chunk lacks an embedding or vector dimensions mismatch.
        """
        if not chunks:
            return

        embeddings: list[list[float]] = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(
                    f"Chunk '{chunk.id}' has no embedding. "
                    "Generate embeddings before indexing."
                )
            embeddings.append(chunk.embedding)

        dim = len(embeddings[0])
        if self.dimension is None:
            self._init_index(dim)
        elif self.dimension != dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {dim}"
            )

        # Validate all incoming chunks match expected dimension
        for i, emb in enumerate(embeddings):
            if len(emb) != self.dimension:
                raise ValueError(
                    f"Chunk '{chunks[i].id}' embedding dimension {len(emb)} "
                    f"does not match index dimension {self.dimension}."
                )

        matrix = self._prepare_vectors(embeddings)
        assert self._index is not None

        start_id = len(self._id_to_chunk)
        self._index.add(matrix)

        for offset, chunk in enumerate(chunks):
            self._id_to_chunk[start_id + offset] = chunk

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """Search the FAISS index for nearest neighbors.
        
        Args:
            query_vector: Dense query embedding vector.
            top_k: Number of nearest neighbors to retrieve.
            
        Returns:
            List of (Chunk, similarity_score) tuples ordered descending by score.
            
        Raises:
            ValueError: If index is empty, query_vector dimension mismatches, or top_k <= 0.
        """
        if self.size() == 0 or self._index is None:
            raise ValueError("Cannot search an empty FAISS index. Build or add chunks first.")

        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        if len(query_vector) != self.dimension:
            raise ValueError(
                f"Query vector dimension mismatch: expected {self.dimension}, got {len(query_vector)}"
            )

        q_mat = self._prepare_vectors([query_vector])
        k = min(top_k, self.size())
        scores, indices = self._index.search(q_mat, k)

        results: list[tuple[Chunk, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1 and idx in self._id_to_chunk:
                results.append((self._id_to_chunk[idx], float(score)))

        return results

    def clear(self) -> None:
        """Reset the internal FAISS index and chunk mappings."""
        self._index = None
        self._id_to_chunk.clear()
        if self.dimension is not None:
            self._init_index(self.dimension)

    def size(self) -> int:
        """Return the number of vectors indexed in FAISS."""
        return len(self._id_to_chunk)


class FAISSRetriever(Retriever):
    """Dense retriever powered by FAISS vector indexing.
    
    Adheres to RetrievLab's Retriever interface. Integrates with an EmbeddingClient
    for on-the-fly query encoding and delegates vector similarity search to FAISSIndex.
    """

    def __init__(
        self,
        client: EmbeddingClient,
        index: FAISSIndex | None = None,
    ) -> None:
        """Initialize FAISSRetriever.
        
        Args:
            client: EmbeddingClient instance for generating query embeddings.
            index: Optional pre-configured FAISSIndex. If None, a new FAISSIndex is created.
        """
        self.embedding_model = client
        self.index = index if index is not None else FAISSIndex()

    def retrieve(
        self,
        query: str,
        top_k: int,
        chunks: list[Chunk],
    ) -> list[SearchResult]:
        """Retrieve top-k most semantically similar chunks for a query using FAISS.
        
        Args:
            query: Query string.
            top_k: Number of results to return.
            chunks: Candidate chunks to search over. If the index is empty or does not
                    match the supplied candidate set, it is built automatically.
            
        Returns:
            List of SearchResult objects sorted by descending similarity score.
            
        Raises:
            ValueError: If chunks have missing embeddings or top_k <= 0.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        if not chunks:
            return []

        # If index is empty or candidate chunk count differs, build index
        if self.index.size() != len(chunks):
            self.index.build(chunks)

        # Generate query embedding
        query_embedding = self.embedding_model.get_embeddings([query])[0]

        # Search index
        raw_results = self.index.search(query_embedding, top_k=top_k)

        # Convert to SearchResult objects
        return [SearchResult(chunk=chunk, score=score) for chunk, score in raw_results]
