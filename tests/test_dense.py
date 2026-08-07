"""
Unit tests for the Dense Retriever.
"""

import pytest

from retrievlab.models import Chunk
from retrievlab.retrieval.dense import DenseRetriever


class MockEmbeddingClient:
    """Mock embedding client for testing."""

    def __init__(self, embeddings: dict[str, list[float]]) -> None:
        """Initialize with a mapping of text to embeddings.
        
        Args:
            embeddings: Dictionary mapping text to embedding vectors.
        """
        self.embeddings = embeddings

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Return pre-stored embeddings for texts.
        
        Args:
            texts: List of text strings.
        
        Returns:
            List of embedding vectors.
        
        Raises:
            KeyError: If text is not in the embeddings dictionary.
        """
        return [self.embeddings[text] for text in texts]


def test_similarity_calculation() -> None:
    """Test that _similarity correctly computes dot product of normalized embeddings."""
    client = MockEmbeddingClient({"query": [1.0, 0.0]})
    retriever = DenseRetriever(client)

    # Test orthogonal vectors (should have zero similarity)
    sim_orthogonal = retriever._similarity([1.0, 0.0], [0.0, 1.0])
    assert sim_orthogonal == 0.0

    # Test identical vectors (should have similarity of 1.0)
    sim_identical = retriever._similarity([0.6, 0.8], [0.6, 0.8])
    assert sim_identical == pytest.approx(1.0)

    # Test opposite vectors (should have negative similarity)
    sim_opposite = retriever._similarity([1.0, 0.0], [-1.0, 0.0])
    assert sim_opposite == -1.0


def test_retrieval_ranking_correctness() -> None:
    """Test that retrieve() returns chunks in order of descending similarity."""
    client = MockEmbeddingClient({
        "python": [1.0, 0.0],
    })
    retriever = DenseRetriever(client)

    # Create chunks with different similarities to query
    c1 = Chunk(id="c1", document_id="doc1", text="python", embedding=[1.0, 0.0])  # perfect match
    c2 = Chunk(id="c2", document_id="doc1", text="java", embedding=[0.6, 0.8])   # partial match
    c3 = Chunk(id="c3", document_id="doc1", text="rust", embedding=[0.0, 1.0])   # orthogonal

    results = retriever.retrieve("python", top_k=3, chunks=[c1, c2, c3])

    assert len(results) == 3
    assert results[0].chunk.id == "c1"
    assert results[1].chunk.id == "c2"
    assert results[2].chunk.id == "c3"
    # Verify scores are in descending order
    assert results[0].score > results[1].score > results[2].score


def test_top_k_filtering() -> None:
    """Test that retrieve() respects the top_k parameter."""
    client = MockEmbeddingClient({
        "query": [1.0, 0.0],
    })
    retriever = DenseRetriever(client)

    chunks = [
        Chunk(id=f"c{i}", document_id="doc1", text=f"text{i}", embedding=[1.0, 0.0])
        for i in range(5)
    ]

    results = retriever.retrieve("query", top_k=3, chunks=chunks)
    assert len(results) == 3


def test_deterministic_retrieval() -> None:
    """Test that retrieve() produces consistent results across multiple calls."""
    client = MockEmbeddingClient({
        "query": [1.0, 0.0],
    })
    retriever = DenseRetriever(client)

    chunks = [
        Chunk(id="c1", document_id="doc1", text="text1", embedding=[0.9, 0.1]),
        Chunk(id="c2", document_id="doc1", text="text2", embedding=[0.8, 0.2]),
        Chunk(id="c3", document_id="doc1", text="text3", embedding=[0.7, 0.3]),
    ]

    # Run retrieval multiple times
    results1 = retriever.retrieve("query", top_k=3, chunks=chunks)
    results2 = retriever.retrieve("query", top_k=3, chunks=chunks)
    results3 = retriever.retrieve("query", top_k=3, chunks=chunks)

    # All results should be identical
    assert [r.chunk.id for r in results1] == [r.chunk.id for r in results2]
    assert [r.chunk.id for r in results2] == [r.chunk.id for r in results3]
    assert [r.score for r in results1] == [r.score for r in results2]


def test_missing_embedding_error() -> None:
    """Test that retrieve() raises ValueError when a chunk has no embedding."""
    client = MockEmbeddingClient({
        "query": [1.0, 0.0],
    })
    retriever = DenseRetriever(client)

    # Create a chunk without an embedding
    chunk_with_embedding = Chunk(
        id="c1",
        document_id="doc1",
        text="text1",
        embedding=[1.0, 0.0],
    )
    chunk_without_embedding = Chunk(
        id="c2",
        document_id="doc1",
        text="text2",
        embedding=None,  # Missing embedding
    )

    with pytest.raises(ValueError) as exc_info:
        retriever.retrieve(
            "query",
            top_k=2,
            chunks=[chunk_with_embedding, chunk_without_embedding],
        )

    assert "c2" in str(exc_info.value)
    assert "no embedding" in str(exc_info.value)


def test_embedding_dimension_mismatch() -> None:
    """Test that _similarity raises ValueError for mismatched embedding dimensions."""
    client = MockEmbeddingClient({"query": [1.0, 0.0]})
    retriever = DenseRetriever(client)

    embedding1 = [1.0, 0.0, 0.5]  # 3-dimensional
    embedding2 = [1.0, 0.0]       # 2-dimensional

    with pytest.raises(ValueError) as exc_info:
        retriever._similarity(embedding1, embedding2)

    assert "dimension mismatch" in str(exc_info.value)
    assert "3" in str(exc_info.value)
    assert "2" in str(exc_info.value)


def test_empty_chunk_list() -> None:
    """Test that retrieve() handles an empty chunk list gracefully."""
    client = MockEmbeddingClient({
        "query": [1.0, 0.0],
    })
    retriever = DenseRetriever(client)

    results = retriever.retrieve("query", top_k=10, chunks=[])
    assert len(results) == 0
    assert isinstance(results, list)


def test_retriever_interface_compliance() -> None:
    """Test that DenseRetriever implements the Retriever interface."""
    from retrievlab.retrieval.interface import Retriever

    client = MockEmbeddingClient({"query": [1.0, 0.0]})
    retriever = DenseRetriever(client)

    # Verify it's an instance of Retriever
    assert isinstance(retriever, Retriever)

    # Verify it has the required retrieve method
    assert hasattr(retriever, "retrieve")
    assert callable(retriever.retrieve)
