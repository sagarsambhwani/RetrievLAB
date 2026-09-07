"""
Unit tests for FAISS vector indexing and FAISSRetriever.
"""

import pytest

from retrievlab.models import Chunk
from retrievlab.indexing.interface import VectorIndex
from retrievlab.indexing.faiss import FAISSIndex, FAISSRetriever
from retrievlab.retrieval.interface import Retriever
from retrievlab.retrieval.dense import DenseRetriever


class MockEmbeddingClient:
    """Mock embedding client for deterministic test vectors."""

    def __init__(self, embeddings: dict[str, list[float]]) -> None:
        self.embeddings = embeddings

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [self.embeddings[text] for text in texts]


def test_interface_compliance() -> None:
    """Verify VectorIndex and Retriever interface compliance."""
    index = FAISSIndex(dimension=4)
    assert isinstance(index, VectorIndex)
    assert hasattr(index, "build")
    assert hasattr(index, "add")
    assert hasattr(index, "search")
    assert hasattr(index, "clear")
    assert hasattr(index, "size")

    client = MockEmbeddingClient({"query": [1.0, 0.0, 0.0, 0.0]})
    retriever = FAISSRetriever(client=client, index=index)
    assert isinstance(retriever, Retriever)
    assert hasattr(retriever, "retrieve")


def test_faiss_index_build_and_search() -> None:
    """Test standard build and k-NN search ranking on FAISSIndex."""
    c1 = Chunk(id="c1", document_id="doc1", text="chunk 1", embedding=[1.0, 0.0])
    c2 = Chunk(id="c2", document_id="doc1", text="chunk 2", embedding=[0.6, 0.8])
    c3 = Chunk(id="c3", document_id="doc1", text="chunk 3", embedding=[0.0, 1.0])

    index = FAISSIndex()
    index.build([c1, c2, c3])

    assert index.size() == 3
    assert index.dimension == 2

    # Query matching c1 exactly
    results = index.search([1.0, 0.0], top_k=3)
    assert len(results) == 3
    assert results[0][0].id == "c1"
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)
    assert results[1][0].id == "c2"
    assert results[1][1] == pytest.approx(0.6, abs=1e-5)
    assert results[2][0].id == "c3"
    assert results[2][1] == pytest.approx(0.0, abs=1e-5)


def test_incremental_add_and_clear() -> None:
    """Test incremental add, size tracking, and clear functionality."""
    index = FAISSIndex(dimension=2)
    assert index.size() == 0

    c1 = Chunk(id="c1", document_id="doc1", text="chunk 1", embedding=[1.0, 0.0])
    c2 = Chunk(id="c2", document_id="doc1", text="chunk 2", embedding=[0.0, 1.0])

    index.add([c1])
    assert index.size() == 1

    index.add([c2])
    assert index.size() == 2

    results = index.search([0.0, 1.0], top_k=2)
    assert results[0][0].id == "c2"
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    index.clear()
    assert index.size() == 0

    with pytest.raises(ValueError, match="Cannot search an empty FAISS index"):
        index.search([1.0, 0.0], top_k=1)


def test_mathematical_equivalence_with_dense_retriever() -> None:
    """Verify FAISSRetriever produces exact ranking and scores as DenseRetriever."""
    # Create normalized test embeddings
    vectors = [
        [0.8, 0.6],
        [-0.6, 0.8],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.7071, 0.7071],
    ]
    chunks = [
        Chunk(id=f"c{i}", document_id="doc1", text=f"text{i}", embedding=v)
        for i, v in enumerate(vectors)
    ]

    client = MockEmbeddingClient({"query": [0.6, 0.8]})
    dense_retriever = DenseRetriever(client)
    faiss_retriever = FAISSRetriever(client)

    dense_results = dense_retriever.retrieve("query", top_k=5, chunks=chunks)
    faiss_results = faiss_retriever.retrieve("query", top_k=5, chunks=chunks)

    assert len(dense_results) == len(faiss_results) == 5

    for d_res, f_res in zip(dense_results, faiss_results):
        assert d_res.chunk.id == f_res.chunk.id
        assert d_res.score == pytest.approx(f_res.score, abs=1e-5)


def test_top_k_larger_than_index_size() -> None:
    """Test searching with top_k greater than the number of indexed chunks."""
    c1 = Chunk(id="c1", document_id="doc1", text="c1", embedding=[1.0, 0.0])
    c2 = Chunk(id="c2", document_id="doc1", text="c2", embedding=[0.0, 1.0])

    index = FAISSIndex()
    index.build([c1, c2])

    results = index.search([1.0, 0.0], top_k=10)
    assert len(results) == 2
    assert results[0][0].id == "c1"
    assert results[1][0].id == "c2"


def test_error_on_missing_embeddings() -> None:
    """Test that indexing chunks with missing embeddings raises ValueError."""
    c1 = Chunk(id="c1", document_id="doc1", text="c1", embedding=[1.0, 0.0])
    c2 = Chunk(id="c2", document_id="doc1", text="c2", embedding=None)

    index = FAISSIndex()
    with pytest.raises(ValueError, match="no embedding"):
        index.build([c1, c2])

    with pytest.raises(ValueError, match="no embedding"):
        index.add([c2])


def test_error_on_dimension_mismatch() -> None:
    """Test that dimension mismatches in search and add are caught."""
    c1 = Chunk(id="c1", document_id="doc1", text="c1", embedding=[1.0, 0.0, 0.0])
    c2 = Chunk(id="c2", document_id="doc1", text="c2", embedding=[1.0, 0.0])

    index = FAISSIndex(dimension=3)
    index.add([c1])

    with pytest.raises(ValueError, match="dimension mismatch"):
        index.add([c2])

    with pytest.raises(ValueError, match="dimension mismatch"):
        index.search([1.0, 0.0], top_k=1)


def test_error_on_invalid_top_k_or_dimension() -> None:
    """Test validation on non-positive top_k and dimension."""
    with pytest.raises(ValueError, match="dimension must be positive"):
        FAISSIndex(dimension=0)

    with pytest.raises(ValueError, match="empty chunks list"):
        FAISSIndex().build([])

    index = FAISSIndex()
    index.build([Chunk(id="c1", document_id="d1", text="t", embedding=[1.0, 0.0])])

    with pytest.raises(ValueError, match="top_k must be positive"):
        index.search([1.0, 0.0], top_k=0)

    client = MockEmbeddingClient({"query": [1.0, 0.0]})
    retriever = FAISSRetriever(client, index)
    with pytest.raises(ValueError, match="top_k must be positive"):
        retriever.retrieve("query", top_k=-1, chunks=[])


def test_empty_chunks_retrieval_returns_empty_list() -> None:
    """Test that FAISSRetriever handles empty candidate chunks gracefully."""
    client = MockEmbeddingClient({"query": [1.0, 0.0]})
    retriever = FAISSRetriever(client)

    results = retriever.retrieve("query", top_k=5, chunks=[])
    assert results == []
