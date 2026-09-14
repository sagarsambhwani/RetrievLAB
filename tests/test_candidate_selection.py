"""
Unit tests for candidate generation and candidate pool data structures (RLB-310).
"""

import pytest
from retrievlab.models import Chunk, SearchResult
from retrievlab.retrieval.interface import Retriever
from retrievlab.selection import (
    Candidate,
    CandidatePool,
    MultiRetrieverCandidateGenerator,
    SingleRetrieverCandidateGenerator,
)


class MockRetriever(Retriever):
    """Mock retriever returning a pre-defined sequence of chunks and scores."""

    def __init__(self, return_chunks: list[tuple[Chunk, float]]) -> None:
        self.return_chunks = return_chunks

    def retrieve(
        self, query: str, top_k: int, chunks: list[Chunk]
    ) -> list[SearchResult]:
        return [
            SearchResult(chunk=c, score=s)
            for c, s in self.return_chunks[:top_k]
        ]


@pytest.fixture
def sample_chunks():
    c1 = Chunk(id="chunk_1", document_id="doc1", text="Text 1")
    c2 = Chunk(id="chunk_2", document_id="doc1", text="Text 2")
    c3 = Chunk(id="chunk_3", document_id="doc2", text="Text 3")
    c4 = Chunk(id="chunk_4", document_id="doc2", text="Text 4")
    return [c1, c2, c3, c4]


def test_candidate_model(sample_chunks):
    c1 = sample_chunks[0]
    candidate = Candidate(
        chunk=c1,
        retriever_scores={"bm25": 12.5, "dense": 0.88},
        retriever_ranks={"bm25": 1, "dense": 3},
        sources=["bm25", "dense"],
    )
    assert candidate.id == "chunk_1"
    assert candidate.retriever_scores["bm25"] == 12.5
    assert candidate.retriever_ranks["dense"] == 3
    assert candidate.sources == ["bm25", "dense"]


def test_single_retriever_generator(sample_chunks):
    c1, c2, c3, _ = sample_chunks
    mock = MockRetriever([(c1, 10.0), (c2, 8.0), (c3, 6.0)])
    generator = SingleRetrieverCandidateGenerator(mock, name="bm25")

    pool = generator.generate(query="test query", top_k_per_retriever=2, chunks=sample_chunks)

    assert isinstance(pool, CandidatePool)
    assert pool.query == "test query"
    assert pool.top_k_per_retriever == 2
    assert len(pool) == 2

    # Check candidates
    cand1 = pool.get_candidate("chunk_1")
    assert cand1 is not None
    assert cand1.retriever_scores == {"bm25": 10.0}
    assert cand1.retriever_ranks == {"bm25": 1}
    assert cand1.sources == ["bm25"]

    cand2 = pool.get_candidate("chunk_2")
    assert cand2 is not None
    assert cand2.retriever_scores == {"bm25": 8.0}
    assert cand2.retriever_ranks == {"bm25": 2}

    assert pool.get_candidate("chunk_3") is None
    assert pool.get_chunk_ids() == ["chunk_1", "chunk_2"]
    assert len(pool.get_chunks()) == 2


def test_multi_retriever_deduplication_and_provenance(sample_chunks):
    c1, c2, c3, c4 = sample_chunks

    # Retriever A returns: chunk_1 (rank 1), chunk_2 (rank 2)
    retriever_a = MockRetriever([(c1, 15.0), (c2, 10.0)])
    # Retriever B returns: chunk_2 (rank 1), chunk_3 (rank 2)
    retriever_b = MockRetriever([(c2, 0.95), (c3, 0.80)])

    generator = MultiRetrieverCandidateGenerator(
        {"bm25": retriever_a, "dense": retriever_b}
    )

    pool = generator.generate(
        query="hybrid query", top_k_per_retriever=2, chunks=sample_chunks
    )

    # Union of {chunk_1, chunk_2} and {chunk_2, chunk_3} -> 3 unique candidates
    assert len(pool) == 3
    assert set(pool.get_chunk_ids()) == {"chunk_1", "chunk_2", "chunk_3"}

    # chunk_1: only in bm25
    cand1 = pool.get_candidate("chunk_1")
    assert cand1 is not None
    assert cand1.sources == ["bm25"]
    assert cand1.retriever_scores == {"bm25": 15.0}
    assert cand1.retriever_ranks == {"bm25": 1}

    # chunk_2: in both bm25 and dense
    cand2 = pool.get_candidate("chunk_2")
    assert cand2 is not None
    assert set(cand2.sources) == {"bm25", "dense"}
    assert cand2.retriever_scores == {"bm25": 10.0, "dense": 0.95}
    assert cand2.retriever_ranks == {"bm25": 2, "dense": 1}

    # chunk_3: only in dense
    cand3 = pool.get_candidate("chunk_3")
    assert cand3 is not None
    assert cand3.sources == ["dense"]
    assert cand3.retriever_scores == {"dense": 0.80}
    assert cand3.retriever_ranks == {"dense": 2}


def test_multi_retriever_non_overlapping(sample_chunks):
    c1, c2, c3, c4 = sample_chunks
    retriever_a = MockRetriever([(c1, 1.0), (c2, 0.8)])
    retriever_b = MockRetriever([(c3, 0.9), (c4, 0.7)])

    generator = MultiRetrieverCandidateGenerator({"ret1": retriever_a, "ret2": retriever_b})
    pool = generator.generate("query", top_k_per_retriever=2, chunks=sample_chunks)

    assert len(pool) == 4
    assert pool.get_chunk_ids() == ["chunk_1", "chunk_2", "chunk_3", "chunk_4"]


def test_empty_retriever_config():
    with pytest.raises(ValueError, match="Must provide at least one retriever"):
        MultiRetrieverCandidateGenerator({})


def test_invalid_top_k(sample_chunks):
    c1 = sample_chunks[0]
    mock = MockRetriever([(c1, 1.0)])
    generator = SingleRetrieverCandidateGenerator(mock)

    with pytest.raises(ValueError, match="top_k_per_retriever must be positive"):
        generator.generate("query", top_k_per_retriever=0, chunks=sample_chunks)

    with pytest.raises(ValueError, match="top_k_per_retriever must be positive"):
        generator.generate("query", top_k_per_retriever=-5, chunks=sample_chunks)


def test_pool_iteration(sample_chunks):
    c1, c2, _, _ = sample_chunks
    mock = MockRetriever([(c1, 2.0), (c2, 1.0)])
    generator = SingleRetrieverCandidateGenerator(mock)
    pool = generator.generate("query", top_k_per_retriever=2, chunks=sample_chunks)

    ids = [c.id for c in pool]
    assert ids == ["chunk_1", "chunk_2"]
