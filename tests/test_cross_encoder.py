"""
Unit tests for CrossEncoderReRanker and ReRanker interface.
"""

import pytest

from retrievlab.models import Chunk, SearchResult
from retrievlab.ranking.interface import ReRanker
from retrievlab.ranking.cross_encoder import CrossEncoderReRanker
from retrievlab.selection.candidate import Candidate, CandidatePool


@pytest.fixture(scope="module")
def ranker() -> CrossEncoderReRanker:
    """Instantiate CrossEncoderReRanker once for the test session."""
    return CrossEncoderReRanker()


def test_cross_encoder_inherits_interface(ranker: CrossEncoderReRanker) -> None:
    assert isinstance(ranker, ReRanker)
    assert ranker.batch_size == 32
    assert "MiniLM" in ranker.model_name


def test_rerank_chunk_list(ranker: CrossEncoderReRanker) -> None:
    chunk_rel = Chunk(
        id="c1",
        document_id="d1",
        text="Aspirin (acetylsalicylic acid) is commonly used to treat pain, reduce fever, and prevent heart disease.",
    )
    chunk_irrel = Chunk(
        id="c2",
        document_id="d2",
        text="The solar system consists of eight planets orbiting the central Sun in elliptical paths.",
    )

    query = "What diseases does aspirin prevent or treat?"
    results = ranker.rerank(query, candidates=[chunk_irrel, chunk_rel])

    assert len(results) == 2
    assert isinstance(results[0], SearchResult)
    # The relevant chunk should be ranked #1 with a higher score
    assert results[0].chunk.id == "c1"
    assert results[1].chunk.id == "c2"
    assert results[0].score > results[1].score


def test_rerank_candidate_pool(ranker: CrossEncoderReRanker) -> None:
    chunk_a = Chunk(
        id="ca",
        document_id="da",
        text="COVID-19 is an infectious disease caused by the SARS-CoV-2 coronavirus.",
    )
    chunk_b = Chunk(
        id="cb",
        document_id="db",
        text="Chocolate chip cookies are baked with flour, butter, sugar, and chocolate chunks.",
    )

    cand_a = Candidate(chunk=chunk_a, sources=["dense"])
    cand_b = Candidate(chunk=chunk_b, sources=["bm25"])

    pool = CandidatePool(
        query="What virus causes COVID-19?",
        candidates=[cand_b, cand_a],
        top_k_per_retriever=10,
    )

    results = ranker.rerank(pool.query, candidates=pool)

    assert len(results) == 2
    assert results[0].chunk.id == "ca"
    assert results[1].chunk.id == "cb"
    assert results[0].score > results[1].score


def test_rerank_top_k_limiting(ranker: CrossEncoderReRanker) -> None:
    chunks = [
        Chunk(id=f"c_{i}", document_id=f"d_{i}", text=f"Document text number {i} with some content.")
        for i in range(5)
    ]

    results = ranker.rerank("Document query", candidates=chunks, top_k=2)
    assert len(results) == 2
    assert results[0].score >= results[1].score


def test_rerank_empty_candidates(ranker: CrossEncoderReRanker) -> None:
    results = ranker.rerank("Query", candidates=[])
    assert results == []

    empty_pool = CandidatePool(query="Query", candidates=[])
    pool_results = ranker.rerank(empty_pool.query, candidates=empty_pool)
    assert pool_results == []


def test_rerank_invalid_type_raises(ranker: CrossEncoderReRanker) -> None:
    with pytest.raises(TypeError, match="Expected Chunk, Candidate, or CandidatePool"):
        ranker.rerank("Query", candidates=["invalid_string_candidate"])  # type: ignore[list-item]
