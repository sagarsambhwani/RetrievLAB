import math
import pytest

from retrievlab.models import Chunk
from retrievlab.retrieval.bm25 import BM25Retriever


def test_basic_correctness():
    """Test standard BM25 statistics and scoring calculations."""
    retriever = BM25Retriever()

    c1 = Chunk(id="c1", document_id="doc1", text="python code")
    c2 = Chunk(id="c2", document_id="doc1", text="python java python")

    retriever.index([c1, c2])

    # Check length statistics
    assert retriever.chunk_lengths == {"c1": 2, "c2": 3}
    assert retriever.average_chunk_length == 2.5

    # Check term frequency statistics
    assert retriever.term_frequencies["python"]["c1"] == 1
    assert retriever.term_frequencies["python"]["c2"] == 2
    assert retriever.term_frequencies["code"]["c1"] == 1
    assert retriever.term_frequencies["java"]["c2"] == 1

    # Check IDF precomputations (total_chunks = 2)
    # df("python") = 2 => idf = log(2/2) = 0.0
    assert retriever._inverse_document_frequency("python") == 0.0
    # df("code") = 1 => idf = log(2/1) = log(2)
    assert math.isclose(retriever._inverse_document_frequency("code"), math.log(2.0))
    # df("java") = 1 => idf = log(2/1) = log(2)
    assert math.isclose(retriever._inverse_document_frequency("java"), math.log(2.0))

    # Test retrieval logic
    results = retriever.retrieve("code", top_k=2, chunks=[c1, c2])
    assert len(results) == 2
    # c1 should have a positive score, c2 should have 0.0
    assert results[0].chunk.id == "c1"
    assert results[0].score > 0.0
    assert results[1].chunk.id == "c2"
    assert results[1].score == 0.0


def test_deterministic_ranking():
    """Verify that ranking is stable and consistent with the input chunk ordering (as code wasn't modified)."""
    retriever = BM25Retriever()
    c1 = Chunk(id="c1", document_id="doc1", text="python")
    c2 = Chunk(id="c2", document_id="doc1", text="python")

    retriever.index([c1, c2])

    # Both chunks have identical scores for query "python" (which is 0.0 because df=2, idf=0)
    # The output ordering is stable and preserves input chunk sequence.
    results_ordered_1 = retriever.retrieve("python", top_k=2, chunks=[c1, c2])
    assert results_ordered_1[0].chunk.id == "c1"
    assert results_ordered_1[1].chunk.id == "c2"

    results_ordered_2 = retriever.retrieve("python", top_k=2, chunks=[c2, c1])
    assert results_ordered_2[0].chunk.id == "c2"
    assert results_ordered_2[1].chunk.id == "c1"


def test_unknown_terms():
    """Verify that unknown terms return zero score contribution and do not raise errors."""
    retriever = BM25Retriever()
    c1 = Chunk(id="c1", document_id="doc1", text="python code")
    retriever.index([c1])

    # Unknown term only
    results = retriever.retrieve("unknown", top_k=1, chunks=[c1])
    assert len(results) == 1
    assert results[0].score == 0.0

    # Mix of known and unknown terms
    results_mixed = retriever.retrieve("python unknown", top_k=1, chunks=[c1])
    assert len(results_mixed) == 1
    # df("python") = 1, total_chunks = 1 => idf = log(1) = 0.0, so total score remains 0.0
    assert results_mixed[0].score == 0.0


def test_empty_corpus():
    """Verify that empty corpora are handled gracefully."""
    retriever = BM25Retriever()
    retriever.index([])

    assert retriever.average_chunk_length == 0.0
    assert retriever.idf == {}
    assert retriever.term_frequencies == {}

    # Scoring against an empty list of candidate chunks
    results = retriever.retrieve("python", top_k=5, chunks=[])
    assert results == []

    # Scoring against candidate chunks when retriever stats are empty
    c1 = Chunk(id="c1", document_id="doc1", text="python code")
    results_with_candidate = retriever.retrieve("python", top_k=1, chunks=[c1])
    assert len(results_with_candidate) == 1
    assert results_with_candidate[0].score == 0.0


def test_edge_case_inputs():
    """Test retrieval behavior with empty queries or queries with only whitespace and punctuation."""
    retriever = BM25Retriever()
    c1 = Chunk(id="c1", document_id="doc1", text="python code")
    retriever.index([c1])

    # Empty query string
    results_empty = retriever.retrieve("", top_k=1, chunks=[c1])
    assert len(results_empty) == 1
    assert results_empty[0].score == 0.0

    # Whitespace only query string
    results_spaces = retriever.retrieve("   ", top_k=1, chunks=[c1])
    assert len(results_spaces) == 1
    assert results_spaces[0].score == 0.0

    # Punctuation only query string (should return no tokens)
    results_punc = retriever.retrieve("!!! ???", top_k=1, chunks=[c1])
    assert len(results_punc) == 1
    assert results_punc[0].score == 0.0


def test_parameter_validation():
    """Verify that BM25 parameters k1 and b are strictly validated."""
    # Valid configurations
    BM25Retriever(k1=0.0, b=0.0)
    BM25Retriever(k1=2.5, b=1.0)
    BM25Retriever(k1=1.2, b=0.5)

    # Invalid k1 (must be >= 0)
    with pytest.raises(ValueError, match="k1 must be non-negative"):
        BM25Retriever(k1=-0.5)

    # Invalid b (must be in [0, 1])
    with pytest.raises(ValueError, match="b must be in the range"):
        BM25Retriever(b=-0.1)

    with pytest.raises(ValueError, match="b must be in the range"):
        BM25Retriever(b=1.1)


def test_configurable_k1_and_b():
    """Verify that custom k1 and b parameters alter scoring calculations appropriately."""
    c1 = Chunk(id="c1", document_id="doc1", text="fastapi framework")
    c2 = Chunk(id="c2", document_id="doc1", text="fastapi modern web framework for building fast api with python")
    c3 = Chunk(id="c3", document_id="doc1", text="docker container runtime")

    # When b=0.0, length normalization is disabled. Both c1 and c2 have 1 occurrence of 'fastapi',
    # so they should receive identical term-frequency scores for 'fastapi'.
    retriever_no_len_norm = BM25Retriever(k1=1.5, b=0.0)
    retriever_no_len_norm.index([c1, c2, c3])

    score_c1_b0 = retriever_no_len_norm.bm25_score(["fastapi"], c1)
    score_c2_b0 = retriever_no_len_norm.bm25_score(["fastapi"], c2)
    assert score_c1_b0 > 0.0
    assert math.isclose(score_c1_b0, score_c2_b0)

    # When b=1.0 (full length normalization), shorter chunk c1 gets a significantly higher score than long chunk c2.
    retriever_full_len_norm = BM25Retriever(k1=1.5, b=1.0)
    retriever_full_len_norm.index([c1, c2, c3])

    score_c1_b1 = retriever_full_len_norm.bm25_score(["fastapi"], c1)
    score_c2_b1 = retriever_full_len_norm.bm25_score(["fastapi"], c2)
    assert score_c1_b1 > score_c2_b1


if __name__ == "__main__":
    print("Running BM25 unit and edge case tests...")
    test_basic_correctness()
    test_deterministic_ranking()
    test_unknown_terms()
    test_empty_corpus()
    test_edge_case_inputs()
    test_parameter_validation()
    test_configurable_k1_and_b()
    print("All tests passed successfully!")


