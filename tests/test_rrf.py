"""Unit tests for Reciprocal Rank Fusion (RRF)."""

import pytest

from retrievlab.models import Chunk, SearchResult
from retrievlab.ranking.fusion import ReciprocalRankFusion, reciprocal_rank_fusion


def make_result(chunk_id: str, score: float = 1.0, doc_id: str = "doc1") -> SearchResult:
    return SearchResult(
        chunk=Chunk(id=chunk_id, document_id=doc_id, text=f"text for {chunk_id}"),
        score=score,
    )


def test_basic_two_way_rrf() -> None:
    """Test standard 2-way RRF calculation with default k=60."""
    list1 = [make_result("c_a"), make_result("c_b")]
    list2 = [make_result("c_b"), make_result("c_c")]

    # Manual calculations (k=60):
    # c_a: rank 1 in list1 -> 1 / 61
    # c_b: rank 2 in list1 + rank 1 in list2 -> 1 / 62 + 1 / 61
    # c_c: rank 2 in list2 -> 1 / 62
    expected_score_a = 1.0 / 61.0
    expected_score_b = (1.0 / 62.0) + (1.0 / 61.0)
    expected_score_c = 1.0 / 62.0

    results = reciprocal_rank_fusion([list1, list2], k=60)

    assert len(results) == 3
    assert [r.chunk.id for r in results] == ["c_b", "c_a", "c_c"]
    assert results[0].score == pytest.approx(expected_score_b)
    assert results[1].score == pytest.approx(expected_score_a)
    assert results[2].score == pytest.approx(expected_score_c)


def test_configurable_k_zero() -> None:
    """Test RRF with k=0 (pure reciprocal rank 1/r)."""
    list1 = [make_result("c1"), make_result("c2")]
    list2 = [make_result("c2"), make_result("c3")]

    # k=0:
    # c1: 1/1 = 1.0
    # c2: 1/2 + 1/1 = 1.5
    # c3: 1/2 = 0.5
    results = reciprocal_rank_fusion([list1, list2], k=0)

    assert len(results) == 3
    assert [r.chunk.id for r in results] == ["c2", "c1", "c3"]
    assert results[0].score == pytest.approx(1.5)
    assert results[1].score == pytest.approx(1.0)
    assert results[2].score == pytest.approx(0.5)


def test_configurable_k_custom() -> None:
    """Test custom k parameter (e.g., k=20)."""
    list1 = [make_result("c1")]
    list2 = [make_result("c1")]

    rrf = ReciprocalRankFusion(k=20)
    results = rrf.fuse([list1, list2])

    assert len(results) == 1
    # 1/21 + 1/21 = 2/21
    assert results[0].score == pytest.approx(2.0 / 21.0)


def test_weighted_rrf() -> None:
    """Test weighted RRF where one retriever has higher influence."""
    # List 1 has weight 3.0, List 2 has weight 1.0
    list1 = [make_result("c_lexical")]
    list2 = [make_result("c_dense")]

    results = reciprocal_rank_fusion(
        [list1, list2],
        k=60,
        weights=[3.0, 1.0],
    )

    assert len(results) == 2
    assert results[0].chunk.id == "c_lexical"
    assert results[0].score == pytest.approx(3.0 / 61.0)
    assert results[1].chunk.id == "c_dense"
    assert results[1].score == pytest.approx(1.0 / 61.0)


def test_zero_weight_ignored() -> None:
    """Test that a retriever with weight 0 is ignored in the fusion."""
    list1 = [make_result("c1")]
    list2 = [make_result("c2")]

    results = reciprocal_rank_fusion([list1, list2], k=60, weights=[1.0, 0.0])

    assert len(results) == 1
    assert results[0].chunk.id == "c1"


def test_three_way_fusion() -> None:
    """Test multi-retriever fusion with 3 distinct rank lists."""
    list1 = [make_result("c1"), make_result("c2"), make_result("c3")]
    list2 = [make_result("c2"), make_result("c3"), make_result("c4")]
    list3 = [make_result("c3"), make_result("c1"), make_result("c5")]

    # k=60
    # c3 appears at rank 3 (list1), rank 2 (list2), rank 1 (list3) -> 1/63 + 1/62 + 1/61
    # c2 appears at rank 2 (list1), rank 1 (list2) -> 1/62 + 1/61
    # c1 appears at rank 1 (list1), rank 2 (list3) -> 1/61 + 1/62
    expected_c3 = (1.0 / 63.0) + (1.0 / 62.0) + (1.0 / 61.0)
    results = reciprocal_rank_fusion([list1, list2, list3], k=60)

    assert results[0].chunk.id == "c3"
    assert results[0].score == pytest.approx(expected_c3)


def test_top_k_truncation() -> None:
    """Test top_k slicing on fused results."""
    list1 = [make_result("c1"), make_result("c2"), make_result("c3")]
    list2 = [make_result("c4"), make_result("c5")]

    results = reciprocal_rank_fusion([list1, list2], top_k=2)
    assert len(results) == 2


def test_empty_and_edge_case_inputs() -> None:
    """Test empty rankings, empty sublists, and single ranking list."""
    # Empty outer list
    assert reciprocal_rank_fusion([]) == []

    # Empty sublists
    assert reciprocal_rank_fusion([[], []]) == []

    # Single ranking list
    list1 = [make_result("c1"), make_result("c2")]
    single_res = reciprocal_rank_fusion([list1], k=60)
    assert len(single_res) == 2
    assert single_res[0].chunk.id == "c1"
    assert single_res[0].score == pytest.approx(1.0 / 61.0)
    assert single_res[1].chunk.id == "c2"
    assert single_res[1].score == pytest.approx(1.0 / 62.0)


def test_duplicate_chunks_in_single_list() -> None:
    """Test that if a chunk appears multiple times in the same list, only the first occurrence counts."""
    list1 = [make_result("c1"), make_result("c1")]
    results = reciprocal_rank_fusion([list1], k=60)

    assert len(results) == 1
    assert results[0].chunk.id == "c1"
    assert results[0].score == pytest.approx(1.0 / 61.0)


def test_deterministic_tie_breaking() -> None:
    """Test that equal-scoring chunks are ordered deterministically by chunk id."""
    list1 = [make_result("chunk_z"), make_result("chunk_a")]
    list2 = [make_result("chunk_a"), make_result("chunk_z")]

    # Both chunk_a and chunk_z will have score = 1/61 + 1/62
    results = reciprocal_rank_fusion([list1, list2], k=60)
    assert len(results) == 2
    assert results[0].score == pytest.approx(results[1].score)
    # Tie-break by chunk id alphabetically: chunk_a before chunk_z
    assert results[0].chunk.id == "chunk_a"
    assert results[1].chunk.id == "chunk_z"


def test_validation_errors() -> None:
    """Test input validation for invalid parameters."""
    # Negative k
    with pytest.raises(ValueError, match="k must be non-negative"):
        ReciprocalRankFusion(k=-1)

    # Negative weight in constructor
    with pytest.raises(ValueError, match="Weight at index 0 must be non-negative"):
        ReciprocalRankFusion(weights=[-0.5, 1.0])

    # Weights count mismatch during fuse
    rrf = ReciprocalRankFusion()
    list1 = [make_result("c1")]
    list2 = [make_result("c2")]
    with pytest.raises(ValueError, match="Number of weights .* does not match"):
        rrf.fuse([list1, list2], weights=[1.0])

    # Negative weight during fuse
    with pytest.raises(ValueError, match="Weight at index 1 must be non-negative"):
        rrf.fuse([list1, list2], weights=[1.0, -2.0])

    # Invalid top_k <= 0
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        rrf.fuse([list1], top_k=0)
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        rrf.fuse([list1], top_k=-5)


def test_convenience_function_parity() -> None:
    """Verify that reciprocal_rank_fusion helper matches ReciprocalRankFusion class."""
    list1 = [make_result("c1"), make_result("c2")]
    list2 = [make_result("c2"), make_result("c3")]

    res_class = ReciprocalRankFusion(k=40, weights=[1.5, 2.5]).fuse([list1, list2], top_k=2)
    res_func = reciprocal_rank_fusion([list1, list2], k=40, weights=[1.5, 2.5], top_k=2)

    assert len(res_class) == len(res_func)
    for r1, r2 in zip(res_class, res_func):
        assert r1.chunk.id == r2.chunk.id
        assert r1.score == pytest.approx(r2.score)
