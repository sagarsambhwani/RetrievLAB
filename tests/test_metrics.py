import math
import pytest
from retrievlab.models import SearchResult, Chunk
from retrievlab.evaluation.benchmark import BenchmarkCase
from retrievlab.evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    reciprocal_rank,
    hit_at_k,
    ndcg_at_k,
    average_precision_at_k,
)

@pytest.fixture
def sample_case():
    return BenchmarkCase(query="test query", relevant_chunk_ids=["chunk1", "chunk2"])

@pytest.fixture
def sample_results():
    c1 = Chunk(id="chunk1", document_id="doc1", text="content 1")
    c2 = Chunk(id="chunk3", document_id="doc1", text="content 3")
    c3 = Chunk(id="chunk2", document_id="doc1", text="content 2")
    return [
        SearchResult(chunk=c1, score=0.9),
        SearchResult(chunk=c2, score=0.8),
        SearchResult(chunk=c3, score=0.7),
    ]

def test_recall_without_k(sample_results, sample_case):
    # All 3 results considered: chunk1 and chunk2 retrieved out of 2 relevant
    score = recall_at_k(sample_results, sample_case)
    assert score == 1.0

def test_recall_with_k(sample_results, sample_case):
    # k=1: only chunk1 retrieved -> 1/2 relevant = 0.5
    score_k1 = recall_at_k(sample_results, sample_case, k=1)
    assert score_k1 == 0.5

    # k=2: chunk1 and chunk3 retrieved -> 1/2 relevant = 0.5
    score_k2 = recall_at_k(sample_results, sample_case, k=2)
    assert score_k2 == 0.5

    # k=3: chunk1, chunk3, chunk2 retrieved -> 2/2 relevant = 1.0
    score_k3 = recall_at_k(sample_results, sample_case, k=3)
    assert score_k3 == 1.0

def test_precision_with_and_without_k(sample_results, sample_case):
    # Without k: 2 relevant out of 3 total retrieved -> 2/3
    p_all = precision_at_k(sample_results, sample_case)
    assert abs(p_all - 2/3) < 1e-6

    # k=1: 1 relevant out of 1 -> 1.0
    p_k1 = precision_at_k(sample_results, sample_case, k=1)
    assert p_k1 == 1.0

    # k=2: 1 relevant out of 2 -> 0.5
    p_k2 = precision_at_k(sample_results, sample_case, k=2)
    assert p_k2 == 0.5

def test_invalid_k(sample_results, sample_case):
    with pytest.raises(ValueError):
        recall_at_k(sample_results, sample_case, k=0)

    with pytest.raises(ValueError):
        precision_at_k(sample_results, sample_case, k=-1)

    with pytest.raises(ValueError):
        hit_at_k(sample_results, sample_case, k=0)

    with pytest.raises(ValueError):
        ndcg_at_k(sample_results, sample_case, k=-2)

    with pytest.raises(ValueError):
        average_precision_at_k(sample_results, sample_case, k=0)

def test_reciprocal_rank(sample_results, sample_case):
    # First item (chunk1) is relevant -> rank 1 -> RR = 1.0
    rr_first = reciprocal_rank(sample_results, sample_case)
    assert rr_first == 1.0

    # Rank 2 item (chunk2) relevant, rank 1 (chunk3) irrelevant -> rank 2 -> RR = 0.5
    c2 = Chunk(id="chunk3", document_id="doc1", text="content 3")
    c3 = Chunk(id="chunk2", document_id="doc1", text="content 2")
    rank2_results = [SearchResult(chunk=c2, score=0.9), SearchResult(chunk=c3, score=0.8)]
    assert reciprocal_rank(rank2_results, sample_case) == 0.5

    # No relevant items retrieved -> RR = 0.0
    c_none = Chunk(id="chunk99", document_id="doc1", text="content 99")
    no_match_results = [SearchResult(chunk=c_none, score=0.9)]
    assert reciprocal_rank(no_match_results, sample_case) == 0.0


def test_hit_at_k(sample_results, sample_case):
    # k=1: chunk1 is in top 1 -> 1.0
    assert hit_at_k(sample_results, sample_case, k=1) == 1.0

    # Rank 2 only relevant:
    c_irrel = Chunk(id="chunk_other", document_id="doc1", text="...")
    c_rel = Chunk(id="chunk1", document_id="doc1", text="...")
    delayed_results = [SearchResult(chunk=c_irrel, score=0.9), SearchResult(chunk=c_rel, score=0.8)]
    assert hit_at_k(delayed_results, sample_case, k=1) == 0.0
    assert hit_at_k(delayed_results, sample_case, k=2) == 1.0

    # No relevant items
    assert hit_at_k([SearchResult(chunk=c_irrel, score=0.9)], sample_case) == 0.0


def test_hit_at_k_with_grades():
    case = BenchmarkCase(
        query="graded test",
        relevant_chunk_ids=["docA", "docB"],
        relevance_grades={"docA": 2, "docB": 1},
    )
    cA = Chunk(id="docA", document_id="d1", text="...")
    cB = Chunk(id="docB", document_id="d1", text="...")

    # min_grade=2: docB (grade 1) does not qualify, docA (grade 2) does
    assert hit_at_k([SearchResult(chunk=cB, score=0.9)], case, min_grade=2) == 0.0
    assert hit_at_k([SearchResult(chunk=cA, score=0.9)], case, min_grade=2) == 1.0


def test_ndcg_at_k_binary(sample_results, sample_case):
    # Perfect ranking: [chunk1, chunk2, chunk3] -> nDCG = 1.0
    c1 = Chunk(id="chunk1", document_id="doc1", text="...")
    c2 = Chunk(id="chunk2", document_id="doc1", text="...")
    c3 = Chunk(id="chunk3", document_id="doc1", text="...")
    perfect_results = [
        SearchResult(chunk=c1, score=0.9),
        SearchResult(chunk=c2, score=0.8),
        SearchResult(chunk=c3, score=0.7),
    ]
    assert abs(ndcg_at_k(perfect_results, sample_case) - 1.0) < 1e-6

    # sample_results: [chunk1 (rel), chunk3 (irrel), chunk2 (rel)]
    # DCG@3 = 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5 = 1.5
    # IDCG@3 = 1/log2(2) + 1/log2(3) = 1.0 + 1/1.5849625 = 1.63092975
    # nDCG@3 = 1.5 / 1.63092975 = 0.9197208
    expected_ndcg = 1.5 / (1.0 + 1.0 / math.log2(3))
    actual_ndcg = ndcg_at_k(sample_results, sample_case, k=3)
    assert abs(actual_ndcg - expected_ndcg) < 1e-6

    # Empty results -> 0.0
    assert ndcg_at_k([], sample_case) == 0.0

    # No relevant ground truth -> 0.0
    empty_case = BenchmarkCase(query="none", relevant_chunk_ids=[])
    assert ndcg_at_k(sample_results, empty_case) == 0.0


def test_ndcg_at_k_graded():
    case = BenchmarkCase(
        query="multi grade",
        relevant_chunk_ids=["doc1", "doc2", "doc3"],
        relevance_grades={"doc1": 3, "doc2": 2, "doc3": 1},
    )
    c1 = Chunk(id="doc1", document_id="d", text="...")
    c2 = Chunk(id="doc2", document_id="d", text="...")
    c3 = Chunk(id="doc3", document_id="d", text="...")

    # Perfect ranking: [doc1 (3), doc2 (2), doc3 (1)] -> nDCG = 1.0
    perfect = [SearchResult(chunk=c1, score=3), SearchResult(chunk=c2, score=2), SearchResult(chunk=c3, score=1)]
    assert abs(ndcg_at_k(perfect, case) - 1.0) < 1e-6

    # Inverted ranking: [doc2 (2), doc1 (3), doc3 (1)]
    # DCG@3 = (2^2 - 1)/log2(2) + (2^3 - 1)/log2(3) + (2^1 - 1)/log2(4)
    #       = 3/1 + 7/log2(3) + 1/2
    # IDCG@3 = (2^3 - 1)/log2(2) + (2^2 - 1)/log2(3) + (2^1 - 1)/log2(4)
    #        = 7/1 + 3/log2(3) + 1/2
    dcg = 3.0 / math.log2(2) + 7.0 / math.log2(3) + 1.0 / math.log2(4)
    idcg = 7.0 / math.log2(2) + 3.0 / math.log2(3) + 1.0 / math.log2(4)
    expected_ndcg = dcg / idcg

    inverted = [SearchResult(chunk=c2, score=3), SearchResult(chunk=c1, score=2), SearchResult(chunk=c3, score=1)]
    assert abs(ndcg_at_k(inverted, case, k=3) - expected_ndcg) < 1e-6


def test_average_precision_at_k(sample_results, sample_case):
    # sample_results: [chunk1 (rel), chunk3 (irrel), chunk2 (rel)], 2 relevant total in ground truth
    # k=1: rank 1 chunk1 (rel) -> P@1 = 1/1 -> AP@1 = 1.0 / min(2, 1) = 1.0
    assert average_precision_at_k(sample_results, sample_case, k=1) == 1.0

    # k=2: rank 1 chunk1 (rel) -> P@1 = 1/1; rank 2 chunk3 (irrel) -> AP@2 = 1.0 / min(2, 2) = 0.5
    assert average_precision_at_k(sample_results, sample_case, k=2) == 0.5

    # k=3: rank 1 (P@1=1), rank 2 (irrel), rank 3 (P@3=2/3) -> AP@3 = (1.0 + 2/3) / min(2, 3) = (5/3) / 2 = 5/6
    assert abs(average_precision_at_k(sample_results, sample_case, k=3) - (5 / 6)) < 1e-6

    # Empty retrieved -> 0.0
    assert average_precision_at_k([], sample_case) == 0.0

    # No relevant ground truth -> 0.0
    empty_case = BenchmarkCase(query="empty", relevant_chunk_ids=[])
    assert average_precision_at_k(sample_results, empty_case) == 0.0

