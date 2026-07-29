import pytest
from retrievlab.models import SearchResult, Chunk
from retrievlab.evaluation.benchmark import BenchmarkCase
from retrievlab.evaluation.metrics import recall_at_k, precision_at_k, reciprocal_rank

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
