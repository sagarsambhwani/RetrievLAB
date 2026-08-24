"""Unit tests for HybridRetriever."""

import pytest

from retrievlab.evaluation.benchmark import Benchmark, BenchmarkCase
from retrievlab.evaluation.evaluate import evaluate_retriever
from retrievlab.models import Chunk, SearchResult
from retrievlab.ranking.fusion import ReciprocalRankFusion
from retrievlab.retrieval.hybrid import HybridRetriever
from retrievlab.retrieval.interface import Retriever


class MockRetriever(Retriever):
    """Mock retriever returning predetermined rankings for test queries."""

    def __init__(self, responses: dict[str, list[SearchResult]]) -> None:
        self.responses = responses
        self.last_query_top_k: int | None = None

    def retrieve(self, query: str, top_k: int, chunks: list[Chunk]) -> list[SearchResult]:
        self.last_query_top_k = top_k
        results = self.responses.get(query, [])
        return results[:top_k]


def make_result(chunk_id: str, score: float = 1.0) -> SearchResult:
    return SearchResult(
        chunk=Chunk(id=chunk_id, document_id="doc1", text=f"text for {chunk_id}"),
        score=score,
    )


def test_interface_compliance() -> None:
    """Verify HybridRetriever implements the Retriever interface."""
    retriever1 = MockRetriever({})
    hybrid = HybridRetriever(retrievers=[retriever1])
    assert isinstance(hybrid, Retriever)


def test_basic_hybrid_retrieval() -> None:
    """Test 2-way hybrid retrieval combining two retrievers with standard RRF (k=60)."""
    c_a = make_result("chunk_a", score=10.0)
    c_b = make_result("chunk_b", score=8.0)
    c_c = make_result("chunk_c", score=5.0)

    # Retriever 1 ranks: chunk_a (rank 1), chunk_b (rank 2)
    # Retriever 2 ranks: chunk_b (rank 1), chunk_c (rank 2)
    r1 = MockRetriever({"python": [c_a, c_b]})
    r2 = MockRetriever({"python": [c_b, c_c]})

    chunks = [c_a.chunk, c_b.chunk, c_c.chunk]
    hybrid = HybridRetriever(retrievers=[r1, r2])

    results = hybrid.retrieve(query="python", top_k=3, chunks=chunks)

    assert len(results) == 3
    # chunk_b has rank 2 in r1 and rank 1 in r2 -> 1/62 + 1/61 (~0.0325)
    # chunk_a has rank 1 in r1 -> 1/61 (~0.01639)
    # chunk_c has rank 2 in r2 -> 1/62 (~0.01613)
    assert [r.chunk.id for r in results] == ["chunk_b", "chunk_a", "chunk_c"]
    assert results[0].score == pytest.approx((1.0 / 62.0) + (1.0 / 61.0))
    assert results[1].score == pytest.approx(1.0 / 61.0)
    assert results[2].score == pytest.approx(1.0 / 62.0)


def test_custom_fusion_strategy() -> None:
    """Test HybridRetriever with custom ReciprocalRankFusion parameter (e.g. k=20)."""
    c1 = make_result("c1")
    c2 = make_result("c2")

    r1 = MockRetriever({"q": [c1, c2]})
    r2 = MockRetriever({"q": [c2, c1]})

    custom_rrf = ReciprocalRankFusion(k=20)
    hybrid = HybridRetriever(retrievers=[r1, r2], fusion_strategy=custom_rrf)

    results = hybrid.retrieve(query="q", top_k=2, chunks=[c1.chunk, c2.chunk])
    assert len(results) == 2
    # Both c1 and c2 score 1/21 + 1/22
    assert results[0].score == pytest.approx((1.0 / 21.0) + (1.0 / 22.0))
    assert results[1].score == pytest.approx((1.0 / 21.0) + (1.0 / 22.0))


def test_weighted_hybrid_retrieval() -> None:
    """Test weighted hybrid retrieval giving higher weight to one retriever."""
    c_lex = make_result("chunk_lex")
    c_dense = make_result("chunk_dense")

    r1 = MockRetriever({"search": [c_lex]})
    r2 = MockRetriever({"search": [c_dense]})

    # Weights: [2.0, 1.0]
    hybrid = HybridRetriever(retrievers=[r1, r2], weights=[2.0, 1.0])
    results = hybrid.retrieve(query="search", top_k=2, chunks=[c_lex.chunk, c_dense.chunk])

    assert len(results) == 2
    assert results[0].chunk.id == "chunk_lex"
    assert results[0].score == pytest.approx(2.0 / 61.0)
    assert results[1].chunk.id == "chunk_dense"
    assert results[1].score == pytest.approx(1.0 / 61.0)


def test_candidate_k_depth_forwarding() -> None:
    """Test that candidate_k properly sets candidate pool depth in sub-retriever retrieve calls."""
    c1 = make_result("c1")
    r1 = MockRetriever({"query": [c1]})
    r2 = MockRetriever({"query": [c1]})

    hybrid = HybridRetriever(retrievers=[r1, r2], candidate_k=50)
    hybrid.retrieve(query="query", top_k=5, chunks=[c1.chunk])

    assert r1.last_query_top_k == 50
    assert r2.last_query_top_k == 50

    # Test default fallback when candidate_k is None (max(top_k, 20))
    hybrid_default = HybridRetriever(retrievers=[r1, r2])
    hybrid_default.retrieve(query="query", top_k=5, chunks=[c1.chunk])
    assert r1.last_query_top_k == 20

    hybrid_default.retrieve(query="query", top_k=30, chunks=[c1.chunk])
    assert r1.last_query_top_k == 30


def test_three_way_retrievers() -> None:
    """Test hybrid retrieval with 3 distinct sub-retrievers."""
    c1 = make_result("c1")
    c2 = make_result("c2")
    c3 = make_result("c3")

    r1 = MockRetriever({"test": [c1, c2]})
    r2 = MockRetriever({"test": [c2, c3]})
    r3 = MockRetriever({"test": [c3, c1]})

    hybrid = HybridRetriever(retrievers=[r1, r2, r3])
    results = hybrid.retrieve(query="test", top_k=3, chunks=[c1.chunk, c2.chunk, c3.chunk])

    assert len(results) == 3
    # All 3 chunks have one rank 1 and one rank 2 -> equal scores
    assert results[0].score == pytest.approx((1.0 / 61.0) + (1.0 / 62.0))


def test_empty_chunks_and_queries() -> None:
    """Test edge cases with empty candidate chunks."""
    r1 = MockRetriever({})
    hybrid = HybridRetriever(retrievers=[r1])

    results = hybrid.retrieve(query="anything", top_k=5, chunks=[])
    assert results == []


def test_validation_errors() -> None:
    """Test validation errors for improper initialization and retrieval calls."""
    r1 = MockRetriever({})

    # Empty retrievers
    with pytest.raises(ValueError, match="retrievers sequence must not be empty"):
        HybridRetriever(retrievers=[])

    # Weights count mismatch
    with pytest.raises(ValueError, match="Number of weights .* does not match"):
        HybridRetriever(retrievers=[r1], weights=[1.0, 2.0])

    # Invalid candidate_k <= 0
    with pytest.raises(ValueError, match="candidate_k must be a positive integer"):
        HybridRetriever(retrievers=[r1], candidate_k=0)
    with pytest.raises(ValueError, match="candidate_k must be a positive integer"):
        HybridRetriever(retrievers=[r1], candidate_k=-5)

    # Invalid top_k in retrieve
    hybrid = HybridRetriever(retrievers=[r1])
    c1 = make_result("c1")
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        hybrid.retrieve(query="q", top_k=0, chunks=[c1.chunk])
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        hybrid.retrieve(query="q", top_k=-1, chunks=[c1.chunk])


def test_compatibility_with_evaluate_retriever() -> None:
    """Test that HybridRetriever seamlessly evaluates with evaluate_retriever."""
    c1 = make_result("chunk1")
    c2 = make_result("chunk2")
    chunks = [c1.chunk, c2.chunk]

    r1 = MockRetriever({"q1": [c1, c2]})
    r2 = MockRetriever({"q1": [c1, c2]})

    hybrid = HybridRetriever(retrievers=[r1, r2])
    benchmark = Benchmark(cases=[BenchmarkCase(query="q1", relevant_chunk_ids=["chunk1"])])

    eval_res = evaluate_retriever(
        retriever=hybrid,
        benchmark=benchmark,
        chunks=chunks,
        k=5,
        retriever_name="Hybrid(RRF)",
    )

    assert eval_res.retriever_name == "Hybrid(RRF)"
    assert eval_res.recall_at_k == 1.0
    assert eval_res.mrr == 1.0
    assert eval_res.num_cases == 1
