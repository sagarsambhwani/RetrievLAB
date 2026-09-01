"""Unit tests for automated failure analysis and diagnostic tooling."""

from retrievlab.models import Chunk, SearchResult
from retrievlab.retrieval.interface import Retriever
from retrievlab.evaluation.benchmark import Benchmark, BenchmarkCase
from retrievlab.evaluation.diagnostics import (
    QueryOutcomeCategory,
    categorize_query_outcome,
    analyze_query_outcome,
    analyze_hybrid_failures,
    HybridFailureAnalysisReport,
)


class MockRetriever(Retriever):
    """Mock retriever returning predetermined chunk IDs."""

    def __init__(self, mapping: dict[str, list[str]]):
        self.mapping = mapping

    def retrieve(self, query: str, top_k: int = 5, chunks: list[Chunk] | None = None) -> list[SearchResult]:
        chunk_ids = self.mapping.get(query, [])[:top_k]
        results = []
        for idx, cid in enumerate(chunk_ids):
            chunk = Chunk(id=cid, document_id="doc1", text=f"Text for {cid}", metadata={})
            score = 1.0 / (idx + 1)
            results.append(SearchResult(chunk=chunk, score=score))
        return results


def test_categorize_query_outcome():
    """Verify outcome categorization across all permutations of recall."""
    # 1. Joint Hit
    assert categorize_query_outcome(1.0, 1.0, 1.0) == QueryOutcomeCategory.JOINT_HIT

    # 2. Dense Win Recovered
    assert categorize_query_outcome(0.0, 1.0, 1.0) == QueryOutcomeCategory.DENSE_WIN_HYBRID_RECOVERED

    # 3. BM25 Win Recovered
    assert categorize_query_outcome(1.0, 0.0, 1.0) == QueryOutcomeCategory.BM25_WIN_HYBRID_RECOVERED

    # 4. Dense Win Missed by Hybrid
    assert categorize_query_outcome(0.0, 1.0, 0.0) == QueryOutcomeCategory.DENSE_WIN_HYBRID_MISSED

    # 5. BM25 Win Missed by Hybrid
    assert categorize_query_outcome(1.0, 0.0, 0.0) == QueryOutcomeCategory.BM25_WIN_HYBRID_MISSED

    # 6. Hybrid Degradation
    assert categorize_query_outcome(1.0, 1.0, 0.0) == QueryOutcomeCategory.HYBRID_DEGRADATION

    # 7. Joint Miss
    assert categorize_query_outcome(0.0, 0.0, 0.0) == QueryOutcomeCategory.JOINT_MISS


def test_analyze_query_outcome():
    """Test single query diagnostic extraction and rank shift."""
    case = BenchmarkCase(query="test query", relevant_chunk_ids=["chunk_target"])

    chunk_target = Chunk(id="chunk_target", document_id="doc1", text="target text", metadata={})
    chunk_other = Chunk(id="chunk_other", document_id="doc1", text="other text", metadata={})

    # BM25 missed target (returned chunk_other)
    bm25_res = [SearchResult(chunk=chunk_other, score=5.0)]
    # Dense hit target at rank 1
    dense_res = [SearchResult(chunk=chunk_target, score=0.9)]
    # Hybrid hit target at rank 1
    hybrid_res = [SearchResult(chunk=chunk_target, score=0.03)]

    diag = analyze_query_outcome(
        case=case,
        bm25_results=bm25_res,
        dense_results=dense_res,
        hybrid_results=hybrid_res,
        k=5,
        query_index=1,
    )

    assert diag.query_index == 1
    assert diag.query == "test query"
    assert diag.bm25_recall == 0.0
    assert diag.bm25_rank is None
    assert diag.dense_recall == 1.0
    assert diag.dense_rank == 1
    assert diag.hybrid_recall == 1.0
    assert diag.hybrid_rank == 1
    assert diag.category == QueryOutcomeCategory.DENSE_WIN_HYBRID_RECOVERED
    assert diag.is_recovered is True
    assert diag.is_degradation is False


def test_analyze_hybrid_failures_end_to_end():
    """Test end-to-end benchmark failure analysis using mock retrievers."""
    cases = [
        BenchmarkCase(query="joint query", relevant_chunk_ids=["c1"]),
        BenchmarkCase(query="dense query", relevant_chunk_ids=["c2"]),
        BenchmarkCase(query="miss query", relevant_chunk_ids=["c3"]),
    ]
    benchmark = Benchmark(cases=cases)

    # BM25 finds c1, misses c2, misses c3
    bm25 = MockRetriever({"joint query": ["c1"], "dense query": ["cx"], "miss query": ["cx"]})
    # Dense finds c1, finds c2, misses c3
    dense = MockRetriever({"joint query": ["c1"], "dense query": ["c2"], "miss query": ["cy"]})
    # Hybrid finds c1, recovers c2, misses c3
    hybrid = MockRetriever({"joint query": ["c1"], "dense query": ["c2"], "miss query": ["cz"]})

    chunks = [Chunk(id=f"c{i}", document_id="doc1", text="text", metadata={}) for i in range(1, 5)]

    report = analyze_hybrid_failures(
        bm25_retriever=bm25,
        dense_retriever=dense,
        hybrid_retriever=hybrid,
        benchmark=benchmark,
        chunks=chunks,
        k=5,
    )

    assert isinstance(report, HybridFailureAnalysisReport)
    assert report.total_queries == 3
    assert report.category_counts[QueryOutcomeCategory.JOINT_HIT.value] == 1
    assert report.category_counts[QueryOutcomeCategory.DENSE_WIN_HYBRID_RECOVERED.value] == 1
    assert report.category_counts[QueryOutcomeCategory.JOINT_MISS.value] == 1
    assert len(report.recoveries) == 1
    assert report.recoveries[0].query_index == 2

    md = report.to_markdown()
    assert "# Hybrid Failure Analysis & Recovery Diagnosis" in md
    assert "Dense Wins Recovered by Hybrid:** 1" in md
    assert "[RECOVERED]" in md
