"""
Unit tests for CascadedReRanker (coarse-to-fine two-stage re-ranking).
"""

import pytest
from retrievlab.models import Chunk, SearchResult
from retrievlab.ranking.cascade import CascadedReRanker
from retrievlab.ranking.interface import ReRanker
from retrievlab.selection.candidate import Candidate, CandidatePool


class MockFilterRanker(ReRanker):
    """Simple mock ranker that assigns scores based on chunk id numeric suffix."""

    def rerank(self, query: str, candidates, top_k: int | None = None) -> list[SearchResult]:
        chunks = []
        if isinstance(candidates, CandidatePool):
            chunks = candidates.get_chunks()
        else:
            for item in candidates:
                if isinstance(item, (Chunk, Candidate, SearchResult)):
                    chunks.append(item.chunk if hasattr(item, "chunk") else item)

        results = [
            SearchResult(chunk=c, score=float(c.id.replace("chunk_", "")))
            for c in chunks
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        if top_k is not None and top_k > 0:
            return results[:top_k]
        return results


class MockRefinerRanker(ReRanker):
    """Simple mock ranker that inverts scores or prioritizes specific query terms."""

    def rerank(self, query: str, candidates, top_k: int | None = None) -> list[SearchResult]:
        chunks = []
        if isinstance(candidates, CandidatePool):
            chunks = candidates.get_chunks()
        else:
            for item in candidates:
                if isinstance(item, (Chunk, Candidate, SearchResult)):
                    chunks.append(item.chunk if hasattr(item, "chunk") else item)

        results = [
            SearchResult(chunk=c, score=100.0 / (float(c.id.replace("chunk_", "")) + 1.0))
            for c in chunks
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        if top_k is not None and top_k > 0:
            return results[:top_k]
        return results


@pytest.fixture
def sample_candidate_pool() -> CandidatePool:
    cands = [
        Candidate(
            chunk=Chunk(id=f"chunk_{i}", document_id=f"doc_{i}", text=f"Text for chunk {i}"),
            retriever_scores={"bm25": 1.0},
            retriever_ranks={"bm25": i + 1},
            sources=["bm25"],
        )
        for i in range(20)
    ]
    return CandidatePool(query="test query", candidates=cands)


def test_cascaded_reranker_basic(sample_candidate_pool):
    filter_ranker = MockFilterRanker()
    refiner_ranker = MockRefinerRanker()

    cascade = CascadedReRanker(
        filter_ranker=filter_ranker,
        refiner_ranker=refiner_ranker,
        intermediate_k=5,
    )

    # 20 items in pool -> filter keeps top 5 (chunk_19, 18, 17, 16, 15)
    # refiner inverts order among those 5 -> chunk_15 has lowest id -> highest refiner score
    results = cascade.rerank("test query", sample_candidate_pool, top_k=3)

    assert len(results) == 3
    assert results[0].chunk.id == "chunk_15"
    assert results[1].chunk.id == "chunk_16"
    assert results[2].chunk.id == "chunk_17"


def test_cascaded_reranker_empty():
    filter_ranker = MockFilterRanker()
    refiner_ranker = MockRefinerRanker()
    cascade = CascadedReRanker(filter_ranker, refiner_ranker, intermediate_k=5)

    results = cascade.rerank("query", [], top_k=5)
    assert results == []


def test_cascaded_reranker_invalid_intermediate_k():
    filter_ranker = MockFilterRanker()
    refiner_ranker = MockRefinerRanker()

    with pytest.raises(ValueError, match="intermediate_k must be greater than 0"):
        CascadedReRanker(filter_ranker, refiner_ranker, intermediate_k=0)
