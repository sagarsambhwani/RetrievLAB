from retrievlab.models import Chunk, SearchResult
from retrievlab.retrieval.interface import Retriever
from retrievlab.evaluation.benchmark import Benchmark, BenchmarkCase
from retrievlab.evaluation.evaluate import evaluate_retriever
from retrievlab.evaluation.reports import RetrieverEvaluationResult


class DummyRetriever(Retriever):
    def retrieve(self, query: str, top_k: int, chunks: list[Chunk]) -> list[SearchResult]:
        # Return first top_k chunks
        return [SearchResult(chunk=c, score=1.0) for c in chunks[:top_k]]


def test_evaluate_retriever():
    c1 = Chunk(id="c1", document_id="doc1", text="text 1")
    c2 = Chunk(id="c2", document_id="doc1", text="text 2")
    c3 = Chunk(id="c3", document_id="doc1", text="text 3")
    chunks = [c1, c2, c3]

    case1 = BenchmarkCase(query="q1", relevant_chunk_ids=["c1"])
    case2 = BenchmarkCase(query="q2", relevant_chunk_ids=["c2"])
    benchmark = Benchmark(cases=[case1, case2])

    retriever = DummyRetriever()

    # k=1: for q1 -> retrieves [c1] -> recall=1.0, prec=1.0, mrr=1.0
    #      for q2 -> retrieves [c1] -> recall=0.0, prec=0.0, mrr=0.0
    result = evaluate_retriever(retriever, benchmark, chunks, k=1, retriever_name="Dummy")

    assert isinstance(result, RetrieverEvaluationResult)
    assert result.retriever_name == "Dummy"
    assert result.k == 1
    assert result.num_cases == 2
    assert result.recall_at_k == 0.5
    assert result.precision_at_k == 0.5
    assert result.mrr == 0.5


def test_evaluate_retriever_empty_benchmark():
    retriever = DummyRetriever()
    benchmark = Benchmark(cases=[])
    result = evaluate_retriever(retriever, benchmark, chunks=[], k=5)
    assert result.num_cases == 0
    assert result.recall_at_k == 0.0
    assert result.mrr == 0.0
