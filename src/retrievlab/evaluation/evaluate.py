from retrievlab.models import Chunk
from retrievlab.retrieval.interface import Retriever
from retrievlab.evaluation.benchmark import Benchmark
from retrievlab.evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    reciprocal_rank,
    hit_at_k,
    ndcg_at_k,
    average_precision_at_k,
)
from retrievlab.evaluation.reports import RetrieverEvaluationResult


def evaluate_retriever(
    retriever: Retriever,
    benchmark: Benchmark,
    chunks: list[Chunk],
    k: int = 5,
    retriever_name: str | None = None,
) -> RetrieverEvaluationResult:
    """Evaluate a retriever over a benchmark dataset.

    Args:
        retriever: An instance of a Retriever implementation.
        benchmark: The Benchmark containing query test cases.
        chunks: Candidate pool of chunks to retrieve from.
        k: Top-K cutoff value for evaluation. Defaults to 5.
        retriever_name: Optional custom display name for the retriever.
            Defaults to the retriever's class name.

    Returns:
        RetrieverEvaluationResult containing mean Recall@K, Precision@K, MRR, nDCG@K, MAP@K, and Hit@K.
    """
    if retriever_name is None:
        retriever_name = retriever.__class__.__name__

    if not benchmark.cases:
        return RetrieverEvaluationResult(
            retriever_name=retriever_name,
            recall_at_k=0.0,
            precision_at_k=0.0,
            mrr=0.0,
            ndcg_at_k=0.0,
            map_at_k=0.0,
            hit_at_k=0.0,
            k=k,
            num_cases=0,
        )

    recalls = []
    precisions = []
    rrs = []
    ndcgs = []
    maps = []
    hits = []

    for case in benchmark.cases:
        results = retriever.retrieve(query=case.query, top_k=k, chunks=chunks)

        recalls.append(recall_at_k(retrieved_results=results, expected_results=case, k=k))
        precisions.append(precision_at_k(retrieved_results=results, expected_results=case, k=k))
        rrs.append(reciprocal_rank(retrieved_results=results, expected_results=case))
        ndcgs.append(ndcg_at_k(retrieved_results=results, expected_results=case, k=k))
        maps.append(average_precision_at_k(retrieved_results=results, expected_results=case, k=k))
        hits.append(hit_at_k(retrieved_results=results, expected_results=case, k=k))

    num_cases = len(benchmark.cases)

    return RetrieverEvaluationResult(
        retriever_name=retriever_name,
        recall_at_k=sum(recalls) / num_cases,
        precision_at_k=sum(precisions) / num_cases,
        mrr=sum(rrs) / num_cases,
        ndcg_at_k=sum(ndcgs) / num_cases,
        map_at_k=sum(maps) / num_cases,
        hit_at_k=sum(hits) / num_cases,
        k=k,
        num_cases=num_cases,
    )
