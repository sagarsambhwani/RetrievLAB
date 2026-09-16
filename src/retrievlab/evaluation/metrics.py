import math
from retrievlab.models import SearchResult
from retrievlab.evaluation.benchmark import BenchmarkCase

def recall_at_k(retrieved_results: list[SearchResult], expected_results: BenchmarkCase, k: int | None = None) -> float:
    """
    Calculate the recall metric for a set of retrieved results against expected results.

    Args:
        retrieved_results (list[SearchResult]): The list of retrieved search results.
        expected_results (BenchmarkCase): The benchmark case containing the expected relevant chunk IDs.
        k (int | None, optional): Cutoff value for recall. If None, evaluates all retrieved results.

    Returns:
        float: The recall value, which is the proportion of relevant items that were retrieved.
    """
    if k is not None:
        if k <= 0:
            raise ValueError("k must be greater than 0.")
        retrieved_results = retrieved_results[:k]
    
    # Extract the IDs of the retrieved chunks 
    retrieved_ids = set(result.chunk.id for result in retrieved_results)

    # Extract the expected relevant chunk IDs from the benchmark case
    relevant_ids = set(expected_results.relevant_chunk_ids)

    # Calculate the number of relevant items that were retrieved
    relevant_retrieved_ids = len(retrieved_ids & relevant_ids)

    # Calculate the total number of relevant items
    total_relevant = len(relevant_ids)

    # Handle the case where there are no relevant items to avoid division by zero
    if total_relevant == 0:
        return 0.0

    # Calculate recall as the proportion of relevant items that were retrieved
    recall_value = relevant_retrieved_ids / total_relevant

    return recall_value

def precision_at_k(retrieved_results: list[SearchResult], expected_results: BenchmarkCase, k: int | None = None) -> float:
    """
    Calculate the Precision@K metric for a set of retrieved results against expected results.

    Args:
        retrieved_results (list[SearchResult]): The list of retrieved search results.
        expected_results (BenchmarkCase): The benchmark case containing the expected relevant chunk IDs.
        k (int | None, optional): Cutoff value for precision calculation. If None, considers all retrieved results.

    Returns:
        float: The Precision@K value, which is the proportion of retrieved items that are relevant.
    """
    if k is not None:
        if k <= 0:
            raise ValueError("k must be greater than 0.")
        retrieved_results = retrieved_results[:k]

    if not retrieved_results:
        return 0.0

    # Extract the IDs of the retrieved chunks
    retrieved_ids = set(result.chunk.id for result in retrieved_results)

    # Extract the expected relevant chunk IDs from the benchmark case
    relevant_ids = set(expected_results.relevant_chunk_ids)

    # Calculate the number of relevant items that were retrieved
    relevant_retrieved_ids = len(retrieved_ids & relevant_ids)

    denom = k if k is not None else len(retrieved_results)
    if denom == 0:
        return 0.0

    # Calculate precision at k as the proportion of retrieved items that are relevant
    precision_at_k_value = relevant_retrieved_ids / denom

    return precision_at_k_value

def reciprocal_rank(retrieved_results: list[SearchResult], expected_results: BenchmarkCase) -> float:
    """
    Calculate the Mean Reciprocal Rank (MRR) for a set of retrieved results against expected results.

    Args:
        retrieved_results (list[SearchResult]): The list of retrieved search results.
        expected_results (BenchmarkCase): The benchmark case containing the expected relevant chunk IDs.

    Returns:
        float: The MRR value, which is the average of the reciprocal ranks of the first relevant item.
    """
    # Extract the IDs of the retrieved chunks
    retrieved_ids = [result.chunk.id for result in retrieved_results]

    # Extract the expected relevant chunk IDs from the benchmark case
    relevant_ids = set(expected_results.relevant_chunk_ids)

    # Initialize reciprocal rank
    reciprocal_rank = 0.0

    # Iterate through the retrieved results to find the rank of the first relevant item
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            reciprocal_rank = 1.0 / rank
            break

    return reciprocal_rank


def hit_at_k(
    retrieved_results: list[SearchResult],
    expected_results: BenchmarkCase,
    k: int | None = None,
    min_grade: int = 1,
) -> float:
    """Calculate Hit@K (1.0 if at least one relevant chunk with grade >= min_grade is in top-K, else 0.0).

    Args:
        retrieved_results: The list of retrieved search results.
        expected_results: The benchmark case containing relevant IDs and optional relevance grades.
        k: Optional top-K cutoff. If None, considers all retrieved results.
        min_grade: Minimum relevance grade threshold to be considered relevant. Defaults to 1.

    Returns:
        float: 1.0 if at least one relevant document is found in top-K, 0.0 otherwise.
    """
    if k is not None:
        if k <= 0:
            raise ValueError("k must be greater than 0.")
        retrieved_results = retrieved_results[:k]

    for result in retrieved_results:
        if expected_results.get_grade(result.chunk.id) >= min_grade:
            return 1.0

    return 0.0


def ndcg_at_k(
    retrieved_results: list[SearchResult],
    expected_results: BenchmarkCase,
    k: int | None = None,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain (nDCG@K).

    Formulas:
        DCG@K = sum_{i=1}^K (2^{grade_i} - 1) / log2(i + 1)
        IDCG@K = sum_{i=1}^{min(K, |R|)} (2^{ideal_grade_i} - 1) / log2(i + 1)
        nDCG@K = DCG@K / IDCG@K (returns 0.0 if IDCG == 0)

    Args:
        retrieved_results: The list of retrieved search results.
        expected_results: The benchmark case containing relevant IDs and optional relevance grades.
        k: Optional top-K cutoff. If None, evaluates all retrieved results.

    Returns:
        float: The nDCG@K score in range [0.0, 1.0].
    """
    if k is not None:
        if k <= 0:
            raise ValueError("k must be greater than 0.")
        retrieved_results = retrieved_results[:k]

    # Calculate DCG@K
    dcg = 0.0
    for rank, result in enumerate(retrieved_results, start=1):
        grade = expected_results.get_grade(result.chunk.id)
        if grade > 0:
            gain = (2 ** grade) - 1
            discount = math.log2(rank + 1)
            dcg += gain / discount

    # Calculate IDCG@K (Ideal DCG)
    all_grades = sorted(expected_results.get_all_grades().values(), reverse=True)
    cutoff = k if k is not None else len(all_grades)
    ideal_grades = all_grades[:cutoff]

    idcg = 0.0
    for rank, grade in enumerate(ideal_grades, start=1):
        if grade > 0:
            gain = (2 ** grade) - 1
            discount = math.log2(rank + 1)
            idcg += gain / discount

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def average_precision_at_k(
    retrieved_results: list[SearchResult],
    expected_results: BenchmarkCase,
    k: int | None = None,
    min_grade: int = 1,
) -> float:
    """Calculate Average Precision at K (AP@K).

    Formula:
        AP@K = (1 / min(|R|, K)) * sum_{i=1}^K (Precision@i * is_relevant(item_i))

    Args:
        retrieved_results: The list of retrieved search results.
        expected_results: The benchmark case containing relevant IDs and optional relevance grades.
        k: Optional top-K cutoff. If None, evaluates all retrieved results.
        min_grade: Minimum relevance grade threshold to be considered relevant. Defaults to 1.

    Returns:
        float: Average precision score in range [0.0, 1.0].
    """
    if k is not None:
        if k <= 0:
            raise ValueError("k must be greater than 0.")
        retrieved_results = retrieved_results[:k]

    if not retrieved_results:
        return 0.0

    all_grades = expected_results.get_all_grades()
    total_relevant = sum(1 for grade in all_grades.values() if grade >= min_grade)
    if total_relevant == 0:
        return 0.0

    running_relevant_count = 0
    precision_sum = 0.0

    for rank, result in enumerate(retrieved_results, start=1):
        if expected_results.get_grade(result.chunk.id) >= min_grade:
            running_relevant_count += 1
            precision_at_rank = running_relevant_count / rank
            precision_sum += precision_at_rank

    denom = min(total_relevant, k) if k is not None else total_relevant
    return precision_sum / denom if denom > 0 else 0.0