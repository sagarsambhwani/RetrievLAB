from retrievlab.models import SearchResult
from retrievlab.evaluation.benchmark import BenchmarkCase

def recall(retrieved_results: list[SearchResult], expected_results: BenchmarkCase) -> float:
    """
    Calculate the recall metric for a set of retrieved results against expected results.

    Args:
        retrieved_results (list[SearchResult]): The list of retrieved search results.
        expected_results (BenchmarkCase): The benchmark case containing the expected relevant chunk IDs.

    Returns:
        float: The recall value, which is the proportion of relevant items that were retrieved.
    """
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
    recall_value = relevant_retrieved_ids/total_relevant

    return recall_value

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