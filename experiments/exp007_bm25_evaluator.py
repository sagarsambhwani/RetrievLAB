"""
Experiment: 006

Question:
How does the BM25Retriever perform on the simple benchmark dataset in terms of Recall@5 and Mean Reciprocal Rank (MRR)?
Does it handle unknown terms and empty queries gracefully as expected?

Expected Result:
- The evaluator should compute Recall@5 and MRR for each benchmark query.
- Metrics should match expected intuition, and unknown terms should return zero score.
"""

from pathlib import Path

from retrievlab.ingestion.loader import DocumentLoader
from retrievlab.chunking.markdown import MarkdownChunker
from retrievlab.retrieval.bm25 import BM25Retriever
from retrievlab.evaluation.metrics import recall_at_k, reciprocal_rank
from retrievlab.evaluation.benchmark import load_benchmark


def run_experiment():
    print("Starting Experiment 006: BM25 Retriever Evaluation\n")

    # Initialize components
    loader = DocumentLoader()
    chunker = MarkdownChunker()
    retriever = BM25Retriever()

    # Load and chunk documents
    data_path = Path("data/raw")
    documents = loader.load(data_path)
    print(f"Loaded {len(documents)} documents from {data_path}")

    chunks = []
    for doc in documents:
        chunks.extend(chunker.chunk(doc))
    print(f"Generated {len(chunks)} chunks from documents.")

    # Index chunks in BM25 Retriever
    retriever.index(chunks)
    print(f"Indexed corpus. Average chunk length: {retriever.average_chunk_length:.2f} tokens.")

    # Load benchmark dataset
    benchmark = load_benchmark("data/benchmarks/simple.json")
    print(f"Loaded {len(benchmark.cases)} benchmark cases from data/benchmarks/simple.json\n")

    recalls = []
    rrs = []

    print(f"{'Query':<45} | {'Expected':<25} | {'Top-1 Retrieved (Score)':<25} | {'Recall@5':<8} | {'MRR':<8}")
    print("-" * 118)

    for case in benchmark.cases:
        results = retriever.retrieve(
            query=case.query,
            top_k=5,
            chunks=chunks,
        )
        
        rec_score = recall_at_k(retrieved_results=results, expected_results=case, k=5)
        rr_score = reciprocal_rank(retrieved_results=results, expected_results=case)
        
        recalls.append(rec_score)
        rrs.append(rr_score)

        # Format details for printing
        expected_str = ",".join(case.relevant_chunk_ids)
        if len(expected_str) > 23:
            expected_str = expected_str[:20] + "..."

        if results:
            top_1_str = f"{results[0].chunk.id} ({results[0].score:.2f})"
        else:
            top_1_str = "None"
        
        query_truncated = case.query
        if len(query_truncated) > 43:
            query_truncated = query_truncated[:40] + "..."

        print(f"{query_truncated:<45} | {expected_str:<25} | {top_1_str:<25} | {rec_score:<8.2f} | {rr_score:<8.2f}")

    # Compute overall metrics
    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    mean_mrr = sum(rrs) / len(rrs) if rrs else 0.0

    print("-" * 118)
    print(f"Mean Recall@5: {mean_recall:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {mean_mrr:.4f}\n")

    # Test unknown terms
    print("Testing Edge Case: Unknown Query Terms")
    unknown_query = "supercalifragilisticexpialidocious query"
    unknown_results = retriever.retrieve(unknown_query, top_k=3, chunks=chunks)
    print(f"Query: '{unknown_query}'")
    print(f"Results count: {len(unknown_results)}")
    for i, res in enumerate(unknown_results, start=1):
        print(f"  {i}. Chunk ID: {res.chunk.id} | Score: {res.score:.4f}")

    # Test empty query
    print("\nTesting Edge Case: Empty Query")
    empty_results = retriever.retrieve("", top_k=3, chunks=chunks)
    print(f"Query: ''")
    print(f"Results count: {len(empty_results)}")
    for i, res in enumerate(empty_results, start=1):
        print(f"  {i}. Chunk ID: {res.chunk.id} | Score: {res.score:.4f}")

    
    # test for query: What are the key features of FastAPI? 
    query_tokens = [
        "what",
        "are",
        "the",
        "key",
        "features",
        "of",
        "fastapi"
    ]
    print("\nScore of each word across fastapi.md:1 and docker.md:1:")
    print(f"{'Chunk ID':<20} | {'Token':<10} | {'Score':<6}")
    print("-" * 44)
    for token in query_tokens:
        scores = [
            (retriever.bm25_score([token], chunk), chunk)
            for chunk in chunks
        ]
        for score, chunk in scores:
            if chunk.id in ["fastapi.md:1", "docker.md:1"]:
                print(f"{chunk.id:<20} | {token:<10} | {score:.3f}")


if __name__ == "__main__":
    run_experiment()
