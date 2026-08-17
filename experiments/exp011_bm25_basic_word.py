"""
Experiment 011

Question:
How does BM25 perform with explicit BasicWordTokenizer dependency injection on the raw document corpus
and each query in the simple2.json benchmark dataset?

Expected Result:
- BasicWordTokenizer tokenizes text into lowercase word tokens using standard word boundaries.
- BM25Retriever successfully computes index statistics (average chunk length, vocabulary size, IDF).
- Prints a detailed per-query evaluation table showing Recall@5, Precision@5, and MRR for all 22 benchmark cases.
"""

from pathlib import Path

from retrievlab.chunking.markdown import MarkdownChunker
from retrievlab.evaluation import (
    load_benchmark,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from retrievlab.ingestion.loader import DocumentLoader
from retrievlab.preprocessing import BasicWordTokenizer
from retrievlab.retrieval.bm25 import BM25Retriever


def run_experiment() -> None:
    print("=" * 118)
    print("Experiment 011: BM25 with BasicWordTokenizer (Evaluated on simple2.json)")
    print("=" * 118)

    # 1. Load and chunk documents
    loader = DocumentLoader()
    chunker = MarkdownChunker()
    raw_path = Path("data/raw")
    documents = loader.load(raw_path)

    chunks = []
    for doc in documents:
        chunks.extend(chunker.chunk(doc))

    print(f"\n1. Corpus Loading:")
    print(f"   Loaded {len(documents)} document(s) producing {len(chunks)} chunk(s).")

    # 2. Initialize BasicWordTokenizer and BM25Retriever
    tokenizer = BasicWordTokenizer(lower=True)
    retriever = BM25Retriever(tokenizer=tokenizer)

    # 3. Index corpus
    retriever.index(chunks)
    vocab_size = len(retriever.term_frequencies)
    print(f"\n2. Index Statistics:")
    print(f"   Tokenizer: BasicWordTokenizer(lower=True)")
    print(f"   Vocabulary Size: {vocab_size} unique terms")
    print(f"   Average Chunk Length: {retriever.average_chunk_length:.2f} tokens")

    # 4. Inspect sample token DF and IDF values
    sample_tokens = ["fastapi", "docker", "python", "framework", "containers", "the"]
    print(f"\n3. Sample Token Statistics:")
    print(f"   {'Token':<15} | {'Doc Frequency (DF)':<20} | {'IDF':<8}")
    print(f"   {'-' * 48}")
    for token in sample_tokens:
        df = len(retriever.term_frequencies.get(token, {}))
        idf = retriever._inverse_document_frequency(token)
        print(f"   {token:<15} | {df:<20} | {idf:<8.3f}")

    # 5. Per-Query Benchmark Evaluation on simple2.json
    benchmark_path = "data/benchmarks/simple2.json"
    benchmark = load_benchmark(benchmark_path)
    print(f"\n4. Per-Query Benchmark Evaluation ({benchmark_path}):")
    print(f"   Loaded {len(benchmark.cases)} benchmark cases.\n")

    print(f"{'#':<3} | {'Query':<40} | {'Expected':<20} | {'Top-1 Retrieved (Score)':<24} | {'R@5':<5} | {'P@5':<5} | {'MRR':<5}")
    print("-" * 118)

    recalls = []
    precisions = []
    rrs = []

    for i, case in enumerate(benchmark.cases, start=1):
        results = retriever.retrieve(query=case.query, top_k=5, chunks=chunks)

        rec = recall_at_k(retrieved_results=results, expected_results=case, k=5)
        prec = precision_at_k(retrieved_results=results, expected_results=case, k=5)
        rr = reciprocal_rank(retrieved_results=results, expected_results=case)

        recalls.append(rec)
        precisions.append(prec)
        rrs.append(rr)

        expected_str = ",".join(case.relevant_chunk_ids)
        if len(expected_str) > 20:
            expected_str = expected_str[:17] + "..."

        if results:
            top_1_str = f"{results[0].chunk.id} ({results[0].score:.2f})"
        else:
            top_1_str = "None"

        query_str = case.query
        if len(query_str) > 40:
            query_str = query_str[:37] + "..."

        print(f"{i:<3} | {query_str:<40} | {expected_str:<20} | {top_1_str:<24} | {rec:<5.2f} | {prec:<5.2f} | {rr:<5.2f}")

    print("-" * 118)

    # 6. Summary Aggregate Metrics
    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    mean_precision = sum(precisions) / len(precisions) if precisions else 0.0
    mean_mrr = sum(rrs) / len(rrs) if rrs else 0.0
    print(f"Aggregate Mean Recall@5: {mean_recall:.4f} | Precision@5: {mean_precision:.4f} | MRR: {mean_mrr:.4f}\n")


if __name__ == "__main__":
    run_experiment()
