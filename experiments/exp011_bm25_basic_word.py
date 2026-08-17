"""
Experiment 011

Question:
How does BM25 perform with explicit BasicWordTokenizer dependency injection on the raw document corpus
and the simple2.json benchmark dataset?

Expected Result:
- BasicWordTokenizer tokenizes text into lowercase word tokens using standard word boundaries.
- BM25Retriever successfully computes index statistics (average chunk length, vocabulary size, IDF).
- Computes baseline Recall@5, Precision@5, and MRR across all 22 benchmark queries in simple2.json.
"""

from pathlib import Path

from retrievlab.chunking.markdown import MarkdownChunker
from retrievlab.evaluation import evaluate_retriever, load_benchmark
from retrievlab.ingestion.loader import DocumentLoader
from retrievlab.preprocessing import BasicWordTokenizer
from retrievlab.retrieval.bm25 import BM25Retriever


def run_experiment() -> None:
    print("=" * 80)
    print("Experiment 011: BM25 with BasicWordTokenizer (Evaluated on simple2.json)")
    print("=" * 80)

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

    # 5. Evaluate against simple2.json benchmark
    benchmark_path = "data/benchmarks/simple2.json"
    benchmark = load_benchmark(benchmark_path)
    print(f"\n4. Benchmark Evaluation ({benchmark_path}):")
    print(f"   Loaded {len(benchmark.cases)} benchmark query cases.")

    result = evaluate_retriever(
        retriever=retriever,
        benchmark=benchmark,
        chunks=chunks,
        k=5,
        retriever_name="BM25 (BasicWordTokenizer)",
    )

    print(f"\n   {'Retriever':<30} | {'Recall@5':<10} | {'Precision@5':<14} | {'MRR':<8}")
    print(f"   {'-' * 68}")
    print(f"   {result.retriever_name:<30} | {result.recall_at_k:<10.4f} | {result.precision_at_k:<14.4f} | {result.mrr:<8.4f}")


if __name__ == "__main__":
    run_experiment()
