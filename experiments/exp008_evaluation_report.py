"""
Experiment: 008

Question:
Does the EvaluationReport properly compare multiple retrievers (BM25 vs Dense)
and render a clear summary report table over the benchmark dataset?

Expected Result:
- Both BM25Retriever and DenseRetriever are evaluated against the benchmark dataset.
- An EvaluationReport object collects the results and outputs a Markdown summary table.
"""
from pathlib import Path

from retrievlab.ingestion.loader import DocumentLoader
from retrievlab.chunking.markdown import MarkdownChunker
from retrievlab.embeddings.embedder import Embedder
from retrievlab.embeddings.fastembed import FastEmbedClient
from retrievlab.retrieval.bm25 import BM25Retriever
from retrievlab.retrieval.dense import DenseRetriever
from retrievlab.evaluation import load_benchmark, evaluate_retriever, EvaluationReport


def run_experiment():
    print("Starting Experiment 008: Evaluation Report (BM25 vs Dense)\n")

    # Load and chunk documents
    loader = DocumentLoader()
    chunker = MarkdownChunker()

    data_path = Path("data/raw")
    documents = loader.load(data_path)

    chunks = []
    for doc in documents:
        chunks.extend(chunker.chunk(doc))

    # Load benchmark dataset
    benchmark = load_benchmark("data/benchmarks/simple.json")
    print(f"Loaded {len(documents)} documents ({len(chunks)} chunks) and {len(benchmark.cases)} benchmark cases.\n")

    report = EvaluationReport()

    # 1. Evaluate BM25 Retriever
    print("Evaluating BM25 Retriever...")
    bm25 = BM25Retriever()
    bm25.index(chunks)
    bm25_result = evaluate_retriever(bm25, benchmark, chunks, k=5, retriever_name="BM25")
    report.add_result(bm25_result)

    # 2. Evaluate Dense Retriever
    print("Evaluating Dense Retriever...")
    client = FastEmbedClient()
    embedder = Embedder(client)
    embedded_chunks = embedder.embed(chunks)
    dense = DenseRetriever(client)
    dense_result = evaluate_retriever(dense, benchmark, embedded_chunks, k=5, retriever_name="Dense")
    report.add_result(dense_result)

    # Print Evaluation Summary Report
    print("\n=== Evaluation Summary Report ===")
    print(report.to_markdown())


if __name__ == "__main__":
    run_experiment()
