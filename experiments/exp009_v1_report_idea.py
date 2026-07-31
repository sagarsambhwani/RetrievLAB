"""
Experiment 009: Dynamic Noteworthy Query Study (RLB-030)

Design Goal:
Dynamically isolate only "noteworthy" queries (divergent model outcomes or retriever failures)
without hardcoded manual annotations.

This is a experimental version for the RLB-030 request. It will later changed to more nuanced version.
"""
from pathlib import Path

from retrievlab.ingestion.loader import DocumentLoader
from retrievlab.chunking.markdown import MarkdownChunker
from retrievlab.embeddings.embedder import Embedder
from retrievlab.embeddings.fastembed import FastEmbedClient
from retrievlab.retrieval.bm25 import BM25Retriever
from retrievlab.retrieval.dense import DenseRetriever
from retrievlab.evaluation import (
    load_benchmark,
    evaluate_retriever,
    EvaluationReport,
    reciprocal_rank,
)


def run_experiment():
    # 1. Load and chunk documents
    loader = DocumentLoader()
    chunker = MarkdownChunker()
    documents = loader.load(Path("data/raw"))
    chunks = []
    for doc in documents:
        chunks.extend(chunker.chunk(doc))

    # 2. Initialize Retrievers
    bm25 = BM25Retriever()
    bm25.index(chunks)

    client = FastEmbedClient()
    embedder = Embedder(client)
    embedded_chunks = embedder.embed(chunks)
    dense = DenseRetriever(client)

    # 3. Load Benchmark
    benchmark = load_benchmark("data/benchmarks/simple.json")

    # 4. Display Overall Aggregate Metrics
    report = EvaluationReport()
    report.add_result(evaluate_retriever(bm25, benchmark, chunks, k=5, retriever_name="BM25"))
    report.add_result(evaluate_retriever(dense, benchmark, embedded_chunks, k=5, retriever_name="Dense"))

    print("=== Evaluation Summary Report ===")
    print(report.to_markdown())
    print()

    # 5. Dynamic Noteworthy Query Extraction
    # Rule: Keep query if models diverge (abs(delta) > 0) or if any model had zero hits (failure).
    headers = ["Query", "BM25 RR", "Dense RR", "Winner", "Delta", "Flags"]
    col_widths = [54, 9, 10, 8, 7, 24]

    header_str = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    sep_str = "-+-".join("-" * w for w in col_widths)

    print("=== Noteworthy Queries (Dynamic Filter: Divergence or Failure) ===")
    print(header_str)
    print(sep_str)

    noteworthy_count = 0

    for case in benchmark.cases:
        bm25_hits = bm25.retrieve(case.query, top_k=5, chunks=chunks)
        dense_hits = dense.retrieve(case.query, top_k=5, chunks=embedded_chunks)

        bm25_rr = reciprocal_rank(bm25_hits, case)
        dense_rr = reciprocal_rank(dense_hits, case)
        delta = abs(bm25_rr - dense_rr)

        is_divergent = delta > 0.0
        is_bm25_fail = bm25_rr == 0.0
        is_dense_fail = dense_rr == 0.0

        if is_divergent or is_bm25_fail or is_dense_fail:
            noteworthy_count += 1
            winner = "BM25" if bm25_rr > dense_rr else ("Dense" if dense_rr > bm25_rr else "Tie")

            flags = []
            if is_bm25_fail:
                flags.append("BM25 Zero-Hit")
            if is_dense_fail:
                flags.append("Dense Zero-Hit")
            if is_divergent:
                flags.append(f"Delta={delta:.2f}")

            flag_str = ", ".join(flags)
            row_str = f"{case.query:<54} | {bm25_rr:<9.2f} | {dense_rr:<10.2f} | {winner:<8} | {delta:<7.2f} | {flag_str}"
            print(row_str)

    print(f"\nFiltered {noteworthy_count} noteworthy query case(s) out of {len(benchmark.cases)} total cases.")


if __name__ == "__main__":
    run_experiment()
