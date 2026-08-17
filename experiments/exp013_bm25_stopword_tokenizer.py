"""
Experiment 013

Question:
How does removing English stopwords alter per-query retrieval metrics and ranking across simple2.json?

Expected Result:
- Prints a per-query comparison table showing Baseline vs Stopword-Filtered Recall@5 and MRR.
- Identifies queries where stopword removal resolved ranking inversions.
"""

from pathlib import Path

from retrievlab.chunking.markdown import MarkdownChunker
from retrievlab.evaluation import (
    load_benchmark,
    recall_at_k,
    reciprocal_rank,
)
from retrievlab.ingestion.loader import DocumentLoader
from retrievlab.preprocessing import BasicWordTokenizer, StopwordTokenizer
from retrievlab.retrieval.bm25 import BM25Retriever


def run_experiment() -> None:
    print("=" * 118)
    print("Experiment 013: BM25 with StopwordTokenizer (Evaluated on simple2.json)")
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

    # 2. Initialize Baseline and Stopword Retrievers
    retriever_baseline = BM25Retriever(tokenizer=BasicWordTokenizer(lower=True))
    retriever_baseline.index(chunks)

    retriever_stopwords = BM25Retriever(tokenizer=StopwordTokenizer())
    retriever_stopwords.index(chunks)

    # 3. Compare index statistics
    vocab_baseline = len(retriever_baseline.term_frequencies)
    vocab_stopwords = len(retriever_stopwords.term_frequencies)
    len_baseline = retriever_baseline.average_chunk_length
    len_stopwords = retriever_stopwords.average_chunk_length

    print(f"\n2. Index Statistics Comparison:")
    print(f"   {'Configuration':<30} | {'Vocab Size':<12} | {'Avg Chunk Length':<16}")
    print(f"   {'-' * 63}")
    print(f"   {'Baseline (BasicWordTokenizer)':<30} | {vocab_baseline:<12} | {len_baseline:<16.2f}")
    print(f"   {'Filtered (StopwordTokenizer)':<30} | {vocab_stopwords:<12} | {len_stopwords:<16.2f}")
    print(f"   Reduction in average chunk token count: {((len_baseline - len_stopwords) / len_baseline) * 100:.1f}%")

    # 4. Per-Query Benchmark Evaluation
    benchmark_path = "data/benchmarks/simple2.json"
    benchmark = load_benchmark(benchmark_path)
    print(f"\n3. Per-Query Benchmark Evaluation ({benchmark_path}):")
    print(f"   Loaded {len(benchmark.cases)} benchmark cases.\n")

    print(f"{'#':<3} | {'Query':<42} | {'Expected':<18} | {'Baseline (R@5/MRR)':<22} | {'+Stopwords (R@5/MRR)':<22} | {'Diff'}")
    print("-" * 118)

    rec_base, rr_base = [], []
    rec_stop, rr_stop = [], []

    for i, case in enumerate(benchmark.cases, start=1):
        res_b = retriever_baseline.retrieve(query=case.query, top_k=5, chunks=chunks)
        res_s = retriever_stopwords.retrieve(query=case.query, top_k=5, chunks=chunks)

        r_b = recall_at_k(retrieved_results=res_b, expected_results=case, k=5)
        m_b = reciprocal_rank(retrieved_results=res_b, expected_results=case)
        rec_base.append(r_b)
        rr_base.append(m_b)

        r_s = recall_at_k(retrieved_results=res_s, expected_results=case, k=5)
        m_s = reciprocal_rank(retrieved_results=res_s, expected_results=case)
        rec_stop.append(r_s)
        rr_stop.append(m_s)

        diff_str = f"+{m_s - m_b:.2f}" if m_s > m_b else (f"{m_s - m_b:.2f}" if m_s < m_b else "=")

        exp_str = ",".join(case.relevant_chunk_ids)
        if len(exp_str) > 18:
            exp_str = exp_str[:15] + "..."

        q_str = case.query
        if len(q_str) > 42:
            q_str = q_str[:39] + "..."

        score_b_str = f"{r_b:.2f} / {m_b:.2f}"
        score_s_str = f"{r_s:.2f} / {m_s:.2f}"

        print(f"{i:<3} | {q_str:<42} | {exp_str:<18} | {score_b_str:<22} | {score_s_str:<22} | {diff_str}")

    print("-" * 118)
    print(f"Baseline   Mean Recall@5: {sum(rec_base)/len(rec_base):.4f} | MRR: {sum(rr_base)/len(rr_base):.4f}")
    print(f"+Stopwords Mean Recall@5: {sum(rec_stop)/len(rec_stop):.4f} | MRR: {sum(rr_stop)/len(rr_stop):.4f}\n")


if __name__ == "__main__":
    run_experiment()
