"""
Experiment 012

Question:
How does configuring regex patterns in RegexTokenizer affect per-query retrieval metrics (Recall@5, Precision@5, MRR)
on simple2.json?

Expected Result:
- Prints a per-query comparison table comparing Alphanumeric vs Alpha-Only regex configurations across Recall, Precision, and MRR.
- Quantifies whether stripping numeric tokens affects precision or introduces false positives.
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
from retrievlab.preprocessing import RegexTokenizer
from retrievlab.retrieval.bm25 import BM25Retriever


def run_experiment() -> None:
    print("=" * 118)
    print("Experiment 012: BM25 with Configurable RegexTokenizer (Evaluated on simple2.json)")
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

    # 2. Configure two regex tokenizer variations
    tokenizer_alphanumeric = RegexTokenizer(pattern=r"\b\w+\b", lower=True)
    retriever_alphanumeric = BM25Retriever(tokenizer=tokenizer_alphanumeric)
    retriever_alphanumeric.index(chunks)

    tokenizer_alpha_only = RegexTokenizer(pattern=r"[a-zA-Z]+", lower=True)
    retriever_alpha_only = BM25Retriever(tokenizer=tokenizer_alpha_only)
    retriever_alpha_only.index(chunks)

    # 3. Compare index statistics
    vocab_alphanumeric = set(retriever_alphanumeric.term_frequencies.keys())
    vocab_alpha_only = set(retriever_alpha_only.term_frequencies.keys())

    print(f"\n2. Index Comparison:")
    print(f"   {'Tokenizer Configuration':<40} | {'Vocab Size':<12} | {'Avg Chunk Length':<16}")
    print(f"   {'-' * 73}")
    print(f"   {'RegexTokenizer(\\b\\w+\\b) [Alphanumeric]':<40} | {len(vocab_alphanumeric):<12} | {retriever_alphanumeric.average_chunk_length:<16.2f}")
    print(f"   {'RegexTokenizer([a-zA-Z]+) [Alpha Only]':<40} | {len(vocab_alpha_only):<12} | {retriever_alpha_only.average_chunk_length:<16.2f}")

    # 4. Per-Query Benchmark Evaluation
    benchmark_path = "data/benchmarks/simple2.json"
    benchmark = load_benchmark(benchmark_path)
    print(f"\n3. Per-Query Benchmark Evaluation ({benchmark_path}):")
    print(f"   Loaded {len(benchmark.cases)} benchmark cases.\n")

    print(f"{'#':<3} | {'Query':<40} | {'Expected':<16} | {'Alphanumeric (R/P/MRR)':<26} | {'Alpha-Only (R/P/MRR)':<24}")
    print("-" * 118)

    rec_a, prec_a, rr_a = [], [], []
    rec_b, prec_b, rr_b = [], [], []

    for i, case in enumerate(benchmark.cases, start=1):
        res_a = retriever_alphanumeric.retrieve(query=case.query, top_k=5, chunks=chunks)
        res_b = retriever_alpha_only.retrieve(query=case.query, top_k=5, chunks=chunks)

        r_a = recall_at_k(res_a, case, 5)
        p_a = precision_at_k(res_a, case, 5)
        m_a = reciprocal_rank(res_a, case)
        rec_a.append(r_a)
        prec_a.append(p_a)
        rr_a.append(m_a)

        r_b = recall_at_k(res_b, case, 5)
        p_b = precision_at_k(res_b, case, 5)
        m_b = reciprocal_rank(res_b, case)
        rec_b.append(r_b)
        prec_b.append(p_b)
        rr_b.append(m_b)

        exp_str = ",".join(case.relevant_chunk_ids)
        if len(exp_str) > 16:
            exp_str = exp_str[:13] + "..."

        q_str = case.query
        if len(q_str) > 40:
            q_str = q_str[:37] + "..."

        score_a_str = f"{r_a:.2f} / {p_a:.2f} / {m_a:.2f}"
        score_b_str = f"{r_b:.2f} / {p_b:.2f} / {m_b:.2f}"

        print(f"{i:<3} | {q_str:<40} | {exp_str:<16} | {score_a_str:<26} | {score_b_str:<24}")

    print("-" * 118)
    print(f"Alphanumeric Mean Recall@5: {sum(rec_a)/len(rec_a):.4f} | Precision@5: {sum(prec_a)/len(prec_a):.4f} | MRR: {sum(rr_a)/len(rr_a):.4f}")
    print(f"Alpha-Only   Mean Recall@5: {sum(rec_b)/len(rec_b):.4f} | Precision@5: {sum(prec_b)/len(prec_b):.4f} | MRR: {sum(rr_b)/len(rr_b):.4f}\n")


if __name__ == "__main__":
    run_experiment()
