"""
Experiment 014

Question:
How do stemming algorithms (Porter, Snowball, Lancaster) perform across Recall@5, Precision@5, and MRR
on individual queries in simple2.json?

Expected Result:
- Prints a per-query comparison table showing Baseline vs Porter vs Snowball metrics across Recall, Precision, and MRR.
- Highlights queries where stemming recovered missed chunks and improved both Recall and Precision.
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
from retrievlab.preprocessing import BasicWordTokenizer, StemmedTokenizer
from retrievlab.retrieval.bm25 import BM25Retriever


def run_experiment() -> None:
    print("=" * 118)
    print("Experiment 014: BM25 with StemmedTokenizer (Evaluated on simple2.json)")
    print("=" * 118)

    # 1. Load and chunk documents
    loader = DocumentLoader()
    chunker = MarkdownChunker()
    raw_path = Path("data/raw")
    documents = loader.load(raw_path)

    chunks = []
    for doc in documents:
        chunks.extend(chunker.chunk(doc))

    print("\n1. Corpus Loading:")
    print(f"   Loaded {len(documents)} document(s) producing {len(chunks)} chunk(s).")

    # 2. Initialize Retrievers with different stemming algorithms
    retriever_base = BM25Retriever(tokenizer=BasicWordTokenizer(lower=True))
    retriever_porter = BM25Retriever(tokenizer=StemmedTokenizer(algorithm="porter"))
    retriever_snowball = BM25Retriever(tokenizer=StemmedTokenizer(algorithm="snowball"))
    retriever_lancaster = BM25Retriever(tokenizer=StemmedTokenizer(algorithm="lancaster"))

    retrievers = {
        "Baseline": retriever_base,
        "Porter": retriever_porter,
        "Snowball": retriever_snowball,
        "Lancaster": retriever_lancaster,
    }

    for r in retrievers.values():
        r.index(chunks)

    # 3. Per-Query Benchmark Evaluation
    benchmark_path = "data/benchmarks/simple2.json"
    benchmark = load_benchmark(benchmark_path)
    print(f"\n2. Per-Query Benchmark Evaluation ({benchmark_path}):")
    print(f"   Loaded {len(benchmark.cases)} benchmark cases.\n")

    print(f"{'#':<3} | {'Query':<36} | {'Expected':<16} | {'Baseline (R/P/MRR)':<24} | {'Porter (R/P/MRR)':<22} | {'Snowball (R/P/MRR)':<22} | {'Lancaster (R/P/MRR)':<22}")
    print("-" * 140)

    rec_base, prec_base, rr_base = [], [], []
    rec_port, prec_port, rr_port = [], [], []
    rec_snow, prec_snow, rr_snow = [], [], []
    rec_lanc, prec_lanc, rr_lanc = [], [], []

    for i, case in enumerate(benchmark.cases, start=1):
        res_b = retriever_base.retrieve(query=case.query, top_k=5, chunks=chunks)
        res_p = retriever_porter.retrieve(query=case.query, top_k=5, chunks=chunks)
        res_s = retriever_snowball.retrieve(query=case.query, top_k=5, chunks=chunks)
        res_l = retriever_lancaster.retrieve(query=case.query, top_k=5, chunks=chunks)

        r_b, p_b, m_b = recall_at_k(res_b, case, 5), precision_at_k(res_b, case, 5), reciprocal_rank(res_b, case)
        r_p, p_p, m_p = recall_at_k(res_p, case, 5), precision_at_k(res_p, case, 5), reciprocal_rank(res_p, case)
        r_s, p_s, m_s = recall_at_k(res_s, case, 5), precision_at_k(res_s, case, 5), reciprocal_rank(res_s, case)
        r_l, p_l, m_l = recall_at_k(res_l, case, 5), precision_at_k(res_l, case, 5), reciprocal_rank(res_l, case)

        rec_base.append(r_b)
        prec_base.append(p_b)
        rr_base.append(m_b)

        rec_port.append(r_p)
        prec_port.append(p_p)
        rr_port.append(m_p)

        rec_snow.append(r_s)
        prec_snow.append(p_s)
        rr_snow.append(m_s)

        rec_lanc.append(r_l)
        prec_lanc.append(p_l)
        rr_lanc.append(m_l)

        exp_str = ",".join(case.relevant_chunk_ids)
        if len(exp_str) > 16:
            exp_str = exp_str[:13] + "..."

        q_str = case.query
        if len(q_str) > 36:
            q_str = q_str[:33] + "..."

        print(f"{i:<3} | {q_str:<36} | {exp_str:<16} | {f'{r_b:.2f}/{p_b:.2f}/{m_b:.2f}':<24} | {f'{r_p:.2f}/{p_p:.2f}/{m_p:.2f}':<22} | {f'{r_s:.2f}/{p_s:.2f}/{m_s:.2f}':<22} | {f'{r_l:.2f}/{p_l:.2f}/{m_l:.2f}':<22}")

    print("-" * 140)
    print(f"Baseline Mean Recall@5: {sum(rec_base)/len(rec_base):.4f} | Precision@5: {sum(prec_base)/len(prec_base):.4f} | MRR: {sum(rr_base)/len(rr_base):.4f}")
    print(f"Porter   Mean Recall@5: {sum(rec_port)/len(rec_port):.4f} | Precision@5: {sum(prec_port)/len(prec_port):.4f} | MRR: {sum(rr_port)/len(rr_port):.4f}")
    print(f"Snowball Mean Recall@5: {sum(rec_snow)/len(rec_snow):.4f} | Precision@5: {sum(prec_snow)/len(prec_snow):.4f} | MRR: {sum(rr_snow)/len(rr_snow):.4f}")
    print(f"Lancaster Mean Recall@5: {sum(rec_lanc)/len(rec_lanc):.4f} | Precision@5: {sum(prec_lanc)/len(prec_lanc):.4f} | MRR: {sum(rr_lanc)/len(rr_lanc):.4f}\n")


if __name__ == "__main__":
    run_experiment()
