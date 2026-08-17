"""
Experiment 013

Question:
How does removing English stopwords alter BM25 index statistics and retrieval metrics (Recall@5, MRR)
on the simple2.json benchmark dataset?

Expected Result:
- Stopword removal drops high-frequency function words (what, is, the, for, are, of).
- Average chunk length decreases by >20%, and stopword tokens have 0 postings in the index.
- Evaluates on simple2.json to quantify overall metric improvements from eliminating stopword noise.
"""

from pathlib import Path

from retrievlab.chunking.markdown import MarkdownChunker
from retrievlab.evaluation import evaluate_retriever, load_benchmark
from retrievlab.ingestion.loader import DocumentLoader
from retrievlab.preprocessing import BasicWordTokenizer, StopwordTokenizer
from retrievlab.retrieval.bm25 import BM25Retriever


def run_experiment() -> None:
    print("=" * 80)
    print("Experiment 013: BM25 with StopwordTokenizer (Evaluated on simple2.json)")
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

    # 4. Inspect stopword DF in both indices
    sample_stopwords = ["what", "is", "the", "are", "of", "for", "and"]
    print(f"\n3. Stopword Index Presence:")
    print(f"   {'Token':<10} | {'Baseline DF':<15} | {'Stopword-Filtered DF':<20}")
    print(f"   {'-' * 50}")
    for word in sample_stopwords:
        df_base = len(retriever_baseline.term_frequencies.get(word, {}))
        df_stop = len(retriever_stopwords.term_frequencies.get(word, {}))
        print(f"   {word:<10} | {df_base:<15} | {df_stop:<20}")

    # 5. Evaluate both on simple2.json benchmark
    benchmark_path = "data/benchmarks/simple2.json"
    benchmark = load_benchmark(benchmark_path)
    print(f"\n4. Benchmark Evaluation ({benchmark_path}):")
    print(f"   Loaded {len(benchmark.cases)} benchmark query cases.")

    res_base = evaluate_retriever(
        retriever=retriever_baseline,
        benchmark=benchmark,
        chunks=chunks,
        k=5,
        retriever_name="BM25 (Baseline Unfiltered)",
    )
    res_stopwords = evaluate_retriever(
        retriever=retriever_stopwords,
        benchmark=benchmark,
        chunks=chunks,
        k=5,
        retriever_name="BM25 (+ StopwordTokenizer)",
    )

    print(f"\n   {'Retriever Configuration':<30} | {'Recall@5':<10} | {'Precision@5':<14} | {'MRR':<8}")
    print(f"   {'-' * 68}")
    for res in [res_base, res_stopwords]:
        print(f"   {res.retriever_name:<30} | {res.recall_at_k:<10.4f} | {res.precision_at_k:<14.4f} | {res.mrr:<8.4f}")


if __name__ == "__main__":
    run_experiment()
