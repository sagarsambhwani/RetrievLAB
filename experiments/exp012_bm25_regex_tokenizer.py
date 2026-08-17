"""
Experiment 012

Question:
How does configuring regex patterns in RegexTokenizer affect token extraction,
vocabulary size, and benchmark metrics on simple2.json?

Expected Result:
- Strict alphabetic regex ([a-zA-Z]+) filters out numbers, altering vocabulary size.
- Alphanumeric regex (\\b\\w+\\b) indexes numbers and words.
- Benchmark comparison across simple2.json measures the quantitative effect of regex filtering on Recall@5 and MRR.
"""

from pathlib import Path

from retrievlab.chunking.markdown import MarkdownChunker
from retrievlab.evaluation import evaluate_retriever, load_benchmark
from retrievlab.ingestion.loader import DocumentLoader
from retrievlab.preprocessing import RegexTokenizer
from retrievlab.retrieval.bm25 import BM25Retriever


def run_experiment() -> None:
    print("=" * 80)
    print("Experiment 012: BM25 with Configurable RegexTokenizer (Evaluated on simple2.json)")
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

    # 2. Configure two regex tokenizer variations
    # Pattern A: Standard alphanumeric word characters (\b\w+\b)
    tokenizer_alphanumeric = RegexTokenizer(pattern=r"\b\w+\b", lower=True)
    retriever_alphanumeric = BM25Retriever(tokenizer=tokenizer_alphanumeric)
    retriever_alphanumeric.index(chunks)

    # Pattern B: Strict alphabetic characters only ([a-zA-Z]+)
    tokenizer_alpha_only = RegexTokenizer(pattern=r"[a-zA-Z]+", lower=True)
    retriever_alpha_only = BM25Retriever(tokenizer=tokenizer_alpha_only)
    retriever_alpha_only.index(chunks)

    # 3. Compare index statistics
    vocab_alphanumeric = set(retriever_alphanumeric.term_frequencies.keys())
    vocab_alpha_only = set(retriever_alpha_only.term_frequencies.keys())
    filtered_out = vocab_alphanumeric - vocab_alpha_only

    print(f"\n2. Index Comparison:")
    print(f"   {'Tokenizer Configuration':<40} | {'Vocab Size':<12} | {'Avg Chunk Length':<16}")
    print(f"   {'-' * 73}")
    print(f"   {'RegexTokenizer(\\b\\w+\\b) [Alphanumeric]':<40} | {len(vocab_alphanumeric):<12} | {retriever_alphanumeric.average_chunk_length:<16.2f}")
    print(f"   {'RegexTokenizer([a-zA-Z]+) [Alpha Only]':<40} | {len(vocab_alpha_only):<12} | {retriever_alpha_only.average_chunk_length:<16.2f}")

    print(f"\n3. Tokens Filtered Out by Strict Alphabetic Regex:")
    print(f"   Filtered tokens ({len(filtered_out)}): {sorted(list(filtered_out))}")

    # 4. Benchmark evaluation on simple2.json
    benchmark_path = "data/benchmarks/simple2.json"
    benchmark = load_benchmark(benchmark_path)
    print(f"\n4. Benchmark Evaluation ({benchmark_path}):")
    print(f"   Loaded {len(benchmark.cases)} benchmark query cases.")

    res_alphanumeric = evaluate_retriever(
        retriever=retriever_alphanumeric,
        benchmark=benchmark,
        chunks=chunks,
        k=5,
        retriever_name="BM25 (Regex Alphanumeric)",
    )
    res_alpha_only = evaluate_retriever(
        retriever=retriever_alpha_only,
        benchmark=benchmark,
        chunks=chunks,
        k=5,
        retriever_name="BM25 (Regex Alpha-Only)",
    )

    print(f"\n   {'Retriever Configuration':<35} | {'Recall@5':<10} | {'Precision@5':<14} | {'MRR':<8}")
    print(f"   {'-' * 73}")
    for res in [res_alphanumeric, res_alpha_only]:
        print(f"   {res.retriever_name:<35} | {res.recall_at_k:<10.4f} | {res.precision_at_k:<14.4f} | {res.mrr:<8.4f}")


if __name__ == "__main__":
    run_experiment()
