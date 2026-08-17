"""
Experiment 014

Question:
How do stemming algorithms (Porter, Snowball, Lancaster) perform on the simple2.json benchmark
compared to the unstemmed baseline?

Expected Result:
- Morphological conflation improves recall and MRR on queries where word suffixes vary.
- Quantifies performance gains across all 22 benchmark queries for Porter, Snowball, and Lancaster stemmers.
"""

from pathlib import Path

from retrievlab.chunking.markdown import MarkdownChunker
from retrievlab.evaluation import evaluate_retriever, load_benchmark
from retrievlab.ingestion.loader import DocumentLoader
from retrievlab.preprocessing import BasicWordTokenizer, StemmedTokenizer
from retrievlab.retrieval.bm25 import BM25Retriever


def run_experiment() -> None:
    print("=" * 80)
    print("Experiment 014: BM25 with StemmedTokenizer (Evaluated on simple2.json)")
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

    # 2. Initialize Retrievers with different stemming algorithms
    retriever_base = BM25Retriever(tokenizer=BasicWordTokenizer(lower=True))
    retriever_porter = BM25Retriever(tokenizer=StemmedTokenizer(algorithm="porter"))
    retriever_snowball = BM25Retriever(tokenizer=StemmedTokenizer(algorithm="snowball"))
    retriever_lancaster = BM25Retriever(tokenizer=StemmedTokenizer(algorithm="lancaster"))

    retrievers = {
        "BM25 (No Stemming)": retriever_base,
        "BM25 (Porter Stemmer)": retriever_porter,
        "BM25 (Snowball Stemmer)": retriever_snowball,
        "BM25 (Lancaster Stemmer)": retriever_lancaster,
    }

    # 3. Index corpus and compare vocabulary compression
    print(f"\n2. Index Vocabulary Compression:")
    print(f"   {'Algorithm':<28} | {'Vocab Size':<12} | {'Avg Chunk Length':<16}")
    print(f"   {'-' * 60}")
    for name, r in retrievers.items():
        r.index(chunks)
        print(f"   {name:<28} | {len(r.term_frequencies):<12} | {r.average_chunk_length:<16.2f}")

    # 4. Inspect morphological transformations across algorithms
    sample_words = ["deploying", "deployment", "containers", "containerized", "programming", "applications"]
    print(f"\n3. Stemming Transformations:")
    print(f"   {'Original Word':<16} | {'Porter':<14} | {'Snowball':<14} | {'Lancaster':<14}")
    print(f"   {'-' * 62}")
    for word in sample_words:
        p = retriever_porter.tokenizer.stem(word)  # type: ignore[attr-defined]
        s = retriever_snowball.tokenizer.stem(word)  # type: ignore[attr-defined]
        l = retriever_lancaster.tokenizer.stem(word)  # type: ignore[attr-defined]
        print(f"   {word:<16} | {p:<14} | {s:<14} | {l:<14}")

    # 5. Evaluate all on simple2.json benchmark
    benchmark_path = "data/benchmarks/simple2.json"
    benchmark = load_benchmark(benchmark_path)
    print(f"\n4. Benchmark Evaluation ({benchmark_path}):")
    print(f"   Loaded {len(benchmark.cases)} benchmark query cases.")

    results = []
    for name, r in retrievers.items():
        res = evaluate_retriever(
            retriever=r,
            benchmark=benchmark,
            chunks=chunks,
            k=5,
            retriever_name=name,
        )
        results.append(res)

    print(f"\n   {'Retriever Configuration':<28} | {'Recall@5':<10} | {'Precision@5':<14} | {'MRR':<8}")
    print(f"   {'-' * 66}")
    for res in results:
        print(f"   {res.retriever_name:<28} | {res.recall_at_k:<10.4f} | {res.precision_at_k:<14.4f} | {res.mrr:<8.4f}")


if __name__ == "__main__":
    run_experiment()
