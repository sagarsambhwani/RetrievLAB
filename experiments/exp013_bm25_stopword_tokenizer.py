"""
Experiment 013

Question:
How does removing English stopwords alter BM25 index statistics, average chunk length,
and scoring on conversational vs keyword queries?

Expected Result:
- Stopword removal drops high-frequency function words (what, is, the, for, are, of).
- Average chunk length decreases, and stopword tokens have 0 postings in the index.
- Conversational queries score only on discriminative content terms.
"""

from pathlib import Path

from retrievlab.chunking.markdown import MarkdownChunker
from retrievlab.ingestion.loader import DocumentLoader
from retrievlab.preprocessing import BasicWordTokenizer, StopwordTokenizer
from retrievlab.retrieval.bm25 import BM25Retriever


def run_experiment() -> None:
    print("=" * 80)
    print("Experiment 013: BM25 with StopwordTokenizer")
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

    # 5. Query Scoring Comparison on a Conversational Question
    query = "What are the key features of FastAPI?"
    print(f"\n4. Query Term Contribution for: '{query}'")

    print("\n   A) Baseline Token Breakdown:")
    base_tokens = retriever_baseline.tokenizer.tokenize(query)
    print(f"      Tokens: {base_tokens}")
    for token in base_tokens:
        idf = retriever_baseline._inverse_document_frequency(token)
        print(f"      - '{token}': idf={idf:.3f}")

    print("\n   B) Stopword-Filtered Token Breakdown:")
    stop_tokens = retriever_stopwords.tokenizer.tokenize(query)
    print(f"      Tokens: {stop_tokens}")
    for token in stop_tokens:
        idf = retriever_stopwords._inverse_document_frequency(token)
        print(f"      - '{token}': idf={idf:.3f}")

    # 6. Execute retrieval
    print(f"\n5. Retrieval Results for '{query}':")
    results_base = retriever_baseline.retrieve(query=query, top_k=2, chunks=chunks)
    results_stop = retriever_stopwords.retrieve(query=query, top_k=2, chunks=chunks)

    print("   Baseline Top 2:")
    for rank, res in enumerate(results_base, start=1):
        print(f"     Rank {rank}: [{res.chunk.id}] Score: {res.score:.3f} | Heading: {res.chunk.metadata.get('heading', 'N/A')}")

    print("   Stopword-Filtered Top 2:")
    for rank, res in enumerate(results_stop, start=1):
        print(f"     Rank {rank}: [{res.chunk.id}] Score: {res.score:.3f} | Heading: {res.chunk.metadata.get('heading', 'N/A')}")


if __name__ == "__main__":
    run_experiment()
