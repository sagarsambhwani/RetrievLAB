"""
Experiment 012

Question:
How does configuring regex patterns in RegexTokenizer affect token extraction,
vocabulary size, and BM25 retrieval for technical queries with numbers and symbols?

Expected Result:
- Strict alphabetic regex ([a-zA-Z]+) filters out numbers, reducing vocabulary size.
- Alphanumeric regex (\\b\\w+\\b) indexes numbers and words.
- BM25 retrieval behaves differently when queries target numeric tokens or version numbers.
"""

from pathlib import Path

from retrievlab.chunking.markdown import MarkdownChunker
from retrievlab.ingestion.loader import DocumentLoader
from retrievlab.preprocessing import RegexTokenizer
from retrievlab.retrieval.bm25 import BM25Retriever


def run_experiment() -> None:
    print("=" * 80)
    print("Experiment 012: BM25 with Configurable RegexTokenizer")
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
    print(f"   {'Tokenizer Configuration':<35} | {'Vocab Size':<12} | {'Avg Chunk Length':<16}")
    print(f"   {'-' * 68}")
    print(f"   {'RegexTokenizer(\\b\\w+\\b) [Alphanumeric]':<35} | {len(vocab_alphanumeric):<12} | {retriever_alphanumeric.average_chunk_length:<16.2f}")
    print(f"   {'RegexTokenizer([a-zA-Z]+) [Alpha Only]':<35} | {len(vocab_alpha_only):<12} | {retriever_alpha_only.average_chunk_length:<16.2f}")

    print(f"\n3. Tokens Filtered Out by Strict Alphabetic Regex:")
    print(f"   Filtered tokens ({len(filtered_out)}): {sorted(list(filtered_out))}")

    # 4. Compare retrieval behavior on query with numbers
    query = "FastAPI Node js and Python 3"
    print(f"\n4. Retrieval Comparison on Query: '{query}'")

    print(f"\n   A) Alphanumeric Retriever (Tokenized query: {tokenizer_alphanumeric.tokenize(query)}):")
    results_alphanumeric = retriever_alphanumeric.retrieve(query=query, top_k=2, chunks=chunks)
    for rank, res in enumerate(results_alphanumeric, start=1):
        print(f"      Rank {rank}: [{res.chunk.id}] Score: {res.score:.3f} | Heading: {res.chunk.metadata.get('heading', 'N/A')}")

    print(f"\n   B) Alpha-Only Retriever (Tokenized query: {tokenizer_alpha_only.tokenize(query)}):")
    results_alpha_only = retriever_alpha_only.retrieve(query=query, top_k=2, chunks=chunks)
    for rank, res in enumerate(results_alpha_only, start=1):
        print(f"      Rank {rank}: [{res.chunk.id}] Score: {res.score:.3f} | Heading: {res.chunk.metadata.get('heading', 'N/A')}")


if __name__ == "__main__":
    run_experiment()
