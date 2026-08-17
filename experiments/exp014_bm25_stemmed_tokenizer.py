"""
Experiment 014

Question:
How do stemming algorithms (Porter, Snowball, Lancaster) enable BM25 to match
inflected query terms against their morphological root forms in the corpus?

Expected Result:
- Unstemmed baseline BM25 misses chunks when query words use different morphological suffixes.
- StemmedTokenizer (Porter/Snowball/Lancaster) conflates inflected variants to shared stems,
  allowing BM25 to retrieve relevant chunks with positive scores.
- Lancaster performs more aggressive stemming than Porter/Snowball.
"""

from pathlib import Path

from retrievlab.chunking.markdown import MarkdownChunker
from retrievlab.ingestion.loader import DocumentLoader
from retrievlab.preprocessing import BasicWordTokenizer, StemmedTokenizer
from retrievlab.retrieval.bm25 import BM25Retriever


def run_experiment() -> None:
    print("=" * 80)
    print("Experiment 014: BM25 with StemmedTokenizer (Porter, Snowball, Lancaster)")
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
        "Baseline (No Stemming)": retriever_base,
        "Porter Stemmer": retriever_porter,
        "Snowball Stemmer": retriever_snowball,
        "Lancaster Stemmer": retriever_lancaster,
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

    # 5. Retrieval test with inflectional query
    query = "deploying applications with containers"
    print(f"\n4. Retrieval Comparison for Inflectional Query: '{query}'")
    print(f"   (Corpus uses: 'Deploy', 'deployment', 'Containers', 'applications')")

    for name, r in retrievers.items():
        tokens = r.tokenizer.tokenize(query)
        results = r.retrieve(query=query, top_k=2, chunks=chunks)
        print(f"\n   [{name}]")
        print(f"   Tokens: {tokens}")
        for rank, res in enumerate(results, start=1):
            heading = res.chunk.metadata.get("heading", "N/A")
            print(f"     Rank {rank}: [{res.chunk.id}] Score: {res.score:.3f} | Heading: {heading}")


if __name__ == "__main__":
    run_experiment()
