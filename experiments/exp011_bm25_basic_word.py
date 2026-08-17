"""
Experiment 011

Question:
How does BM25 perform with explicit BasicWordTokenizer dependency injection on the raw document corpus?

Expected Result:
- BasicWordTokenizer tokenizes text into lowercase word tokens using standard word boundaries.
- BM25Retriever successfully computes index statistics (average chunk length, vocabulary size, IDF).
- Keyword queries retrieve the expected relevant chunks with positive BM25 relevance scores.
"""

from pathlib import Path

from retrievlab.chunking.markdown import MarkdownChunker
from retrievlab.ingestion.loader import DocumentLoader
from retrievlab.preprocessing import BasicWordTokenizer
from retrievlab.retrieval.bm25 import BM25Retriever


def run_experiment() -> None:
    print("=" * 80)
    print("Experiment 011: BM25 with BasicWordTokenizer")
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

    # 2. Initialize BasicWordTokenizer and BM25Retriever
    tokenizer = BasicWordTokenizer(lower=True)
    retriever = BM25Retriever(tokenizer=tokenizer)

    # 3. Index corpus
    retriever.index(chunks)
    vocab_size = len(retriever.term_frequencies)
    print(f"\n2. Index Statistics:")
    print(f"   Tokenizer: BasicWordTokenizer(lower=True)")
    print(f"   Vocabulary Size: {vocab_size} unique terms")
    print(f"   Average Chunk Length: {retriever.average_chunk_length:.2f} tokens")

    # 4. Inspect sample token DF and IDF values
    sample_tokens = ["fastapi", "docker", "python", "framework", "containers", "the"]
    print(f"\n3. Sample Token Statistics:")
    print(f"   {'Token':<15} | {'Doc Frequency (DF)':<20} | {'IDF':<8}")
    print(f"   {'-' * 48}")
    for token in sample_tokens:
        df = len(retriever.term_frequencies.get(token, {}))
        idf = retriever._inverse_document_frequency(token)
        print(f"   {token:<15} | {df:<20} | {idf:<8.3f}")

    # 5. Execute retrieval queries
    queries = [
        "What is FastAPI?",
        "Docker containers",
        "Python programming language",
    ]

    print(f"\n4. Retrieval Results:")
    for query in queries:
        print(f"\n   Query: '{query}'")
        results = retriever.retrieve(query=query, top_k=2, chunks=chunks)
        for rank, res in enumerate(results, start=1):
            heading = res.chunk.metadata.get("heading", "N/A")
            first_line = res.chunk.text.split("\n")[0]
            print(f"     Rank {rank}: [{res.chunk.id}] (Score: {res.score:.3f}) | Heading: {heading}")
            print(f"             Snippet: {first_line[:70]}...")


if __name__ == "__main__":
    run_experiment()
