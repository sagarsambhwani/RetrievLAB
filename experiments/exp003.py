from pathlib import Path

from retrievlab.retrieval.bm25 import BM25Retriever
from retrievlab.ingestion.loader import DocumentLoader
from retrievlab.chunking.markdown import MarkdownChunker

loader = DocumentLoader()
chunker = MarkdownChunker()
retriever = BM25Retriever()

docs = loader.load(Path("data/raw/"))
print(f"Loader loaded files successfully: {len(docs)}")

chunks = []
for doc in docs:
    chunks.extend(chunker.chunk(doc))

print(f"chunks loaded successfully: {len(chunks)}")
print(f"chunk[0]: {chunks[0]}")

indexed = retriever.index(chunks)
print(f"results from bm25 retriever: {retriever.chunk_lengths, retriever.average_chunk_length}")
# print(f"term frequncies: {retriever.term_frequencies}")
for token in ["fastapi", "python", "docker", "framework"]:
    df = len(retriever.term_frequencies.get(token, {}))
    idf = retriever._inverse_document_frequency(token)
    print(f"{token:10} df={df} idf={idf:.3f}")

results = retriever.retrieve(query="What is a fastapi?", top_k=5, chunks=chunks)
print(f"results: {results}")