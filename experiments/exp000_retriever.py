"""
Experiment: 004

Question:
Does the Retriever correctly retrieve relevant Markdown chunks based on a query using the FastEmbed model?
Expected Result:
- The Retriever should return the top k relevant chunks based on the query.
"""

from pathlib import Path
from math import sqrt

from retrievlab.ingestion.loader import DocumentLoader
from retrievlab.chunking.markdown import MarkdownChunker
from retrievlab.embeddings.embedder import Embedder
from retrievlab.embeddings.fastembed import FastEmbedClient
from retrievlab.retrieval.dense import DenseRetriever
loader = DocumentLoader()
chunker = MarkdownChunker()
client = FastEmbedClient()
embedder = Embedder(client)
retriever = DenseRetriever(client)

documents = loader.load(Path("data/raw/"))
print(f"\nLoaded {len(documents)} documents")

chunks = []
for document in documents:
    chunks.extend(chunker.chunk(document))

embedded_chunks = embedder.embed(chunks)
vector = embedded_chunks[0].embedding

norm = sqrt(sum(x**2 for x in vector))
print(f"\nNorm of the first chunk's embedding: {norm}")


print(chunks[0].embedding[:5])

search_results = retriever.retrieve(query="What is FastAPI?", top_k=5, chunks=embedded_chunks)
print(f"\nRetrieved {len(search_results)} results for the query 'What is FastAPI?'\n")
print(chunks[0].embedding[:5])

for i, result in enumerate(search_results, start=1):
    print(f"Result {i}")
    print(f"Chunk: {result.chunk.text}")
    print(f"Score: {result.score}")