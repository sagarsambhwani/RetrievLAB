"""
Experiment: 005

Question:
Does the evaluator correctly compute Recall@K
for the DenseRetriever using the benchmark dataset?

Expected Result:
- The evaluator should compute Recall@K for each benchmark query.
"""
import json
from pathlib import Path

from retrievlab.ingestion.loader import DocumentLoader
from retrievlab.chunking.markdown import MarkdownChunker
from retrievlab.embeddings.embedder import Embedder
from retrievlab.embeddings.fastembed import FastEmbedClient
from retrievlab.retrieval.dense import DenseRetriever
from retrievlab.evaluation.metrics import recall
from retrievlab.evaluation.benchmark import BenchmarkCase


client = FastEmbedClient()

loader = DocumentLoader()
chunker = MarkdownChunker()
embedder = Embedder(client)
retriever = DenseRetriever(client)

documents = loader.load(Path("data/raw"))
benchmark_path  = Path("data/benchmarks/simple.json")

chunks = []
for document in documents:
    chunks.extend(chunker.chunk(document))

embedded_chunks = embedder.embed(chunks)

with benchmark_path.open("r") as file:
    data = json.load(file)

cases = [BenchmarkCase(**item) for item in data]

for case in cases:
    results = retriever.retrieve(
        query=case.query,
        top_k=5,
        chunks=embedded_chunks,
    )
    score = recall(
    retrieved_results=results,
    expected_results=case,
    )
    print(f"\nQuery: {case.query}")
    print(f"Expected Relevant Chunk IDs: {case.relevant_chunk_ids}")
    print(f"Retrieved Chunk IDs: {[result.chunk.id for result in results]}")
    print(f"Recall Score: {score}")
