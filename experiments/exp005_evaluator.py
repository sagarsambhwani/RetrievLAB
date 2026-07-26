"""
Experiment: 005

Question:
Does the evaluator correctly compute Recall@K
for the DenseRetriever using the benchmark dataset?

Expected Result:
- The evaluator should compute Recall@K for each benchmark query.
"""
from pathlib import Path

from retrievlab.ingestion.loader import DocumentLoader
from retrievlab.chunking.markdown import MarkdownChunker
from retrievlab.embeddings.embedder import Embedder
from retrievlab.embeddings.fastembed import FastEmbedClient
from retrievlab.retrieval.dense import DenseRetriever
from retrievlab.evaluation.metrics import recall, reciprocal_rank
from retrievlab.evaluation.benchmark import load_benchmark


client = FastEmbedClient()

loader = DocumentLoader()
chunker = MarkdownChunker()
embedder = Embedder(client)
retriever = DenseRetriever(client)

documents = loader.load(Path("data/raw"))
benchmark = load_benchmark("data/benchmarks/simple.json")

chunks = []
for document in documents:
    chunks.extend(chunker.chunk(document))

embedded_chunks = embedder.embed(chunks)

scores = []
for case in benchmark.cases:
    results = retriever.retrieve(
        query=case.query,
        top_k=5,
        chunks=embedded_chunks,
    )
    score = recall(
    retrieved_results=results,
    expected_results=case,
    )
    scores.append(reciprocal_rank(
        retrieved_results=results,
        expected_results=case
    ))
    print(f"\nQuery: {case.query}")
    print(f"Expected Relevant Chunk IDs: {case.relevant_chunk_ids}")
    print(f"Retrieved Chunk IDs: {[result.chunk.id for result in results]}")
    print(f"Recall Score: {score}")
    print(f"Reciprocal Rank Score: {scores[-1]}")
print(f"\nMean Reciprocal Rank: {sum(scores)/len(scores)}")
