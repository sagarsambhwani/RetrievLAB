"""
Experiment 020: BEIR SciFact Loader, Memory-Safe BM25 vs FAISS Dense Comparative Test.

Demonstrates:
1. Loading the SciFact BEIR dataset (5,183 documents, 300 test queries).
2. Generating and caching dense embeddings in memory-safe micro-batches (batch_size=64).
3. Comparing BM25 vs FAISS Dense on the sample semantic query ('0-dimensional biomaterials...').
4. Comparing BM25 vs FAISS Dense evaluation metrics on a 20-query benchmark subset.
"""

from pathlib import Path
import gc
import numpy as np

from retrievlab.ingestion import BEIRLoader
from retrievlab.retrieval import BM25Retriever
from retrievlab.indexing import FAISSRetriever
from retrievlab.embeddings.fastembed import FastEmbedClient
from retrievlab.evaluation import evaluate_retriever
from retrievlab.models import Chunk


def load_or_generate_embeddings(
    chunks: list[Chunk],
    client: FastEmbedClient,
    cache_path: Path = Path("data/processed/scifact_embeddings.npy"),
    batch_size: int = 64,
) -> list[Chunk]:
    """Load cached embeddings from disk or generate them in memory-safe micro-batches."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"Loading cached embeddings from {cache_path}...")
        emb_matrix = np.load(cache_path)
        for i, chunk in enumerate(chunks):
            chunk.embedding = emb_matrix[i].tolist()
        print(f"Loaded {len(emb_matrix)} cached vectors (dim: {emb_matrix.shape[1]}).")
        return chunks

    total = len(chunks)
    dim = 384  # FastEmbed BGE-small dimension
    print(f"Generating dense embeddings for {total} chunks in micro-batches of {batch_size}...")
    emb_matrix = np.zeros((total, dim), dtype=np.float32)

    for i in range(0, total, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_texts = [c.text for c in batch_chunks]
        # Embed strictly one micro-batch at a time
        batch_vecs = list(client.model.embed(batch_texts))
        emb_matrix[i : i + len(batch_chunks)] = np.array(batch_vecs, dtype=np.float32)

        processed = min(i + batch_size, total)
        if (i // batch_size) % 10 == 0 or processed == total:
            print(f"  [Progress] Embedded {processed}/{total} chunks ({(processed / total) * 100:.1f}%)...")
            gc.collect()

    np.save(cache_path, emb_matrix)
    print(f"Saved {len(emb_matrix)} vectors ({emb_matrix.nbytes / (1024 * 1024):.2f} MB) to cache: {cache_path}")

    for i, chunk in enumerate(chunks):
        chunk.embedding = emb_matrix[i].tolist()

    return chunks


def run_experiment() -> None:
    # 1. Load SciFact data
    print("--- Step 1: Loading SciFact BEIR Dataset ---")
    loader = BEIRLoader("scifact")
    raw_chunks = loader.load_corpus()
    benchmark = loader.load_benchmark(split="test")
    print(f"Loaded total chunks: {len(raw_chunks)}")
    print(f"Loaded total test queries: {len(benchmark.cases)}")

    # 2. Embedding generation & caching (Memory-safe micro-batched)
    print("\n--- Step 2: Dense Embedding Generation / Cache ---")
    client = FastEmbedClient()
    chunks = load_or_generate_embeddings(raw_chunks, client, batch_size=64)

    # 3. Test Sample Query: BM25 vs FAISS Dense
    sample_case = benchmark.cases[0]
    print("\n--- Step 3: Query Comparison on Sample Query ---")
    print(f"Query: '{sample_case.query}'")
    print(f"Target Relevant Document ID: {sample_case.relevant_chunk_ids}")

    # Run BM25
    bm25 = BM25Retriever()
    bm25.index(chunks)
    bm25_res = bm25.retrieve(query=sample_case.query, top_k=5, chunks=chunks)

    # Run FAISS Dense
    faiss_retriever = FAISSRetriever(client)
    faiss_res = faiss_retriever.retrieve(query=sample_case.query, top_k=5, chunks=chunks)

    print("\n[BM25 Top-3 Results]:")
    for rank, res in enumerate(bm25_res[:3], start=1):
        first_line = res.chunk.text.splitlines()[0]
        hit = " [MATCH!]" if res.chunk.id in sample_case.relevant_chunk_ids else ""
        print(f"  Rank {rank}: id='{res.chunk.id}', score={res.score:.4f}, title='{first_line[:60]}'{hit}")

    print("\n[FAISS Dense Top-3 Results]:")
    for rank, res in enumerate(faiss_res[:3], start=1):
        first_line = res.chunk.text.splitlines()[0]
        hit = " [MATCH!]" if res.chunk.id in sample_case.relevant_chunk_ids else ""
        print(f"  Rank {rank}: id='{res.chunk.id}', score={res.score:.4f}, title='{first_line[:60]}'{hit}")

    # Check target rank in both
    bm25_target_rank = next((r for r, x in enumerate(bm25_res, 1) if x.chunk.id in sample_case.relevant_chunk_ids), None)
    dense_target_rank = next((r for r, x in enumerate(faiss_res, 1) if x.chunk.id in sample_case.relevant_chunk_ids), None)
    print(f"\nTarget Document '{sample_case.relevant_chunk_ids[0]}' Rank Summary:")
    print(f"  BM25 Rank:        {bm25_target_rank if bm25_target_rank else 'Not in Top-5 [MISS]'}")
    print(f"  FAISS Dense Rank: {dense_target_rank if dense_target_rank else 'Not in Top-5 [MISS]'}")

    # 4. Comparative Evaluation on First 20 Queries
    print("\n--- Step 4: Comparative Evaluation (First 20 Queries, K=5) ---")
    mini_benchmark = type(benchmark)(cases=benchmark.cases[:20])

    bm25_eval = evaluate_retriever(bm25, mini_benchmark, chunks, k=5)
    dense_eval = evaluate_retriever(faiss_retriever, mini_benchmark, chunks, k=5)

    print(f"\n{'Metric':<18} | {'BM25':<14} | {'FAISS Dense':<14} | {'Delta':<10}")
    print("-" * 62)
    for name, b_val, d_val in [
        ("Recall@5", bm25_eval.recall_at_k, dense_eval.recall_at_k),
        ("Precision@5", bm25_eval.precision_at_k, dense_eval.precision_at_k),
        ("MRR", bm25_eval.mrr, dense_eval.mrr),
    ]:
        delta = d_val - b_val
        print(f"{name:<18} | {b_val:<14.4f} | {d_val:<14.4f} | {delta:+.4f}")


if __name__ == "__main__":
    run_experiment()
