"""
Experiment 024: Two-Stage Cross-Encoder Neural Re-Ranking Benchmark.

Evaluates 6 systems on BEIR SciFact (300 test queries, 5,183 documents):
1. Single-Stage BM25 Baseline
2. Single-Stage FAISS Dense Baseline
3. Single-Stage Hybrid RRF (1:2)
4. Two-Stage: BM25 Candidates (K=50) + Cross-Encoder
5. Two-Stage: Dense Candidates (K=50) + Cross-Encoder
6. Two-Stage: Multi-Retriever Union Candidates (K=50) + Cross-Encoder

Optimized with bulk batch inference across all candidate pools to complete in < 60s.
"""

import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from retrievlab.ingestion import BEIRLoader
from retrievlab.retrieval import BM25Retriever, HybridRetriever
from retrievlab.indexing import FAISSRetriever
from retrievlab.embeddings.fastembed import FastEmbedClient
from retrievlab.selection import (
    SingleRetrieverCandidateGenerator,
    MultiRetrieverCandidateGenerator,
)
from retrievlab.ranking import CrossEncoderReRanker
from retrievlab.evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    reciprocal_rank,
    ndcg_at_k,
    average_precision_at_k,
    hit_at_k,
)
from retrievlab.models import Chunk, SearchResult


@dataclass
class SystemMetrics:
    """Aggregated evaluation metrics for a retrieval/re-ranking system."""

    name: str
    recall_5: float
    precision_5: float
    mrr_val: float
    ndcg_5: float
    map_5: float
    hit_5: float
    recall_10: float
    precision_10: float
    ndcg_10: float
    map_10: float
    hit_10: float
    mean_latency_ms: float


def load_cached_embeddings(chunks: list[Chunk], cache_path: Path) -> list[Chunk]:
    """Load precomputed embedding vectors from disk into chunks."""
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Embeddings cache not found at {cache_path}. "
            "Run exp020_beir_scifact_loader.py first to generate the cache."
        )
    print(f"Loading cached embeddings from {cache_path}...", flush=True)
    emb_matrix = np.load(cache_path)
    for i, chunk in enumerate(chunks):
        chunk.embedding = emb_matrix[i].tolist()
    print(f"Loaded {len(emb_matrix)} cached vectors (dim: {emb_matrix.shape[1]}).", flush=True)
    return chunks


def evaluate_single_stage(
    name: str,
    retriever,
    benchmark_cases,
    chunks: list[Chunk],
) -> tuple[SystemMetrics, list[list[SearchResult]]]:
    """Evaluate a single-stage baseline retriever."""
    recalls_5, precisions_5, mrrs, ndcgs_5, maps_5, hits_5 = [], [], [], [], [], []
    recalls_10, precisions_10, ndcgs_10, maps_10, hits_10 = [], [], [], [], []
    latencies = []
    all_results = []

    for case in benchmark_cases:
        t0 = time.perf_counter()
        results = retriever.retrieve(case.query, top_k=10, chunks=chunks)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        latencies.append(elapsed_ms)
        all_results.append(results)

        # Cutoff @5
        recalls_5.append(recall_at_k(results, case, k=5))
        precisions_5.append(precision_at_k(results, case, k=5))
        mrrs.append(reciprocal_rank(results, case))
        ndcgs_5.append(ndcg_at_k(results, case, k=5))
        maps_5.append(average_precision_at_k(results, case, k=5))
        hits_5.append(hit_at_k(results, case, k=5))

        # Cutoff @10
        recalls_10.append(recall_at_k(results, case, k=10))
        precisions_10.append(precision_at_k(results, case, k=10))
        ndcgs_10.append(ndcg_at_k(results, case, k=10))
        maps_10.append(average_precision_at_k(results, case, k=10))
        hits_10.append(hit_at_k(results, case, k=10))

    metrics = SystemMetrics(
        name=name,
        recall_5=float(np.mean(recalls_5)),
        precision_5=float(np.mean(precisions_5)),
        mrr_val=float(np.mean(mrrs)),
        ndcg_5=float(np.mean(ndcgs_5)),
        map_5=float(np.mean(maps_5)),
        hit_5=float(np.mean(hits_5)),
        recall_10=float(np.mean(recalls_10)),
        precision_10=float(np.mean(precisions_10)),
        ndcg_10=float(np.mean(ndcgs_10)),
        map_10=float(np.mean(maps_10)),
        hit_10=float(np.mean(hits_10)),
        mean_latency_ms=float(np.mean(latencies)),
    )
    return metrics, all_results


def evaluate_two_stage(
    name: str,
    generator,
    cross_encoder: CrossEncoderReRanker,
    benchmark_cases,
    chunks: list[Chunk],
    top_k_cand: int = 50,
) -> tuple[SystemMetrics, list[list[SearchResult]]]:
    """Evaluate a two-stage system with bulk vectorized Cross-Encoder inference."""
    t_start = time.perf_counter()

    # Step 1: Generate all candidate pools
    pools = []
    for case in benchmark_cases:
        pool = generator.generate(case.query, top_k_per_retriever=top_k_cand, chunks=chunks)
        pools.append(pool)

    # Step 2: Build bulk sequence pairs across all queries
    all_pairs: list[tuple[str, str]] = []
    pool_slices: list[tuple[int, int]] = []
    offset = 0

    for pool in pools:
        cand_chunks = pool.get_chunks()
        count = len(cand_chunks)
        for c in cand_chunks:
            all_pairs.append((pool.query, c.text or ""))
        pool_slices.append((offset, offset + count))
        offset += count

    print(f"  Scoring {len(all_pairs)} total candidate pairs in bulk...", flush=True)

    # Step 3: Run single bulk predict pass
    all_scores = cross_encoder.model.predict(
        all_pairs,
        batch_size=cross_encoder.batch_size,
        show_progress_bar=False,
    )
    t_nn_end = time.perf_counter()
    total_elapsed_ms = (t_nn_end - t_start) * 1000.0
    mean_latency_ms = total_elapsed_ms / len(benchmark_cases)

    # Step 4: Reconstruct ranked SearchResults for each query
    all_results: list[list[SearchResult]] = []
    for pool, (start, end) in zip(pools, pool_slices):
        cand_chunks = pool.get_chunks()
        scores = all_scores[start:end]

        scored: list[SearchResult] = [
            SearchResult(chunk=c, score=float(s))
            for c, s in zip(cand_chunks, scores)
        ]
        scored.sort(key=lambda x: x.score, reverse=True)
        all_results.append(scored)

    # Step 5: Compute metrics
    recalls_5, precisions_5, mrrs, ndcgs_5, maps_5, hits_5 = [], [], [], [], [], []
    recalls_10, precisions_10, ndcgs_10, maps_10, hits_10 = [], [], [], [], []

    for results, case in zip(all_results, benchmark_cases):
        # Cutoff @5
        recalls_5.append(recall_at_k(results, case, k=5))
        precisions_5.append(precision_at_k(results, case, k=5))
        mrrs.append(reciprocal_rank(results, case))
        ndcgs_5.append(ndcg_at_k(results, case, k=5))
        maps_5.append(average_precision_at_k(results, case, k=5))
        hits_5.append(hit_at_k(results, case, k=5))

        # Cutoff @10
        recalls_10.append(recall_at_k(results, case, k=10))
        precisions_10.append(precision_at_k(results, case, k=10))
        ndcgs_10.append(ndcg_at_k(results, case, k=10))
        maps_10.append(average_precision_at_k(results, case, k=10))
        hits_10.append(hit_at_k(results, case, k=10))

    metrics = SystemMetrics(
        name=name,
        recall_5=float(np.mean(recalls_5)),
        precision_5=float(np.mean(precisions_5)),
        mrr_val=float(np.mean(mrrs)),
        ndcg_5=float(np.mean(ndcgs_5)),
        map_5=float(np.mean(maps_5)),
        hit_5=float(np.mean(hits_5)),
        recall_10=float(np.mean(recalls_10)),
        precision_10=float(np.mean(precisions_10)),
        ndcg_10=float(np.mean(ndcgs_10)),
        map_10=float(np.mean(maps_10)),
        hit_10=float(np.mean(hits_10)),
        mean_latency_ms=mean_latency_ms,
    )
    return metrics, all_results


def run_experiment() -> None:
    print("=== Step 1: Loading SciFact BEIR Dataset ===", flush=True)
    loader = BEIRLoader("scifact")
    chunks = loader.load_corpus()
    benchmark = loader.load_benchmark(split="test")
    total_queries = len(benchmark.cases)
    print(f"Loaded {len(chunks)} documents, {total_queries} test queries.", flush=True)

    print("\n=== Step 2: Loading Cached Embeddings ===", flush=True)
    cache_path = Path("data/processed/scifact_embeddings.npy")
    chunks = load_cached_embeddings(chunks, cache_path)

    print("\n=== Step 3: Initializing Retrievers, Generators & Cross-Encoder ===", flush=True)
    client = FastEmbedClient()

    # Base Retrievers
    bm25 = BM25Retriever()
    bm25.index(chunks)

    dense = FAISSRetriever(client)

    hybrid_1_2 = HybridRetriever(
        retrievers=[bm25, dense],
        weights=[1.0, 2.0],
    )

    # Candidate Generators (K=50)
    bm25_gen = SingleRetrieverCandidateGenerator(bm25, name="bm25")
    dense_gen = SingleRetrieverCandidateGenerator(dense, name="dense")
    multi_gen = MultiRetrieverCandidateGenerator({"bm25": bm25, "dense": dense})

    # Neural Re-Ranker on GPU
    cross_encoder = CrossEncoderReRanker(batch_size=64, max_length=256, device="cuda")

    print("\n=== Step 4: Evaluating Single-Stage Baselines ===", flush=True)
    all_metrics: list[SystemMetrics] = []
    results_map = {}

    single_systems = [
        ("BM25 Single-Stage", bm25),
        ("FAISS Dense Single-Stage", dense),
        ("Hybrid RRF (1:2) Single-Stage", hybrid_1_2),
    ]

    for name, ret in single_systems:
        print(f"Evaluating: {name}...", flush=True)
        m, res = evaluate_single_stage(name, ret, benchmark.cases, chunks)
        all_metrics.append(m)
        results_map[name] = res
        print(f"  @5  -> Recall: {m.recall_5:.4f}, nDCG: {m.ndcg_5:.4f}, MRR: {m.mrr_val:.4f}, MAP: {m.map_5:.4f}, Hit: {m.hit_5*100:.1f}%", flush=True)
        print(f"  @10 -> Recall: {m.recall_10:.4f}, nDCG: {m.ndcg_10:.4f}, MAP: {m.map_10:.4f}, Hit: {m.hit_10*100:.1f}%", flush=True)
        print(f"  Mean Latency: {m.mean_latency_ms:.2f} ms", flush=True)

    print("\n=== Step 5: Evaluating Two-Stage Cross-Encoder Systems ===", flush=True)
    two_stage_systems = [
        ("Two-Stage: BM25 (K=50) + Cross-Encoder", bm25_gen),
        ("Two-Stage: Dense (K=50) + Cross-Encoder", dense_gen),
        ("Two-Stage: Union (K=50) + Cross-Encoder", multi_gen),
    ]

    for name, gen in two_stage_systems:
        print(f"\nEvaluating: {name}...", flush=True)
        m, res = evaluate_two_stage(name, gen, cross_encoder, benchmark.cases, chunks, top_k_cand=50)
        all_metrics.append(m)
        results_map[name] = res
        print(f"  @5  -> Recall: {m.recall_5:.4f}, nDCG: {m.ndcg_5:.4f}, MRR: {m.mrr_val:.4f}, MAP: {m.map_5:.4f}, Hit: {m.hit_5*100:.1f}%", flush=True)
        print(f"  @10 -> Recall: {m.recall_10:.4f}, nDCG: {m.ndcg_10:.4f}, MAP: {m.map_10:.4f}, Hit: {m.hit_10*100:.1f}%", flush=True)
        print(f"  Mean Latency: {m.mean_latency_ms:.2f} ms", flush=True)

    print("\n" + "=" * 95, flush=True)
    print("SUMMARY COMPARISON TABLE (@5)", flush=True)
    print("=" * 95, flush=True)
    header = f"{'System':<42} | {'Recall@5':<8} | {'nDCG@5':<8} | {'MRR':<8} | {'MAP@5':<8} | {'Hit@5':<8} | {'Latency':<8}"
    print(header, flush=True)
    print("-" * 95, flush=True)
    for m in all_metrics:
        row = (
            f"{m.name:<42} | {m.recall_5:.4f}   | {m.ndcg_5:.4f}   | {m.mrr_val:.4f}   | "
            f"{m.map_5:.4f}   | {m.hit_5:.4f}   | {m.mean_latency_ms:6.1f} ms"
        )
        print(row, flush=True)

    print("\n" + "=" * 95, flush=True)
    print("SUMMARY COMPARISON TABLE (@10)", flush=True)
    print("=" * 95, flush=True)
    header = f"{'System':<42} | {'Recall@10':<9} | {'nDCG@10':<9} | {'MAP@10':<8} | {'Hit@10':<8}"
    print(header, flush=True)
    print("-" * 95, flush=True)
    for m in all_metrics:
        row = (
            f"{m.name:<42} | {m.recall_10:.4f}    | {m.ndcg_10:.4f}    | "
            f"{m.map_10:.4f}   | {m.hit_10:.4f}"
        )
        print(row, flush=True)

    print("\n" + "=" * 95, flush=True)
    print("PAIRWISE DELTA vs. HYBRID RRF BASELINE (@5)", flush=True)
    print("=" * 95, flush=True)
    base_rrf = next(m for m in all_metrics if "Hybrid RRF" in m.name)
    best_two_stage = next(m for m in all_metrics if "Union (K=50)" in m.name)

    print(f"Recall@5 Delta:  {(best_two_stage.recall_5 - base_rrf.recall_5)*100:+.2f} percentage points ({base_rrf.recall_5*100:.1f}% -> {best_two_stage.recall_5*100:.1f}%)", flush=True)
    print(f"nDCG@5 Delta:    {(best_two_stage.ndcg_5 - base_rrf.ndcg_5):+.4f} ({base_rrf.ndcg_5:.4f} -> {best_two_stage.ndcg_5:.4f})", flush=True)
    print(f"MRR Delta:       {(best_two_stage.mrr_val - base_rrf.mrr_val):+.4f} ({base_rrf.mrr_val:.4f} -> {best_two_stage.mrr_val:.4f})", flush=True)
    print(f"MAP@5 Delta:     {(best_two_stage.map_5 - base_rrf.map_5):+.4f} ({base_rrf.map_5:.4f} -> {best_two_stage.map_5:.4f})", flush=True)
    print(f"Hit@5 Delta:     {(best_two_stage.hit_5 - base_rrf.hit_5)*100:+.2f} percentage points ({base_rrf.hit_5*100:.1f}% -> {best_two_stage.hit_5*100:.1f}%)", flush=True)

    # Query-level wins/losses: Two-Stage Union vs Hybrid RRF on nDCG@5
    rrf_res = results_map["Hybrid RRF (1:2) Single-Stage"]
    ts_res = results_map["Two-Stage: Union (K=50) + Cross-Encoder"]
    two_stage_wins = 0
    rrf_wins = 0
    ties = 0

    for i, case in enumerate(benchmark.cases):
        ndcg_rrf_i = ndcg_at_k(rrf_res[i], case, k=5)
        ndcg_ts_i = ndcg_at_k(ts_res[i], case, k=5)
        if ndcg_ts_i > ndcg_rrf_i + 1e-4:
            two_stage_wins += 1
        elif ndcg_rrf_i > ndcg_ts_i + 1e-4:
            rrf_wins += 1
        else:
            ties += 1

    print("\nQuery-Level Win/Loss Comparison on nDCG@5 (Two-Stage Union vs. Hybrid RRF):", flush=True)
    print(f"  Two-Stage Wins: {two_stage_wins} queries ({two_stage_wins/total_queries*100:.1f}%)", flush=True)
    print(f"  Hybrid RRF Wins:{rrf_wins} queries ({rrf_wins/total_queries*100:.1f}%)", flush=True)
    print(f"  Ties:           {ties} queries ({ties/total_queries*100:.1f}%)", flush=True)


if __name__ == "__main__":
    run_experiment()
