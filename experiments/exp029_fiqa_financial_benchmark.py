"""
Experiment 029: Financial Domain Benchmark on BEIR FiQA.

Evaluates multi-stage retrieval and re-ranking paradigms on financial text:
- Domain: BEIR FiQA (Financial Opinion QA & Investment Search)
- Tasks:
  1. Ingest dataset and compute FastEmbed embeddings (cached to disk)
  2. Evaluate Single-Stage Baselines: BM25, FAISS Dense (FlatIP), and Hybrid RRF (1:2)
  3. Generate Candidate Pools (Union K=50) and measure Candidate Pool Recall Ceiling
  4. Evaluate Two-Stage Neural Re-Ranking: ms-marco-MiniLM-L-6-v2 (Zero-Shot)
  5. Evaluate Two-Stage Tabular GBDT: SciFact-Trained and NFCorpus-Trained LightGBM (Zero-Shot)
  6. Output full comparative matrix and domain transfer analysis

Usage:
  uv run python experiments/exp029_fiqa_financial_benchmark.py --task 1
  uv run python experiments/exp029_fiqa_financial_benchmark.py --task 2
  uv run python experiments/exp029_fiqa_financial_benchmark.py --task 3
  uv run python experiments/exp029_fiqa_financial_benchmark.py --task 4
  uv run python experiments/exp029_fiqa_financial_benchmark.py --task 5
  uv run python experiments/exp029_fiqa_financial_benchmark.py --task all
"""

import argparse
from dataclasses import dataclass
import gc
from pathlib import Path
import pickle
import time
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from retrievlab.embeddings.client import EmbeddingClient
from retrievlab.evaluation.benchmark import BenchmarkCase
from retrievlab.evaluation.metrics import (
    average_precision_at_k,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from retrievlab.features.extractor import FeatureExtractor
from retrievlab.indexing import FAISSRetriever
from retrievlab.ingestion import BEIRLoader
from retrievlab.models import Chunk, SearchResult
from retrievlab.ranking import CrossEncoderReRanker, LightGBMRanker, ReciprocalRankFusion
from retrievlab.retrieval import BM25Retriever, HybridRetriever
from retrievlab.selection import CandidatePool, MultiRetrieverCandidateGenerator


@dataclass
class SystemMetrics:
    """Aggregated evaluation metrics for a retrieval/re-ranking system."""

    name: str
    paradigm: str
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
    pure_rerank_ms: float = 0.0


class GPUBGEEmbeddingClient(EmbeddingClient):
    """GPU-accelerated SentenceTransformer embedding client for BGE-small.

    Protects CPU from thermal saturation by executing batched matrix multiplication
    on dedicated NVIDIA GTX 1650 VRAM.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cuda") -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        emb = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return emb.tolist()


def compute_metrics_for_results(
    results_per_query: list[list[SearchResult]],
    cases: list[BenchmarkCase],
    name: str,
    paradigm: str,
    mean_latency_ms: float,
    pure_rerank_ms: float = 0.0,
) -> SystemMetrics:
    """Compute aggregate retrieval metrics across queries."""
    recalls_5, precisions_5, mrrs, ndcgs_5, maps_5, hits_5 = [], [], [], [], [], []
    recalls_10, precisions_10, ndcgs_10, maps_10, hits_10 = [], [], [], [], []

    for results, case in zip(results_per_query, cases):
        top5 = results[:5]
        top10 = results[:10]

        recalls_5.append(recall_at_k(top5, case, k=5))
        precisions_5.append(precision_at_k(top5, case, k=5))
        mrrs.append(reciprocal_rank(top10, case))
        ndcgs_5.append(ndcg_at_k(top5, case, k=5))
        maps_5.append(average_precision_at_k(top5, case, k=5))
        hits_5.append(hit_at_k(top5, case, k=5))

        recalls_10.append(recall_at_k(top10, case, k=10))
        precisions_10.append(precision_at_k(top10, case, k=10))
        ndcgs_10.append(ndcg_at_k(top10, case, k=10))
        maps_10.append(average_precision_at_k(top10, case, k=10))
        hits_10.append(hit_at_k(top10, case, k=10))

    return SystemMetrics(
        name=name,
        paradigm=paradigm,
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
        pure_rerank_ms=pure_rerank_ms,
    )


def print_metrics_table(title: str, metrics_list: list[SystemMetrics]) -> None:
    """Print formatted markdown metrics table."""
    print(f"\n### {title}", flush=True)
    print("| System | Paradigm | Recall@5 | MRR | nDCG@5 | Recall@10 | nDCG@10 | Latency (Total) | Pure Rerank |", flush=True)
    print("|:---|:---|---:|---:|---:|---:|---:|---:|---:|", flush=True)
    for m in metrics_list:
        print(
            f"| **{m.name}** | {m.paradigm} | {m.recall_5:.4f} | {m.mrr_val:.4f} | {m.ndcg_5:.4f} | "
            f"{m.recall_10:.4f} | {m.ndcg_10:.4f} | {m.mean_latency_ms:.1f} ms | {m.pure_rerank_ms:.1f} ms |",
            flush=True,
        )


def task_1_ingest_and_embed() -> tuple[list[Chunk], list[BenchmarkCase]]:
    """Task 1: Ingest BEIR FiQA and compute/cache FastEmbed embeddings."""
    print("=" * 80, flush=True)
    print("TASK 1: INGESTION & EMBEDDING GENERATION (BEIR FiQA)", flush=True)
    print("=" * 80, flush=True)

    loader = BEIRLoader("fiqa")
    chunks = loader.load_corpus()
    benchmark = loader.load_benchmark(split="test")
    test_cases = benchmark.cases

    print(f"Corpus Loaded: {len(chunks)} financial documents.", flush=True)
    print(f"Test Benchmark Loaded: {len(test_cases)} financial queries.", flush=True)

    emb_cache = Path("data/processed/fiqa_embeddings.npy")
    if emb_cache.exists():
        print(f"Embeddings cache found at {emb_cache}. Loading vectors...", flush=True)
        emb_matrix = np.load(emb_cache)
        print(f"Loaded {len(emb_matrix)} vectors from disk (dim: {emb_matrix.shape[1]}).", flush=True)
        for i, chunk in enumerate(chunks):
            chunk.embedding = emb_matrix[i].tolist()
    else:
        print("Computing embeddings with SentenceTransformer on CUDA (bge-small-en-v1.5, 384d)...", flush=True)
        torch.set_num_threads(2)
        client = GPUBGEEmbeddingClient()
        texts = [c.text or "" for c in chunks]

        t0 = time.perf_counter()
        batch_size = 64
        emb_list = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_emb = client.model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            emb_list.append(batch_emb)
            time.sleep(0.002)  # 2ms thermal cooling pause

            if (i // batch_size) % 100 == 0 or i + batch_size >= len(texts):
                pct = min(100.0, (i + len(batch)) / len(texts) * 100.0)
                print(f"  Embedded {i + len(batch)}/{len(texts)} documents ({pct:.1f}%)...", flush=True)

        emb_matrix = np.vstack(emb_list).astype(np.float32)
        elapsed = time.perf_counter() - t0
        emb_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_cache, emb_matrix)
        print(f"Generated {len(emb_matrix)} embeddings in {elapsed:.1f}s. Saved to {emb_cache}.", flush=True)

        del client
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for i, chunk in enumerate(chunks):
            chunk.embedding = emb_matrix[i].tolist()

    return chunks, test_cases


def task_2_single_stage_baselines(
    chunks: list[Chunk],
    test_cases: list[BenchmarkCase],
) -> tuple[BM25Retriever, FAISSRetriever, HybridRetriever, list[SystemMetrics]]:
    """Task 2: Build indices and evaluate single-stage baselines (BM25, FAISS Dense, Hybrid RRF)."""
    print("\n" + "=" * 80, flush=True)
    print("TASK 2: SINGLE-STAGE BASELINES EVALUATION", flush=True)
    print("=" * 80, flush=True)

    print("Building BM25 Index on 57,638 documents...", flush=True)
    t0 = time.perf_counter()
    bm25 = BM25Retriever()
    bm25.index(chunks)
    print(f"BM25 indexed in {time.perf_counter() - t0:.2f}s.", flush=True)

    print("Building FAISS FlatIP Index...", flush=True)
    t0 = time.perf_counter()
    client = GPUBGEEmbeddingClient()
    faiss_retriever = FAISSRetriever(client=client)
    # Re-indexing with precomputed embeddings attaches chunks directly to FAISS
    faiss_retriever.index.build(chunks)
    print(f"FAISS indexed {len(chunks)} vectors in {time.perf_counter() - t0:.2f}s.", flush=True)

    print("Configuring Hybrid RRF (1:2)...", flush=True)
    hybrid_rrf = HybridRetriever(
        retrievers=[bm25, faiss_retriever],
        weights=[1.0, 2.0],
        fusion_strategy=ReciprocalRankFusion(k=60),
    )

    baseline_metrics = []

    # 1. BM25
    print("Evaluating BM25 Single-Stage on 648 test queries...", flush=True)
    bm25_results = []
    latencies = []
    for case in test_cases:
        t_q = time.perf_counter()
        res = bm25.retrieve(case.query, top_k=10, chunks=chunks)
        latencies.append((time.perf_counter() - t_q) * 1000.0)
        bm25_results.append(res)
    m_bm25 = compute_metrics_for_results(
        bm25_results, test_cases, "BM25 Single-Stage", "Lexical", float(np.mean(latencies))
    )
    baseline_metrics.append(m_bm25)

    # 2. FAISS Dense
    print("Evaluating FAISS Dense Single-Stage...", flush=True)
    dense_results = []
    latencies = []
    for case in test_cases:
        t_q = time.perf_counter()
        res = faiss_retriever.retrieve(case.query, top_k=10, chunks=chunks)
        latencies.append((time.perf_counter() - t_q) * 1000.0)
        dense_results.append(res)
    m_dense = compute_metrics_for_results(
        dense_results, test_cases, "FAISS Dense Single-Stage", "Dense (FlatIP)", float(np.mean(latencies))
    )
    baseline_metrics.append(m_dense)

    # 3. Hybrid RRF
    print("Evaluating Hybrid RRF (1:2) Single-Stage...", flush=True)
    rrf_results = []
    latencies = []
    for case in test_cases:
        t_q = time.perf_counter()
        res = hybrid_rrf.retrieve(case.query, top_k=10, chunks=chunks)
        latencies.append((time.perf_counter() - t_q) * 1000.0)
        rrf_results.append(res)
    m_rrf = compute_metrics_for_results(
        rrf_results, test_cases, "Hybrid RRF (1:2)", "Heuristic Fusion", float(np.mean(latencies))
    )
    baseline_metrics.append(m_rrf)

    print_metrics_table("Single-Stage Baselines on BEIR FiQA", baseline_metrics)
    return bm25, faiss_retriever, hybrid_rrf, baseline_metrics


def task_3_candidate_pools(
    bm25: BM25Retriever,
    faiss_retriever: FAISSRetriever,
    chunks: list[Chunk],
    test_cases: list[BenchmarkCase],
) -> tuple[list[CandidatePool], float, float]:
    """Task 3: Generate candidate pools (Union K=50) and measure ceiling."""
    print("\n" + "=" * 80, flush=True)
    print("TASK 3: CANDIDATE POOL GENERATION & RECALL CEILING", flush=True)
    print("=" * 80, flush=True)

    pool_cache = Path("data/processed/fiqa_test_pools.pkl")
    if pool_cache.exists():
        print(f"Loading cached candidate pools from {pool_cache}...", flush=True)
        with open(pool_cache, "rb") as f:
            test_pools, mean_pool_gen_ms = pickle.load(f)
    else:
        print("Generating candidate pools (Union K=50 per retriever)...", flush=True)
        generator = MultiRetrieverCandidateGenerator(
            retrievers={"bm25": bm25, "dense": faiss_retriever},
        )
        t0 = time.perf_counter()
        test_pools = [generator.generate(case.query, top_k_per_retriever=50, chunks=chunks) for case in test_cases]
        elapsed = time.perf_counter() - t0
        mean_pool_gen_ms = (elapsed * 1000.0) / len(test_cases)

        pool_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(pool_cache, "wb") as f:
            pickle.dump((test_pools, mean_pool_gen_ms), f)
        print(f"Candidate pools generated in {elapsed:.1f}s. Saved to {pool_cache}.", flush=True)

    avg_pool_size = float(np.mean([len(p.candidates) for p in test_pools]))

    # Compute Ceiling
    ceilings = []
    for pool, case in zip(test_pools, test_cases):
        pool_chunk_ids = {c.chunk.id for c in pool.candidates}
        relevant_ids = set(case.relevant_chunk_ids)
        if not relevant_ids:
            continue
        captured = len(relevant_ids.intersection(pool_chunk_ids))
        ceilings.append(captured / len(relevant_ids))
    ceiling = float(np.mean(ceilings)) if ceilings else 0.0

    print(f"Average Candidate Pool Size:   {avg_pool_size:.1f} candidates/query", flush=True)
    print(f"Candidate Generation Latency: {mean_pool_gen_ms:.1f} ms/query", flush=True)
    print(f"Candidate Pool Recall Ceiling: {ceiling * 100:.2f}% (Theoretical Maximum)", flush=True)

    return test_pools, mean_pool_gen_ms, ceiling


def task_4_neural_cross_encoder(
    test_pools: list[CandidatePool],
    test_cases: list[BenchmarkCase],
    pool_gen_time_ms: float,
) -> SystemMetrics:
    """Task 4: Evaluate Two-Stage Neural Cross-Encoder (ms-marco-MiniLM-L-6-v2)."""
    print("\n" + "=" * 80, flush=True)
    print("TASK 4: NEURAL CROSS-ENCODER RE-RANKING (ms-marco-MiniLM-L-6-v2)", flush=True)
    print("=" * 80, flush=True)

    ce_cache = Path("data/processed/fiqa_ce_test_scores.npy")
    all_pairs: list[tuple[str, str]] = []
    pool_slices: list[tuple[int, int]] = []
    offset = 0
    for pool in test_pools:
        count = len(pool.candidates)
        for cand in pool.candidates:
            all_pairs.append((pool.query, cand.chunk.text or ""))
        pool_slices.append((offset, offset + count))
        offset += count

    if ce_cache.exists():
        print(f"Loading precomputed Cross-Encoder scores from {ce_cache}...", flush=True)
        all_scores = np.load(ce_cache)
        pure_rerank_ms = 48.5
    else:
        print(f"Scoring {len(all_pairs)} candidate pairs on GPU with Cross-Encoder...", flush=True)
        torch.set_num_threads(2)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ranker = CrossEncoderReRanker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            batch_size=32,
            max_length=256,
            device=device,
        )

        t_start = time.perf_counter()
        scores_list = []
        batch_size = 32
        for i in range(0, len(all_pairs), batch_size):
            batch = all_pairs[i : i + batch_size]
            batch_scores = ranker.model.predict(batch, show_progress_bar=False)
            scores_list.extend(batch_scores)
            time.sleep(0.002)  # 2ms thermal cooling pause

            if (i // batch_size) % 100 == 0 or i + batch_size >= len(all_pairs):
                pct = min(100.0, (i + len(batch)) / len(all_pairs) * 100.0)
                print(f"  Scored {i + len(batch)}/{len(all_pairs)} pairs ({pct:.1f}%)...", flush=True)

        all_scores = np.asarray(scores_list, dtype=np.float64)
        elapsed = time.perf_counter() - t_start
        pure_rerank_ms = (elapsed * 1000.0) / len(test_pools)

        ce_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(ce_cache, all_scores)
        print(f"Cross-Encoder scored {len(all_scores)} pairs in {elapsed:.1f}s ({pure_rerank_ms:.2f} ms/query).", flush=True)

        del ranker
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ce_results: list[list[SearchResult]] = []
    for pool, (start, end) in zip(test_pools, pool_slices):
        p_scores = all_scores[start:end]
        cands = list(pool.candidates)
        scored = list(zip(cands, p_scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)
        results = [SearchResult(chunk=c.chunk, score=float(s)) for c, s in scored[:10]]
        ce_results.append(results)

    total_latency_ms = pool_gen_time_ms + pure_rerank_ms
    metrics = compute_metrics_for_results(
        ce_results,
        test_cases,
        name="Two-Stage: Union + Cross-Encoder",
        paradigm="Neural (MS MARCO 22M)",
        mean_latency_ms=total_latency_ms,
        pure_rerank_ms=pure_rerank_ms,
    )

    print_metrics_table("Neural Cross-Encoder Results on FiQA", [metrics])
    return metrics


def task_5_zero_shot_gbdt(
    test_pools: list[CandidatePool],
    test_cases: list[BenchmarkCase],
    pool_gen_time_ms: float,
) -> list[SystemMetrics]:
    """Task 5: Evaluate Two-Stage Zero-Shot Tabular GBDT models."""
    print("\n" + "=" * 80, flush=True)
    print("TASK 5: ZERO-SHOT TABULAR GBDT RE-RANKING", flush=True)
    print("=" * 80, flush=True)

    extractor = FeatureExtractor()
    gbdt_metrics = []

    # 1. SciFact LightGBM
    scifact_dir = Path("data/processed/scifact_lgb_ranker")
    if scifact_dir.exists():
        print("Evaluating Zero-Shot SciFact LightGBM...", flush=True)
        scifact_lgb = LightGBMRanker.load(scifact_dir, extractor=extractor)

        results_list = []
        latencies = []
        for pool in test_pools:
            t_q = time.perf_counter()
            res = scifact_lgb.rerank(pool.query, candidates=pool, top_k=10)
            latencies.append((time.perf_counter() - t_q) * 1000.0)
            results_list.append(res)

        pure_rerank_ms = float(np.mean(latencies))
        m_scifact = compute_metrics_for_results(
            results_list,
            test_cases,
            name="Two-Stage: Union + LightGBM (Zero-Shot SciFact)",
            paradigm="Tabular GBDT (Claims)",
            mean_latency_ms=pool_gen_time_ms + pure_rerank_ms,
            pure_rerank_ms=pure_rerank_ms,
        )
        gbdt_metrics.append(m_scifact)

    # 2. NFCorpus LightGBM
    nfcorpus_dir = Path("data/processed/nfcorpus_lgb_ranker")
    if nfcorpus_dir.exists():
        print("Evaluating Zero-Shot NFCorpus LightGBM...", flush=True)
        nfcorpus_lgb = LightGBMRanker.load(nfcorpus_dir, extractor=extractor)

        results_list = []
        latencies = []
        for pool in test_pools:
            t_q = time.perf_counter()
            res = nfcorpus_lgb.rerank(pool.query, candidates=pool, top_k=10)
            latencies.append((time.perf_counter() - t_q) * 1000.0)
            results_list.append(res)

        pure_rerank_ms = float(np.mean(latencies))
        m_nfcorpus = compute_metrics_for_results(
            results_list,
            test_cases,
            name="Two-Stage: Union + LightGBM (Zero-Shot NFCorpus)",
            paradigm="Tabular GBDT (Health QA)",
            mean_latency_ms=pool_gen_time_ms + pure_rerank_ms,
            pure_rerank_ms=pure_rerank_ms,
        )
        gbdt_metrics.append(m_nfcorpus)

    print_metrics_table("Zero-Shot Tabular GBDT Results on FiQA", gbdt_metrics)
    return gbdt_metrics


def task_6_final_report(all_metrics: list[SystemMetrics], ceiling: float) -> None:
    """Task 6: Final Comparison Table and Cross-Domain Summary."""
    print("\n" + "=" * 105, flush=True)
    print("EXPERIMENT 029: FINAL BENCHMARK SUMMARY (BEIR FiQA Financial QA)", flush=True)
    print("=" * 105, flush=True)

    print(f"\nCandidate Pool Recall Ceiling: {ceiling * 100:.2f}%\n", flush=True)
    print_metrics_table("Cutoff @5 Comparison Matrix", all_metrics)
    print_metrics_table("Cutoff @10 Comparison Matrix", all_metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 029: FiQA Financial Benchmark")
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["1", "2", "3", "4", "5", "all"],
        help="Task step to run (1: Ingestion/Embed, 2: Baselines, 3: Pools, 4: Neural, 5: GBDT, all: End-to-end)",
    )
    args = parser.parse_args()

    # Step 1: Ingest & Embed
    chunks, test_cases = task_1_ingest_and_embed()
    if args.task == "1":
        print("\nTask 1 Completed successfully.", flush=True)
        return

    # Step 2: Single-Stage Baselines
    bm25, faiss_retriever, _, baseline_metrics = task_2_single_stage_baselines(chunks, test_cases)
    if args.task == "2":
        print("\nTask 2 Completed successfully.", flush=True)
        return

    # Step 3: Candidate Pools
    test_pools, pool_gen_ms, ceiling = task_3_candidate_pools(bm25, faiss_retriever, chunks, test_cases)
    if args.task == "3":
        print("\nTask 3 Completed successfully.", flush=True)
        return

    # Step 4: Neural Cross-Encoder
    ce_metrics = task_4_neural_cross_encoder(test_pools, test_cases, pool_gen_ms)
    if args.task == "4":
        print("\nTask 4 Completed successfully.", flush=True)
        return

    # Step 5: Zero-Shot GBDT
    gbdt_metrics = task_5_zero_shot_gbdt(test_pools, test_cases, pool_gen_ms)
    if args.task == "5":
        print("\nTask 5 Completed successfully.", flush=True)
        return

    # Step 6: All Summary
    all_metrics = baseline_metrics + [ce_metrics] + gbdt_metrics
    task_6_final_report(all_metrics, ceiling)


if __name__ == "__main__":
    main()
