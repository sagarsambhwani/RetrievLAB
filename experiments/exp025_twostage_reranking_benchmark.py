"""
Experiment 025: End-to-End Two-Stage Comparative Benchmark & Latency Profiling.

Compares three retrieval paradigms on BEIR SciFact:
1. Single-Stage Baselines: BM25, FAISS Dense (FlatIP), and Hybrid RRF (1:2)
2. Neural Two-Stage: Union Candidates (K=50) + Cross-Encoder (MS MARCO MiniLM)
3. Tabular GBDT Two-Stage: Union Candidates (K=50) + LightGBM LambdaMART (16 features)

Evaluates quality metrics (Recall@K, nDCG@K, MRR), per-query latency, and
feature gain importance attribution.
"""

from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np

from retrievlab.embeddings.fastembed import FastEmbedClient
from retrievlab.evaluation.benchmark import BenchmarkCase
from retrievlab.evaluation.metrics import (
    average_precision_at_k,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from retrievlab.features.dataset import LTRDataset, build_ltr_dataset
from retrievlab.features.extractor import FeatureExtractor
from retrievlab.indexing import FAISSRetriever
from retrievlab.ingestion import BEIRLoader
from retrievlab.models import Chunk, SearchResult
from retrievlab.ranking import CrossEncoderReRanker, LightGBMRanker
from retrievlab.retrieval import BM25Retriever, HybridRetriever
from retrievlab.selection import MultiRetrieverCandidateGenerator


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
    pure_rerank_ms: float = 0.0


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
    benchmark_cases: list[BenchmarkCase],
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


def evaluate_cross_encoder_stage(
    generator: MultiRetrieverCandidateGenerator,
    cross_encoder: CrossEncoderReRanker,
    benchmark_cases: list[BenchmarkCase],
    chunks: list[Chunk],
) -> tuple[SystemMetrics, list[list[SearchResult]]]:
    """Evaluate two-stage Union + Cross-Encoder re-ranker with bulk GPU batching."""
    print("Generating candidate pools for Cross-Encoder...", flush=True)
    t_start = time.perf_counter()

    pools = []
    for case in benchmark_cases:
        pool = generator.generate(case.query, chunks=chunks)
        pools.append(pool)

    t_pools = time.perf_counter()
    pool_gen_time_ms = (t_pools - t_start) * 1000.0 / len(benchmark_cases)

    # Prepare pairs for vectorized inference
    all_pairs: list[tuple[str, str]] = []
    pool_slices: list[tuple[int, int]] = []
    offset = 0
    for pool in pools:
        count = len(pool.candidates)
        for cand in pool.candidates:
            all_pairs.append((pool.query, cand.chunk.text or ""))
        pool_slices.append((offset, offset + count))
        offset += count

    print(f"Scoring {len(all_pairs)} candidate pairs in bulk with Cross-Encoder...", flush=True)
    t_nn_start = time.perf_counter()
    all_scores = cross_encoder.model.predict(
        all_pairs,
        batch_size=cross_encoder.batch_size,
        show_progress_bar=False,
    )
    t_nn_end = time.perf_counter()
    pure_rerank_ms = (t_nn_end - t_nn_start) * 1000.0 / len(benchmark_cases)
    total_latency_ms = pool_gen_time_ms + pure_rerank_ms

    all_results: list[list[SearchResult]] = []
    recalls_5, precisions_5, mrrs, ndcgs_5, maps_5, hits_5 = [], [], [], [], [], []
    recalls_10, precisions_10, ndcgs_10, maps_10, hits_10 = [], [], [], [], []

    for pool, (start, end), case in zip(pools, pool_slices, benchmark_cases):
        pool_scores = all_scores[start:end]
        cands = list(pool.candidates)
        scored = list(zip(cands, pool_scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)
        results = [SearchResult(chunk=c.chunk, score=float(s)) for c, s in scored[:10]]
        all_results.append(results)

        recalls_5.append(recall_at_k(results, case, k=5))
        precisions_5.append(precision_at_k(results, case, k=5))
        mrrs.append(reciprocal_rank(results, case))
        ndcgs_5.append(ndcg_at_k(results, case, k=5))
        maps_5.append(average_precision_at_k(results, case, k=5))
        hits_5.append(hit_at_k(results, case, k=5))

        recalls_10.append(recall_at_k(results, case, k=10))
        precisions_10.append(precision_at_k(results, case, k=10))
        ndcgs_10.append(ndcg_at_k(results, case, k=10))
        maps_10.append(average_precision_at_k(results, case, k=10))
        hits_10.append(hit_at_k(results, case, k=10))

    metrics = SystemMetrics(
        name="Two-Stage: Union (K=50) + Cross-Encoder",
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
        mean_latency_ms=float(total_latency_ms),
        pure_rerank_ms=float(pure_rerank_ms),
    )
    return metrics, all_results


def evaluate_lightgbm_stage(
    generator: MultiRetrieverCandidateGenerator,
    lgb_ranker: LightGBMRanker,
    benchmark_cases: list[BenchmarkCase],
    chunks: list[Chunk],
) -> tuple[SystemMetrics, list[list[SearchResult]]]:
    """Evaluate two-stage Union + LightGBM LambdaMART re-ranker."""
    print("Evaluating Two-Stage: Union + LightGBM LambdaMART...", flush=True)
    all_results: list[list[SearchResult]] = []
    recalls_5, precisions_5, mrrs, ndcgs_5, maps_5, hits_5 = [], [], [], [], [], []
    recalls_10, precisions_10, ndcgs_10, maps_10, hits_10 = [], [], [], [], []
    total_latencies = []
    pure_rerank_latencies = []

    for case in benchmark_cases:
        t0 = time.perf_counter()
        pool = generator.generate(case.query, chunks=chunks)
        t_gen = time.perf_counter()

        results = lgb_ranker.rerank(case.query, pool, top_k=10)
        t_end = time.perf_counter()

        gen_ms = (t_gen - t0) * 1000.0
        rerank_ms = (t_end - t_gen) * 1000.0

        pure_rerank_latencies.append(rerank_ms)
        total_latencies.append(gen_ms + rerank_ms)
        all_results.append(results)

        recalls_5.append(recall_at_k(results, case, k=5))
        precisions_5.append(precision_at_k(results, case, k=5))
        mrrs.append(reciprocal_rank(results, case))
        ndcgs_5.append(ndcg_at_k(results, case, k=5))
        maps_5.append(average_precision_at_k(results, case, k=5))
        hits_5.append(hit_at_k(results, case, k=5))

        recalls_10.append(recall_at_k(results, case, k=10))
        precisions_10.append(precision_at_k(results, case, k=10))
        ndcgs_10.append(ndcg_at_k(results, case, k=10))
        maps_10.append(average_precision_at_k(results, case, k=10))
        hits_10.append(hit_at_k(results, case, k=10))

    metrics = SystemMetrics(
        name="Two-Stage: Union (K=50) + LightGBM",
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
        mean_latency_ms=float(np.mean(total_latencies)),
        pure_rerank_ms=float(np.mean(pure_rerank_latencies)),
    )
    return metrics, all_results


def get_or_build_train_dataset(
    generator: MultiRetrieverCandidateGenerator,
    extractor: FeatureExtractor,
    train_cases: list[BenchmarkCase],
    chunks: list[Chunk],
    cache_path: Path,
) -> LTRDataset:
    """Load or generate query-grouped LTR training dataset."""
    if cache_path.exists():
        print(f"Loading cached LTR training dataset from {cache_path}...", flush=True)
        data = np.load(cache_path, allow_pickle=True)
        return LTRDataset(
            features=data["features"],
            labels=data["labels"],
            group_sizes=data["group_sizes"].tolist(),
            feature_names=data["feature_names"].tolist(),
        )

    print(f"Generating candidate pools for {len(train_cases)} train queries...", flush=True)
    t0 = time.perf_counter()
    train_pools = []
    for i, case in enumerate(train_cases):
        if (i + 1) % 100 == 0 or i == len(train_cases) - 1:
            print(f"  Processed {i + 1}/{len(train_cases)} train query pools...", flush=True)
        pool = generator.generate(case.query, chunks=chunks)
        train_pools.append(pool)

    print(f"Candidate pools generated in {(time.perf_counter() - t0):.1f}s. Building LTRDataset...", flush=True)
    t1 = time.perf_counter()
    dataset = build_ltr_dataset(train_pools, train_cases, extractor=extractor)
    print(f"LTRDataset built in {(time.perf_counter() - t1):.1f}s: {len(dataset)} rows, {dataset.num_features} features.", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=dataset.features,
        labels=dataset.labels,
        group_sizes=np.array(dataset.group_sizes, dtype=np.int32),
        feature_names=np.array(dataset.feature_names),
    )
    print(f"Cached LTR training dataset to {cache_path}.", flush=True)
    return dataset


def run_experiment() -> None:
    print("=" * 80, flush=True)
    print("EXPERIMENT 025: END-TO-END TWO-STAGE COMPARATIVE BENCHMARK", flush=True)
    print("=" * 80, flush=True)

    # Step 1: Load SciFact Benchmark & Chunks
    loader = BEIRLoader("scifact")
    chunks = loader.load_corpus()
    test_benchmark = loader.load_benchmark(split="test")
    train_benchmark = loader.load_benchmark(split="train")
    test_cases = test_benchmark.cases
    train_cases = train_benchmark.cases

    print(f"Corpus: {len(chunks)} documents.", flush=True)
    print(f"Test Queries: {len(test_cases)} | Train Queries: {len(train_cases)}", flush=True)

    # Step 2: Load Embeddings & Initialize Retrievers
    emb_cache = Path("data/processed/scifact_embeddings.npy")
    chunks = load_cached_embeddings(chunks, emb_cache)

    print("Building BM25 index...", flush=True)
    bm25 = BM25Retriever()
    bm25.index(chunks)

    print("Building FAISS index...", flush=True)
    client = FastEmbedClient()
    faiss_retriever = FAISSRetriever(client=client)
    faiss_retriever.index(chunks)

    print("Initializing Hybrid RRF (1:2)...", flush=True)
    hybrid_rrf = HybridRetriever(
        bm25,
        faiss_retriever,
        strategy="rrf",
        weights={"bm25": 1.0, "dense": 2.0},
    )

    generator = MultiRetrieverCandidateGenerator(
        retrievers=[bm25, faiss_retriever],
        top_k_per_retriever=50,
    )
    extractor = FeatureExtractor()

    # Step 3: Train LightGBM LambdaMART on Train Split
    train_cache = Path("data/processed/scifact_train_ltr.npz")
    train_dataset = get_or_build_train_dataset(
        generator=generator,
        extractor=extractor,
        train_cases=train_cases,
        chunks=chunks,
        cache_path=train_cache,
    )

    print("\nFitting LightGBMRanker (LambdaMART, objective='lambdarank')...", flush=True)
    t_fit_start = time.perf_counter()
    lgb_ranker = LightGBMRanker(
        model_params={
            "n_estimators": 100,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 10,
            "random_state": 42,
        },
        extractor=extractor,
    )
    lgb_ranker.fit(train_dataset)
    t_fit_end = time.perf_counter()
    print(f"LightGBM fitting completed in {(t_fit_end - t_fit_start):.2f} seconds.", flush=True)

    model_dir = Path("data/processed/scifact_lgb_ranker")
    lgb_ranker.save(model_dir)

    # Feature Importance Diagnostic
    gain_importances = lgb_ranker.get_feature_importances(importance_type="gain")
    print("\nLightGBM Feature Importances (Gain):", flush=True)
    for feat_name, gain_val in gain_importances.items():
        print(f"  {feat_name:<24}: {gain_val:>10.2f}", flush=True)

    # Step 4: Evaluate Single-Stage Baselines
    print("\nEvaluating Single-Stage Baselines...", flush=True)
    bm25_metrics, bm25_res = evaluate_single_stage("BM25 Single-Stage", bm25, test_cases, chunks)
    dense_metrics, dense_res = evaluate_single_stage("FAISS Dense Single-Stage", faiss_retriever, test_cases, chunks)
    rrf_metrics, rrf_res = evaluate_single_stage("Hybrid RRF (1:2) Single-Stage", hybrid_rrf, test_cases, chunks)

    # Step 5: Evaluate Two-Stage Neural Cross-Encoder
    print("\nEvaluating Two-Stage Neural Cross-Encoder (MS MARCO MiniLM)...", flush=True)
    ce_ranker = CrossEncoderReRanker(batch_size=64)
    ce_metrics, ce_res = evaluate_cross_encoder_stage(generator, ce_ranker, test_cases, chunks)

    # Step 6: Evaluate Two-Stage LightGBM LambdaMART
    print("\nEvaluating Two-Stage LightGBM LambdaMART...", flush=True)
    lgb_metrics, lgb_res = evaluate_lightgbm_stage(generator, lgb_ranker, test_cases, chunks)

    systems = [bm25_metrics, dense_metrics, rrf_metrics, ce_metrics, lgb_metrics]

    # Step 7: Print Comparative Results Matrix
    print("\n" + "=" * 105, flush=True)
    print("EMPIRICAL BENCHMARK RESULTS SUMMARY (BEIR SciFact, N=300)", flush=True)
    print("=" * 105, flush=True)

    header_5 = f"{'System':<40} | {'Recall@5':>8} | {'Prec@5':>7} | {'MRR':>7} | {'nDCG@5':>7} | {'Latency':>10}"
    print("\n--- PERFORMANCE AT CUTOFF @5 ---", flush=True)
    print(header_5, flush=True)
    print("-" * len(header_5), flush=True)
    for s in systems:
        lat_str = f"{s.mean_latency_ms:.1f} ms"
        print(
            f"{s.name:<40} | {s.recall_5:>8.4f} | {s.precision_5:>7.4f} | "
            f"{s.mrr_val:>7.4f} | {s.ndcg_5:>7.4f} | {lat_str:>10}",
            flush=True,
        )

    header_10 = f"{'System':<40} | {'Recall@10':>9} | {'Prec@10':>8} | {'MRR':>7} | {'nDCG@10':>8} | {'Latency':>10}"
    print("\n--- PERFORMANCE AT CUTOFF @10 ---", flush=True)
    print(header_10, flush=True)
    print("-" * len(header_10), flush=True)
    for s in systems:
        lat_str = f"{s.mean_latency_ms:.1f} ms"
        print(
            f"{s.name:<40} | {s.recall_10:>9.4f} | {s.precision_10:>8.4f} | "
            f"{s.mrr_val:>7.4f} | {s.ndcg_10:>8.4f} | {lat_str:>10}",
            flush=True,
        )

    # Step 8: Query-by-Query Diagnostic Breakdown (LightGBM vs RRF and vs Cross-Encoder)
    print("\n--- QUERY-LEVEL WIN/LOSS ANALYSIS ON nDCG@5 ---", flush=True)
    total_q = len(test_cases)

    # LightGBM vs Hybrid RRF
    lgb_wins_rrf = 0
    rrf_wins_lgb = 0
    ties_rrf = 0
    for i, case in enumerate(test_cases):
        s_lgb = ndcg_at_k(lgb_res[i], case, k=5)
        s_rrf = ndcg_at_k(rrf_res[i], case, k=5)
        if s_lgb > s_rrf + 1e-4:
            lgb_wins_rrf += 1
        elif s_rrf > s_lgb + 1e-4:
            rrf_wins_lgb += 1
        else:
            ties_rrf += 1

    print("LightGBM vs. Hybrid RRF (1:2):", flush=True)
    print(f"  LightGBM Wins: {lgb_wins_rrf} ({lgb_wins_rrf / total_q * 100:.1f}%)", flush=True)
    print(f"  Hybrid RRF Wins: {rrf_wins_lgb} ({rrf_wins_lgb / total_q * 100:.1f}%)", flush=True)
    print(f"  Ties:            {ties_rrf} ({ties_rrf / total_q * 100:.1f}%)", flush=True)

    # LightGBM vs Cross-Encoder
    lgb_wins_ce = 0
    ce_wins_lgb = 0
    ties_ce = 0
    for i, case in enumerate(test_cases):
        s_lgb = ndcg_at_k(lgb_res[i], case, k=5)
        s_ce = ndcg_at_k(ce_res[i], case, k=5)
        if s_lgb > s_ce + 1e-4:
            lgb_wins_ce += 1
        elif s_ce > s_lgb + 1e-4:
            ce_wins_lgb += 1
        else:
            ties_ce += 1

    print("\nLightGBM vs. Cross-Encoder (MiniLM):", flush=True)
    print(f"  LightGBM Wins:      {lgb_wins_ce} ({lgb_wins_ce / total_q * 100:.1f}%)", flush=True)
    print(f"  Cross-Encoder Wins: {ce_wins_lgb} ({ce_wins_lgb / total_q * 100:.1f}%)", flush=True)
    print(f"  Ties:               {ties_ce} ({ties_ce / total_q * 100:.1f}%)", flush=True)

    # Latency Breakdown
    print("\n--- LATENCY & EFFICIENCY PROFILING ---", flush=True)
    print(f"Stage 1 Candidate Generation: ~{dense_metrics.mean_latency_ms + bm25_metrics.mean_latency_ms:.1f} ms/query", flush=True)
    print(f"Pure Cross-Encoder Re-Rank:   {ce_metrics.pure_rerank_ms:.2f} ms/query (GPU CUDA)", flush=True)
    print(f"Pure LightGBM Re-Rank:        {lgb_metrics.pure_rerank_ms:.2f} ms/query (CPU, 16 features)", flush=True)
    speedup = ce_metrics.pure_rerank_ms / max(lgb_metrics.pure_rerank_ms, 0.001)
    print(f"LightGBM Re-Ranking Speedup:  {speedup:.1f}x faster than Neural Cross-Encoder", flush=True)


if __name__ == "__main__":
    run_experiment()
