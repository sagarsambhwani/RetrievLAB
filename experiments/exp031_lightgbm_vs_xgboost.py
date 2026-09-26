"""
Experiment 031: GBDT Head-to-Head Benchmark — LightGBM vs. XGBoost.

Evaluates tabular Learning-to-Rank re-ranking across GBDT frameworks on BEIR SciFact:
Part 1: Head-to-Head Architectural Comparison (Full N=809 Queries)
  - LightGBMRanker (LambdaMART, leaf-wise growth)
  - XGBoostRanker (rank:ndcg, depth-wise growth)
  - XGBoostRanker (rank:pairwise, depth-wise growth)
  - Feature split importance comparison across frameworks

Part 2: Sample Efficiency Head-to-Head
  - LightGBM vs. XGBoost under data starvation: N in [25, 100, 809] queries
  - Tests whether XGBoost's depth-wise splits overfit more or less under small N
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
from retrievlab.features.dataset import LTRDataset
from retrievlab.features.extractor import FeatureExtractor
from retrievlab.indexing import FAISSRetriever
from retrievlab.ingestion import BEIRLoader
from retrievlab.models import Chunk, SearchResult
from retrievlab.ranking.interface import ReRanker
from retrievlab.ranking.lightgbm import LightGBMRanker
from retrievlab.ranking.xgboost import XGBoostRanker
from retrievlab.retrieval import BM25Retriever
from retrievlab.selection.generator import MultiRetrieverCandidateGenerator
from retrievlab.selection.candidate import CandidatePool


@dataclass
class SystemMetrics:
    name: str
    condition: str
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
    fit_time_s: float
    mean_latency_ms: float
    pure_rerank_ms: float = 0.0


def compute_metrics_for_results(
    results_per_query: list[list[SearchResult]],
    cases: list[BenchmarkCase],
    name: str,
    condition: str,
    fit_time_s: float,
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
        condition=condition,
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
        fit_time_s=fit_time_s,
        mean_latency_ms=mean_latency_ms,
        pure_rerank_ms=pure_rerank_ms,
    )


def load_cached_embeddings(chunks: list[Chunk], cache_path: Path) -> list[Chunk]:
    """Load precomputed embedding vectors from disk into chunks."""
    print(f"Loading cached embeddings from {cache_path}...", flush=True)
    emb_matrix = np.load(cache_path)
    for i, chunk in enumerate(chunks):
        chunk.embedding = emb_matrix[i].tolist()
    return chunks


def evaluate_ranker_on_pools(
    ranker: ReRanker,
    pools: list[CandidatePool],
    cases: list[BenchmarkCase],
    name: str,
    condition: str,
    fit_time_s: float,
    pool_gen_ms: float,
) -> SystemMetrics:
    """Evaluate a re-ranker on precomputed candidate pools."""
    results_list = []
    latencies = []
    for pool in pools:
        t0 = time.perf_counter()
        res = ranker.rerank(pool.query, candidates=pool, top_k=10)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        results_list.append(res)

    pure_rerank_ms = float(np.mean(latencies))
    total_ms = pool_gen_ms + pure_rerank_ms
    return compute_metrics_for_results(
        results_list,
        cases,
        name=name,
        condition=condition,
        fit_time_s=fit_time_s,
        mean_latency_ms=total_ms,
        pure_rerank_ms=pure_rerank_ms,
    )


def slice_ltr_dataset(
    full_dataset: LTRDataset,
    num_queries: int | None = None,
) -> LTRDataset:
    """Slice an LTRDataset by number of training queries."""
    features = full_dataset.features
    labels = full_dataset.labels
    group_sizes = list(full_dataset.group_sizes)
    all_names = list(full_dataset.feature_names)

    if num_queries is not None and num_queries < len(group_sizes):
        active_groups = group_sizes[:num_queries]
        item_count = sum(active_groups)
        features = features[:item_count]
        labels = labels[:item_count]
        group_sizes = active_groups

    return LTRDataset(
        features=np.ascontiguousarray(features),
        labels=np.ascontiguousarray(labels),
        group_sizes=group_sizes,
        feature_names=all_names,
    )


def run_experiment() -> None:
    print("=" * 95, flush=True)
    print("EXPERIMENT 031: GBDT HEAD-TO-HEAD BENCHMARK — LIGHTGBM VS. XGBOOST", flush=True)
    print("Domain: BEIR SciFact (Biomedical Claim Verification, N=300 test queries)", flush=True)
    print("=" * 95, flush=True)

    # 1. Ingest SciFact & Build Candidate Pools
    loader = BEIRLoader("scifact")
    chunks = loader.load_corpus()
    benchmark = loader.load_benchmark(split="test")
    test_cases = benchmark.cases
    print(f"Loaded SciFact: {len(chunks)} documents, {len(test_cases)} test queries.", flush=True)

    cache_path = Path("data/processed/scifact_embeddings.npy")
    chunks = load_cached_embeddings(chunks, cache_path)

    print("Building BM25 and FAISS dense retrievers...", flush=True)
    bm25 = BM25Retriever()
    bm25.index(chunks)

    client = FastEmbedClient()
    faiss_retriever = FAISSRetriever(client=client)

    generator = MultiRetrieverCandidateGenerator(
        retrievers={"bm25": bm25, "dense": faiss_retriever},
    )

    print("Generating test candidate pools (Union K=50)...", flush=True)
    t0 = time.perf_counter()
    test_pools = [generator.generate(case.query, top_k_per_retriever=50, chunks=chunks) for case in test_cases]
    pool_gen_ms = (time.perf_counter() - t0) * 1000.0 / len(test_cases)
    avg_pool_size = float(np.mean([len(p.candidates) for p in test_pools]))
    print(
        f"Candidate pools generated: {len(test_pools)} pools (avg size {avg_pool_size:.1f}, {pool_gen_ms:.1f} ms/query).",
        flush=True,
    )

    # 2. Evaluate Single-Stage Baselines
    print("\nEvaluating Single-Stage Baselines for context...", flush=True)
    # BM25 baseline
    t_b0 = time.perf_counter()
    bm25_res = [bm25.retrieve(c.query, top_k=10, chunks=chunks) for c in test_cases]
    bm25_ms = (time.perf_counter() - t_b0) * 1000.0 / len(test_cases)
    m_bm25 = compute_metrics_for_results(bm25_res, test_cases, "BM25 Single-Stage", "Lexical baseline", 0.0, bm25_ms)

    # Dense baseline
    t_d0 = time.perf_counter()
    dense_res = [faiss_retriever.retrieve(c.query, top_k=10, chunks=chunks) for c in test_cases]
    dense_ms = (time.perf_counter() - t_d0) * 1000.0 / len(test_cases)
    m_dense = compute_metrics_for_results(dense_res, test_cases, "FAISS Dense Single-Stage", "bge-small-en-v1.5", 0.0, dense_ms)

    # 3. Load Precomputed Training Data
    train_cache = Path("data/processed/scifact_ltr_train.npz")
    print(f"\nLoading pre-cached LTR training dataset from {train_cache}...", flush=True)
    raw_train = np.load(train_cache)
    full_train_dataset = LTRDataset(
        features=raw_train["features"],
        labels=raw_train["labels"],
        group_sizes=list(raw_train["group_sizes"]),
        feature_names=list(raw_train["feature_names"]),
    )
    print(f"Loaded {full_train_dataset.features.shape[0]} pairs across {full_train_dataset.num_queries} training queries.", flush=True)

    # =========================================================================
    # PART 1: HEAD-TO-HEAD ARCHITECTURAL COMPARISON (N=809 Queries)
    # =========================================================================
    print("\n" + "=" * 95, flush=True)
    print("PART 1: HEAD-TO-HEAD ARCHITECTURAL COMPARISON (Full 809 Training Queries)", flush=True)
    print("=" * 95, flush=True)

    part1_systems: list[SystemMetrics] = [m_bm25, m_dense]
    feature_extractor = FeatureExtractor()

    # 1. LightGBMRanker (LambdaMART)
    print("Training LightGBM (objective=lambdarank, n_estimators=100)...", flush=True)
    lgb_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [5, 10],
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 10,
        "verbosity": -1,
        "random_state": 42,
    }
    t_lgb_fit = time.perf_counter()
    lgb_ranker = LightGBMRanker(model_params=lgb_params, extractor=feature_extractor)
    lgb_ranker.fit(full_train_dataset)
    lgb_fit_s = time.perf_counter() - t_lgb_fit
    m_lgb = evaluate_ranker_on_pools(
        ranker=lgb_ranker,
        pools=test_pools,
        cases=test_cases,
        name="LightGBM LambdaMART",
        condition="leaf-wise (num_leaves=31)",
        fit_time_s=lgb_fit_s,
        pool_gen_ms=pool_gen_ms,
    )
    part1_systems.append(m_lgb)
    print(f"  -> LightGBM fit in {lgb_fit_s:.2f}s | nDCG@10: {m_lgb.ndcg_10:.4f} | Latency: {m_lgb.pure_rerank_ms:.1f} ms", flush=True)

    # 2. XGBoostRanker (rank:ndcg)
    print("Training XGBoost (objective=rank:ndcg, n_estimators=100, max_depth=6)...", flush=True)
    xgb_ndcg_params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@10",
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }
    t_xgb_ndcg_fit = time.perf_counter()
    xgb_ndcg_ranker = XGBoostRanker(model_params=xgb_ndcg_params, extractor=feature_extractor)
    xgb_ndcg_ranker.fit(full_train_dataset)
    xgb_ndcg_fit_s = time.perf_counter() - t_xgb_ndcg_fit
    m_xgb_ndcg = evaluate_ranker_on_pools(
        ranker=xgb_ndcg_ranker,
        pools=test_pools,
        cases=test_cases,
        name="XGBoost (rank:ndcg)",
        condition="depth-wise (max_depth=6)",
        fit_time_s=xgb_ndcg_fit_s,
        pool_gen_ms=pool_gen_ms,
    )
    part1_systems.append(m_xgb_ndcg)
    print(f"  -> XGBoost (rank:ndcg) fit in {xgb_ndcg_fit_s:.2f}s | nDCG@10: {m_xgb_ndcg.ndcg_10:.4f} | Latency: {m_xgb_ndcg.pure_rerank_ms:.1f} ms", flush=True)

    # 3. XGBoostRanker (rank:pairwise)
    print("Training XGBoost (objective=rank:pairwise, n_estimators=100, max_depth=6)...", flush=True)
    xgb_pair_params = {
        "objective": "rank:pairwise",
        "eval_metric": "ndcg@10",
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }
    t_xgb_pair_fit = time.perf_counter()
    xgb_pair_ranker = XGBoostRanker(model_params=xgb_pair_params, extractor=feature_extractor)
    xgb_pair_ranker.fit(full_train_dataset)
    xgb_pair_fit_s = time.perf_counter() - t_xgb_pair_fit
    m_xgb_pair = evaluate_ranker_on_pools(
        ranker=xgb_pair_ranker,
        pools=test_pools,
        cases=test_cases,
        name="XGBoost (rank:pairwise)",
        condition="depth-wise (max_depth=6)",
        fit_time_s=xgb_pair_fit_s,
        pool_gen_ms=pool_gen_ms,
    )
    part1_systems.append(m_xgb_pair)
    print(f"  -> XGBoost (rank:pairwise) fit in {xgb_pair_fit_s:.2f}s | nDCG@10: {m_xgb_pair.ndcg_10:.4f} | Latency: {m_xgb_pair.pure_rerank_ms:.1f} ms", flush=True)

    # Print Part 1 Table
    print("\n### Part 1: GBDT Framework Head-to-Head Matrix (BEIR SciFact, N=300)", flush=True)
    print("| System | Objective / Structure | Recall@5 | MRR | nDCG@5 | Recall@10 | nDCG@10 | Fit Time | Pure Rerank |", flush=True)
    print("|:---|:---|---:|---:|---:|---:|---:|---:|---:|", flush=True)
    for m in part1_systems:
        fit_str = f"{m.fit_time_s:.2f}s" if m.fit_time_s > 0 else "-"
        rerank_str = f"{m.pure_rerank_ms:.1f} ms" if m.pure_rerank_ms > 0 else "-"
        print(
            f"| **{m.name}** | {m.condition} | {m.recall_5:.4f} | {m.mrr_val:.4f} | "
            f"{m.ndcg_5:.4f} | {m.recall_10:.4f} | {m.ndcg_10:.4f} | {fit_str} | {rerank_str} |",
            flush=True,
        )

    # Feature Importance Comparison
    print("\n### Top 5 Feature Importances by Framework", flush=True)
    lgb_imps = lgb_ranker.get_feature_importances(importance_type="gain")
    xgb_ndcg_imps = xgb_ndcg_ranker.get_feature_importances()
    xgb_pair_imps = xgb_pair_ranker.get_feature_importances()

    lgb_top5 = list(lgb_imps.items())[:5]
    xgb_ndcg_top5 = list(xgb_ndcg_imps.items())[:5]
    xgb_pair_top5 = list(xgb_pair_imps.items())[:5]

    print("\nLightGBM LambdaMART Top 5:", flush=True)
    for rank, (fname, val) in enumerate(lgb_top5, 1):
        print(f"  {rank}. {fname:<22}: {val:.2f}", flush=True)

    print("\nXGBoost (rank:ndcg) Top 5:", flush=True)
    for rank, (fname, val) in enumerate(xgb_ndcg_top5, 1):
        print(f"  {rank}. {fname:<22}: {val:.4f}", flush=True)

    print("\nXGBoost (rank:pairwise) Top 5:", flush=True)
    for rank, (fname, val) in enumerate(xgb_pair_top5, 1):
        print(f"  {rank}. {fname:<22}: {val:.4f}", flush=True)

    # =========================================================================
    # PART 2: SAMPLE EFFICIENCY HEAD-TO-HEAD
    # =========================================================================
    print("\n" + "=" * 95, flush=True)
    print("PART 2: SAMPLE EFFICIENCY HEAD-TO-HEAD (LightGBM vs. XGBoost)", flush=True)
    print("=" * 95, flush=True)

    sample_sizes = [25, 100, 809]
    sample_metrics: list[tuple[int, SystemMetrics, SystemMetrics]] = []

    for n_q in sample_sizes:
        print(f"\n--- Subsampling N_train = {n_q} queries ---", flush=True)
        sub_train = slice_ltr_dataset(full_train_dataset, num_queries=n_q)

        # LightGBM
        t_l = time.perf_counter()
        l_ranker = LightGBMRanker(model_params=lgb_params, extractor=feature_extractor)
        l_ranker.fit(sub_train)
        l_fit = time.perf_counter() - t_l
        m_l = evaluate_ranker_on_pools(
            ranker=l_ranker,
            pools=test_pools,
            cases=test_cases,
            name=f"LightGBM (N={n_q})",
            condition=f"{n_q} queries",
            fit_time_s=l_fit,
            pool_gen_ms=pool_gen_ms,
        )

        # XGBoost (rank:ndcg)
        t_x = time.perf_counter()
        x_ranker = XGBoostRanker(model_params=xgb_ndcg_params, extractor=feature_extractor)
        x_ranker.fit(sub_train)
        x_fit = time.perf_counter() - t_x
        m_x = evaluate_ranker_on_pools(
            ranker=x_ranker,
            pools=test_pools,
            cases=test_cases,
            name=f"XGBoost (N={n_q})",
            condition=f"{n_q} queries",
            fit_time_s=x_fit,
            pool_gen_ms=pool_gen_ms,
        )

        sample_metrics.append((n_q, m_l, m_x))
        print(f"  LightGBM (N={n_q}): nDCG@10 = {m_l.ndcg_10:.4f} (Fit {l_fit:.2f}s)", flush=True)
        print(f"  XGBoost  (N={n_q}): nDCG@10 = {m_x.ndcg_10:.4f} (Fit {x_fit:.2f}s)", flush=True)

    # Print Part 2 Table
    print("\n### Part 2: Sample Efficiency Comparison (LightGBM vs. XGBoost)", flush=True)
    print("| N Queries | Model | Recall@5 | MRR | nDCG@5 | Recall@10 | nDCG@10 | Delta (XGB - LGB) |", flush=True)
    print("|:---:|:---|---:|---:|---:|---:|---:|---:|", flush=True)
    for n_q, m_l, m_x in sample_metrics:
        delta = m_x.ndcg_10 - m_l.ndcg_10
        print(
            f"| {n_q} | **LightGBM** | {m_l.recall_5:.4f} | {m_l.mrr_val:.4f} | "
            f"{m_l.ndcg_5:.4f} | {m_l.recall_10:.4f} | {m_l.ndcg_10:.4f} | Baseline |",
            flush=True,
        )
        print(
            f"| {n_q} | **XGBoost (ndcg)** | {m_x.recall_5:.4f} | {m_x.mrr_val:.4f} | "
            f"{m_x.ndcg_5:.4f} | {m_x.recall_10:.4f} | {m_x.ndcg_10:.4f} | {delta:+.4f} |",
            flush=True,
        )

    print("\n" + "=" * 95, flush=True)
    print("EXPERIMENT 031 COMPLETED SUCCESSFULLY.", flush=True)
    print("=" * 95, flush=True)


if __name__ == "__main__":
    run_experiment()
