"""
Experiment 030: Stress-Testing LightGBM (Feature Ablation & Sample Complexity).

Evaluates the architectural boundaries and failure modes of Tabular GBDT ranking on BEIR SciFact:
Part 1: Feature Group Ablation Study (Items 35, 37, 38)
  - Full Suite (16 features, baseline)
  - Dense-Only / Lexical-Ablated (6 features)
  - Lexical-Only / Dense-Ablated (9 features)
  - Rank-Only / Score-Ablated (8 features - Item 37)
  - Score-Only / Rank-Ablated (6 features - Item 38)
  - Core-Only / No-Metadata (12 features)

Part 2: Sample Complexity & Learning Curves (Item 63)
  - Training on N_train in [10, 25, 50, 100, 200, 400, 809] queries
  - Identifies the Viability Threshold (minimum queries to beat BM25 / Dense)
  - Identifies the Saturation Knee (point of diminishing returns)
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
from retrievlab.ranking import LightGBMRanker
from retrievlab.retrieval import BM25Retriever
from retrievlab.selection import CandidatePool, MultiRetrieverCandidateGenerator


@dataclass
class SystemMetrics:
    """Aggregated evaluation metrics for a retrieval/re-ranking system."""

    name: str
    condition: str
    feature_count: int
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


def compute_metrics_for_results(
    results_per_query: list[list[SearchResult]],
    cases: list[BenchmarkCase],
    name: str,
    condition: str,
    feature_count: int,
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
        feature_count=feature_count,
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


def load_cached_embeddings(chunks: list[Chunk], cache_path: Path) -> list[Chunk]:
    """Load precomputed embedding vectors from disk into chunks."""
    print(f"Loading cached embeddings from {cache_path}...", flush=True)
    emb_matrix = np.load(cache_path)
    for i, chunk in enumerate(chunks):
        chunk.embedding = emb_matrix[i].tolist()
    return chunks


def evaluate_ranker_on_pools(
    ranker: LightGBMRanker,
    pools: list[CandidatePool],
    cases: list[BenchmarkCase],
    name: str,
    condition: str,
    feature_count: int,
    pool_gen_ms: float,
) -> SystemMetrics:
    """Evaluate a LightGBM ranker on precomputed candidate pools."""
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
        feature_count=feature_count,
        mean_latency_ms=total_ms,
        pure_rerank_ms=pure_rerank_ms,
    )


def slice_ltr_dataset(
    full_dataset: LTRDataset,
    selected_features: list[str] | None = None,
    num_queries: int | None = None,
) -> LTRDataset:
    """Slice an LTRDataset by feature subset and/or number of training queries."""
    features = full_dataset.features
    labels = full_dataset.labels
    group_sizes = list(full_dataset.group_sizes)
    all_names = list(full_dataset.feature_names)

    # 1. Slice by query count
    if num_queries is not None and num_queries < len(group_sizes):
        active_groups = group_sizes[:num_queries]
        item_count = sum(active_groups)
        features = features[:item_count]
        labels = labels[:item_count]
        group_sizes = active_groups

    # 2. Slice by features
    if selected_features is not None:
        col_indices = [all_names.index(f) for f in selected_features]
        features = features[:, col_indices]
        feature_names = list(selected_features)
    else:
        feature_names = all_names

    return LTRDataset(
        features=np.ascontiguousarray(features),
        labels=np.ascontiguousarray(labels),
        group_sizes=group_sizes,
        feature_names=feature_names,
    )


def run_experiment() -> None:
    print("=" * 95, flush=True)
    print("EXPERIMENT 030: STRESS-TESTING LIGHTGBM (FEATURE ABLATION & SAMPLE COMPLEXITY)", flush=True)
    print("Domain: BEIR SciFact (Biomedical Claim Verification, N=300 test queries)", flush=True)
    print("=" * 95, flush=True)

    # 1. Ingest SciFact & Build Candidate Pools
    loader = BEIRLoader("scifact")
    chunks = loader.load_corpus()
    benchmark = loader.load_benchmark(split="test")
    test_cases = benchmark.cases

    emb_cache = Path("data/processed/scifact_embeddings.npy")
    chunks = load_cached_embeddings(chunks, emb_cache)

    print("Building Stage-1 retrievers...", flush=True)
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
    print(f"Candidate pools generated: {len(test_pools)} pools (avg size {avg_pool_size:.1f}, {pool_gen_ms:.1f} ms/query).", flush=True)

    # 2. Load Precomputed Training Data
    train_cache = Path("data/processed/scifact_ltr_train.npz")
    print(f"Loading pre-cached LTR training dataset from {train_cache}...", flush=True)
    raw_train = np.load(train_cache)
    full_train_dataset = LTRDataset(
        features=raw_train["features"],
        labels=raw_train["labels"],
        group_sizes=list(raw_train["group_sizes"]),
        feature_names=list(raw_train["feature_names"]),
    )
    print(f"Loaded {full_train_dataset.features.shape[0]} pairs across {full_train_dataset.num_queries} training queries.", flush=True)

    # Standard model hyperparams
    model_params = {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 10,
    }

    # =========================================================================
    # PART 1: FEATURE ABLATION SWEEP
    # =========================================================================
    print("\n" + "=" * 95, flush=True)
    print("PART 1: FEATURE GROUP ABLATION STUDY (Full 809 Training Queries)", flush=True)
    print("=" * 95, flush=True)

    ablation_definitions = [
        (
            "Full Feature Suite",
            "Baseline",
            list(full_train_dataset.feature_names),
        ),
        (
            "Dense-Only (Lexical-Ablated)",
            "No BM25 / Overlap",
            [
                "dense_score",
                "dense_rank",
                "dense_reciprocal_rank",
                "query_token_count",
                "doc_token_count",
                "length_ratio",
            ],
        ),
        (
            "Lexical-Only (Dense-Ablated)",
            "No Dense Signals",
            [
                "bm25_score",
                "bm25_rank",
                "bm25_reciprocal_rank",
                "exact_query_match",
                "token_overlap_ratio",
                "jaccard_similarity",
                "query_token_count",
                "doc_token_count",
                "length_ratio",
            ],
        ),
        (
            "Rank-Only (Score-Ablated)",
            "Ordinal Ranks Only",
            [
                "bm25_rank",
                "dense_rank",
                "bm25_reciprocal_rank",
                "dense_reciprocal_rank",
                "rrf_score",
                "retrieved_by_both",
                "rank_discrepancy",
                "modality_preference",
            ],
        ),
        (
            "Score-Only (Rank-Ablated)",
            "Raw Scores Only",
            [
                "bm25_score",
                "dense_score",
                "exact_query_match",
                "token_overlap_ratio",
                "jaccard_similarity",
                "length_ratio",
            ],
        ),
        (
            "Core Signals (Metadata-Ablated)",
            "No Length Metadata",
            [
                "bm25_score",
                "dense_score",
                "bm25_rank",
                "dense_rank",
                "bm25_reciprocal_rank",
                "dense_reciprocal_rank",
                "rrf_score",
                "exact_query_match",
                "token_overlap_ratio",
                "jaccard_similarity",
                "rank_discrepancy",
                "modality_preference",
            ],
        ),
    ]

    ablation_metrics: list[SystemMetrics] = []

    for name, condition, feat_subset in ablation_definitions:
        print(f"Training LightGBM on {name} ({len(feat_subset)} features)...", flush=True)
        t_fit_start = time.perf_counter()
        train_sub = slice_ltr_dataset(full_train_dataset, selected_features=feat_subset)
        extractor_sub = FeatureExtractor(feature_names=feat_subset)
        ranker = LightGBMRanker(model_params=model_params, extractor=extractor_sub)
        ranker.fit(train_sub)
        fit_time = time.perf_counter() - t_fit_start

        metrics = evaluate_ranker_on_pools(
            ranker=ranker,
            pools=test_pools,
            cases=test_cases,
            name=name,
            condition=condition,
            feature_count=len(feat_subset),
            pool_gen_ms=pool_gen_ms,
        )
        ablation_metrics.append(metrics)
        print(
            f"  -> Fit in {fit_time:.2f}s | nDCG@10: {metrics.ndcg_10:.4f} | "
            f"MRR: {metrics.mrr_val:.4f} | Recall@10: {metrics.recall_10:.4f}",
            flush=True,
        )

    # Print Part 1 Table
    print("\n### Feature Ablation Comparison Matrix (BEIR SciFact, N=300)", flush=True)
    print("| Configuration | Condition | # Feats | Recall@5 | MRR | nDCG@5 | Recall@10 | nDCG@10 | Delta vs Baseline | Pure Rerank |", flush=True)
    print("|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|", flush=True)
    base_ndcg10 = ablation_metrics[0].ndcg_10
    for m in ablation_metrics:
        delta = m.ndcg_10 - base_ndcg10
        print(
            f"| **{m.name}** | {m.condition} | {m.feature_count} | {m.recall_5:.4f} | "
            f"{m.mrr_val:.4f} | {m.ndcg_5:.4f} | {m.recall_10:.4f} | {m.ndcg_10:.4f} | "
            f"{delta:+.4f} | {m.pure_rerank_ms:.1f} ms |",
            flush=True,
        )

    # =========================================================================
    # PART 2: SAMPLE COMPLEXITY LEARNING CURVES
    # =========================================================================
    print("\n" + "=" * 95, flush=True)
    print("PART 2: SAMPLE COMPLEXITY LEARNING CURVES (Full 16 Features)", flush=True)
    print("=" * 95, flush=True)

    sample_sizes = [10, 25, 50, 100, 200, 400, 809]
    complexity_metrics: list[SystemMetrics] = []
    full_extractor = FeatureExtractor()

    for n_q in sample_sizes:
        print(f"Training LightGBM on N_train = {n_q} queries...", flush=True)
        t_fit_start = time.perf_counter()
        train_sub = slice_ltr_dataset(full_train_dataset, num_queries=n_q)
        ranker = LightGBMRanker(model_params=model_params, extractor=full_extractor)
        ranker.fit(train_sub)
        fit_time = time.perf_counter() - t_fit_start

        metrics = evaluate_ranker_on_pools(
            ranker=ranker,
            pools=test_pools,
            cases=test_cases,
            name=f"LightGBM (N={n_q})",
            condition=f"{n_q} queries ({n_q / 809 * 100:.1f}%)",
            feature_count=16,
            pool_gen_ms=pool_gen_ms,
        )
        complexity_metrics.append(metrics)
        print(
            f"  -> Fit in {fit_time:.2f}s | nDCG@10: {metrics.ndcg_10:.4f} | "
            f"MRR: {metrics.mrr_val:.4f} | Recall@10: {metrics.recall_10:.4f}",
            flush=True,
        )

    # Print Part 2 Table
    print("\n### Sample Complexity Learning Curve Matrix (BEIR SciFact)", flush=True)
    print("| Training Queries | % of Training Set | Recall@5 | MRR | nDCG@5 | Recall@10 | nDCG@10 | Delta vs Full |", flush=True)
    print("|:---|:---|---:|---:|---:|---:|---:|---:|", flush=True)
    for m in complexity_metrics:
        delta = m.ndcg_10 - base_ndcg10
        print(
            f"| **{m.name}** | {m.condition} | {m.recall_5:.4f} | "
            f"{m.mrr_val:.4f} | {m.ndcg_5:.4f} | {m.recall_10:.4f} | {m.ndcg_10:.4f} | "
            f"{delta:+.4f} |",
            flush=True,
        )

    print("\n" + "=" * 95, flush=True)
    print("EXPERIMENT 030 COMPLETED SUCCESSFULLY.", flush=True)
    print("=" * 95, flush=True)


if __name__ == "__main__":
    run_experiment()
