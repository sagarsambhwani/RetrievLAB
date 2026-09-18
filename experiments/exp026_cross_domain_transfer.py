"""
Experiment 026: Cross-Domain Transfer of Learned Ranking Policies.

Evaluates the cross-domain transferability of learned ranking policies on BEIR NFCorpus:
1. Single-Stage Baselines: BM25, FAISS Dense (FlatIP), and Hybrid RRF (1:2)
2. Two-Stage Neural: Union Candidates (K=50) + Zero-Shot Cross-Encoder (MS MARCO MiniLM)
3. Two-Stage GBDT Zero-Shot: Union Candidates (K=50) + SciFact-Trained LightGBM (No retraining)
4. Two-Stage GBDT In-Domain: Union Candidates (K=50) + NFCorpus-Trained LightGBM (In-domain)

Measures:
- Retrieval accuracy (@5, @10, MRR)
- Candidate pool recall ceiling (disentangling Stage-1 retrieval vs Stage-2 ranking failures)
- Domain-transfer penalty (In-Domain LightGBM - Zero-Shot LightGBM)
- Feature gain attribution shift across domains
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


def get_or_compute_embeddings(
    chunks: list[Chunk],
    client: FastEmbedClient,
    cache_path: Path,
) -> list[Chunk]:
    """Load or precompute FastEmbed embeddings for corpus chunks."""
    if cache_path.exists():
        print(f"Loading cached embeddings from {cache_path}...", flush=True)
        emb_matrix = np.load(cache_path)
        for i, chunk in enumerate(chunks):
            chunk.embedding = emb_matrix[i].tolist()
        print(f"Loaded {len(emb_matrix)} cached vectors (dim: {emb_matrix.shape[1]}).", flush=True)
        return chunks

    print(f"Generating embeddings for {len(chunks)} NFCorpus chunks with FastEmbed...", flush=True)
    t0 = time.perf_counter()
    embedded_chunks = client.embed_chunks(chunks, batch_size=256)
    emb_list = [c.embedding for c in embedded_chunks if c.embedding is not None]
    emb_matrix = np.array(emb_list, dtype=np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, emb_matrix)
    print(f"Generated and cached {len(emb_matrix)} embeddings in {(time.perf_counter() - t0):.1f}s.", flush=True)
    return embedded_chunks


def compute_candidate_pool_ceiling(
    pools,
    benchmark_cases: list[BenchmarkCase],
) -> float:
    """Compute mean candidate pool recall ceiling across all benchmark queries."""
    ceilings = []
    for pool, case in zip(pools, benchmark_cases):
        if not case.relevant_chunk_ids:
            continue
        pool_ids = {cand.chunk.id for cand in pool.candidates}
        rel_ids = set(case.relevant_chunk_ids)
        hits = len(pool_ids & rel_ids)
        ceilings.append(hits / len(rel_ids))
    return float(np.mean(ceilings)) if ceilings else 0.0


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
    pools,
    cross_encoder: CrossEncoderReRanker,
    benchmark_cases: list[BenchmarkCase],
    pool_gen_time_ms: float,
) -> tuple[SystemMetrics, list[list[SearchResult]]]:
    """Evaluate two-stage Union + Cross-Encoder re-ranker on pre-generated candidate pools."""
    all_pairs: list[tuple[str, str]] = []
    pool_slices: list[tuple[int, int]] = []
    offset = 0
    for pool in pools:
        count = len(pool.candidates)
        for cand in pool.candidates:
            all_pairs.append((pool.query, cand.chunk.text or ""))
        pool_slices.append((offset, offset + count))
        offset += count

    print(f"Scoring {len(all_pairs)} candidate pairs in bulk with Cross-Encoder (GPU)...", flush=True)
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
        name="Two-Stage: Union + Cross-Encoder (Zero-Shot MS MARCO)",
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


def evaluate_lightgbm_ranker(
    name: str,
    lgb_ranker: LightGBMRanker,
    pools,
    benchmark_cases: list[BenchmarkCase],
    pool_gen_time_ms: float,
) -> tuple[SystemMetrics, list[list[SearchResult]]]:
    """Evaluate a LightGBM ranker on pre-generated candidate pools."""
    all_results: list[list[SearchResult]] = []
    recalls_5, precisions_5, mrrs, ndcgs_5, maps_5, hits_5 = [], [], [], [], [], []
    recalls_10, precisions_10, ndcgs_10, maps_10, hits_10 = [], [], [], [], []
    pure_rerank_latencies = []

    for pool, case in zip(pools, benchmark_cases):
        t0 = time.perf_counter()
        results = lgb_ranker.rerank(case.query, pool, top_k=10)
        rerank_ms = (time.perf_counter() - t0) * 1000.0

        pure_rerank_latencies.append(rerank_ms)
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

    mean_pure_ms = float(np.mean(pure_rerank_latencies))
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
        mean_latency_ms=float(pool_gen_time_ms + mean_pure_ms),
        pure_rerank_ms=mean_pure_ms,
    )
    return metrics, all_results


def get_or_build_nfcorpus_train_dataset(
    generator: MultiRetrieverCandidateGenerator,
    extractor: FeatureExtractor,
    train_cases: list[BenchmarkCase],
    chunks: list[Chunk],
    cache_path: Path,
    max_train_queries: int = 1000,
) -> LTRDataset:
    """Load or generate query-grouped LTR training dataset for NFCorpus."""
    if cache_path.exists():
        print(f"Loading cached NFCorpus LTR training dataset from {cache_path}...", flush=True)
        data = np.load(cache_path, allow_pickle=True)
        return LTRDataset(
            features=data["features"],
            labels=data["labels"],
            group_sizes=data["group_sizes"].tolist(),
            feature_names=data["feature_names"].tolist(),
        )

    selected_cases = train_cases[:max_train_queries]
    print(f"Generating candidate pools for {len(selected_cases)} NFCorpus train queries...", flush=True)
    t0 = time.perf_counter()
    train_pools = []
    for i, case in enumerate(selected_cases):
        if (i + 1) % 200 == 0 or i == len(selected_cases) - 1:
            print(f"  Processed {i + 1}/{len(selected_cases)} train query pools...", flush=True)
        pool = generator.generate(case.query, top_k_per_retriever=50, chunks=chunks)
        train_pools.append(pool)

    print(f"Candidate pools generated in {(time.perf_counter() - t0):.1f}s. Building LTRDataset...", flush=True)
    t1 = time.perf_counter()
    dataset = build_ltr_dataset(train_pools, selected_cases, extractor=extractor)
    print(f"NFCorpus LTRDataset built in {(time.perf_counter() - t1):.1f}s: {len(dataset)} rows, {dataset.num_features} features.", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=dataset.features,
        labels=dataset.labels,
        group_sizes=np.array(dataset.group_sizes, dtype=np.int32),
        feature_names=np.array(dataset.feature_names),
    )
    print(f"Cached NFCorpus LTR training dataset to {cache_path}.", flush=True)
    return dataset


def run_experiment() -> None:
    print("=" * 80, flush=True)
    print("EXPERIMENT 026: CROSS-DOMAIN TRANSFER OF LEARNED RANKING POLICIES", flush=True)
    print("Target Domain: BEIR NFCorpus (Nutrition Facts Medical QA)", flush=True)
    print("=" * 80, flush=True)

    # Step 1: Ingest NFCorpus
    loader = BEIRLoader("nfcorpus")
    chunks = loader.load_corpus()
    test_benchmark = loader.load_benchmark(split="test")
    train_benchmark = loader.load_benchmark(split="train")
    test_cases = test_benchmark.cases
    train_cases = train_benchmark.cases

    print(f"Corpus: {len(chunks)} documents.", flush=True)
    print(f"Test Queries: {len(test_cases)} | Train Queries: {len(train_cases)}", flush=True)

    # Step 2: Ingest & Embed Chunks
    client = FastEmbedClient()
    emb_cache = Path("data/processed/nfcorpus_embeddings.npy")
    chunks = get_or_compute_embeddings(chunks, client, emb_cache)

    print("Building BM25 index on NFCorpus...", flush=True)
    bm25 = BM25Retriever()
    bm25.index(chunks)

    print("Initializing FAISS Dense index...", flush=True)
    faiss_retriever = FAISSRetriever(client=client)

    print("Initializing Hybrid RRF (1:2)...", flush=True)
    hybrid_rrf = HybridRetriever(
        retrievers=[bm25, faiss_retriever],
        weights=[1.0, 2.0],
    )

    generator = MultiRetrieverCandidateGenerator(
        retrievers={"bm25": bm25, "dense": faiss_retriever}
    )
    extractor = FeatureExtractor()

    # Step 3: Evaluate Single-Stage Baselines
    print("\nEvaluating Single-Stage Baselines on NFCorpus...", flush=True)
    bm25_metrics, _ = evaluate_single_stage("BM25 Single-Stage", bm25, test_cases, chunks)
    dense_metrics, _ = evaluate_single_stage("FAISS Dense Single-Stage", faiss_retriever, test_cases, chunks)
    rrf_metrics, _ = evaluate_single_stage("Hybrid RRF (1:2) Single-Stage", hybrid_rrf, test_cases, chunks)

    # Step 4: Generate Test Candidate Pools (Union K=50) & Measure Ceiling
    print("\nGenerating Test Candidate Pools (Union K=50) for 323 queries...", flush=True)
    t_pool_start = time.perf_counter()
    test_pools = []
    for case in test_cases:
        p = generator.generate(case.query, top_k_per_retriever=50, chunks=chunks)
        test_pools.append(p)
    t_pool_end = time.perf_counter()
    mean_pool_gen_ms = (t_pool_end - t_pool_start) * 1000.0 / len(test_cases)
    avg_pool_size = float(np.mean([len(p.candidates) for p in test_pools]))

    candidate_recall_ceiling = compute_candidate_pool_ceiling(test_pools, test_cases)
    print(f"Mean Pool Generation Latency: {mean_pool_gen_ms:.1f} ms/query", flush=True)
    print(f"Average Candidate Pool Size:  {avg_pool_size:.1f} candidates/query", flush=True)
    print(f"Candidate Pool Recall Ceiling: {candidate_recall_ceiling * 100:.2f}% (Theoretical Maximum)", flush=True)

    # Step 5: Evaluate Zero-Shot Neural Cross-Encoder
    print("\nEvaluating Two-Stage Neural Cross-Encoder (MS MARCO MiniLM)...", flush=True)
    ce_ranker = CrossEncoderReRanker(batch_size=64, max_length=256, device="cuda")
    ce_metrics, _ = evaluate_cross_encoder_stage(test_pools, ce_ranker, test_cases, mean_pool_gen_ms)

    # Step 6: Evaluate Zero-Shot SciFact-Trained LightGBM
    scifact_model_dir = Path("data/processed/scifact_lgb_ranker")
    if not scifact_model_dir.exists():
        raise FileNotFoundError(
            f"SciFact LightGBM model not found at {scifact_model_dir}. "
            "Run exp025_twostage_reranking_benchmark.py first."
        )
    print(f"\nLoading Zero-Shot SciFact LightGBM from {scifact_model_dir}...", flush=True)
    scifact_lgb = LightGBMRanker.load(scifact_model_dir, extractor=extractor)
    zs_lgb_metrics, _ = evaluate_lightgbm_ranker(
        name="Two-Stage: Union + LightGBM (Zero-Shot SciFact Model)",
        lgb_ranker=scifact_lgb,
        pools=test_pools,
        benchmark_cases=test_cases,
        pool_gen_time_ms=mean_pool_gen_ms,
    )

    # Step 7: Train and Evaluate In-Domain NFCorpus LightGBM
    nf_train_cache = Path("data/processed/nfcorpus_train_ltr.npz")
    nf_train_dataset = get_or_build_nfcorpus_train_dataset(
        generator=generator,
        extractor=extractor,
        train_cases=train_cases,
        chunks=chunks,
        cache_path=nf_train_cache,
        max_train_queries=1000,
    )

    print("\nFitting In-Domain NFCorpus LightGBMRanker (LambdaMART)...", flush=True)
    t_fit_start = time.perf_counter()
    nf_lgb = LightGBMRanker(
        model_params={
            "n_estimators": 100,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 10,
            "random_state": 42,
        },
        extractor=extractor,
    )
    nf_lgb.fit(nf_train_dataset)
    t_fit_end = time.perf_counter()
    print(f"NFCorpus LightGBM fitting completed in {(t_fit_end - t_fit_start):.2f}s.", flush=True)

    nf_model_dir = Path("data/processed/nfcorpus_lgb_ranker")
    nf_lgb.save(nf_model_dir)

    id_lgb_metrics, _ = evaluate_lightgbm_ranker(
        name="Two-Stage: Union + LightGBM (In-Domain NFCorpus Model)",
        lgb_ranker=nf_lgb,
        pools=test_pools,
        benchmark_cases=test_cases,
        pool_gen_time_ms=mean_pool_gen_ms,
    )

    # Step 8: Print Results Matrix
    systems = [
        bm25_metrics,
        dense_metrics,
        rrf_metrics,
        ce_metrics,
        zs_lgb_metrics,
        id_lgb_metrics,
    ]

    print("\n" + "=" * 115, flush=True)
    print("EMPIRICAL BENCHMARK RESULTS SUMMARY (BEIR NFCorpus, N=323)", flush=True)
    print("=" * 115, flush=True)

    header_5 = f"{'System':<48} | {'Recall@5':>8} | {'Prec@5':>7} | {'MRR':>7} | {'nDCG@5':>7} | {'Latency':>10}"
    print("\n--- PERFORMANCE AT CUTOFF @5 ---", flush=True)
    print(header_5, flush=True)
    print("-" * len(header_5), flush=True)
    for s in systems:
        lat_str = f"{s.mean_latency_ms:.1f} ms"
        print(
            f"{s.name:<48} | {s.recall_5:>8.4f} | {s.precision_5:>7.4f} | "
            f"{s.mrr_val:>7.4f} | {s.ndcg_5:>7.4f} | {lat_str:>10}",
            flush=True,
        )

    header_10 = f"{'System':<48} | {'Recall@10':>9} | {'Prec@10':>8} | {'MRR':>7} | {'nDCG@10':>8} | {'Latency':>10}"
    print("\n--- PERFORMANCE AT CUTOFF @10 ---", flush=True)
    print(header_10, flush=True)
    print("-" * len(header_10), flush=True)
    for s in systems:
        lat_str = f"{s.mean_latency_ms:.1f} ms"
        print(
            f"{s.name:<48} | {s.recall_10:>9.4f} | {s.precision_10:>8.4f} | "
            f"{s.mrr_val:>7.4f} | {s.ndcg_10:>8.4f} | {lat_str:>10}",
            flush=True,
        )

    # Step 9: Diagnostic Analysis: Ceiling & Domain-Transfer Penalty
    print("\n--- CANDIDATE POOL CEILING VS. RANKING EFFICIENCY ---", flush=True)
    print(f"Candidate Pool Recall Ceiling: {candidate_recall_ceiling:.4f} (100.0% of recoverable truth)", flush=True)
    print(f"Zero-Shot Cross-Encoder Recall@5:  {ce_metrics.recall_5:.4f} ({ce_metrics.recall_5 / candidate_recall_ceiling * 100:.1f}% of ceiling)", flush=True)
    print(f"Zero-Shot SciFact LightGBM Recall@5:{zs_lgb_metrics.recall_5:.4f} ({zs_lgb_metrics.recall_5 / candidate_recall_ceiling * 100:.1f}% of ceiling)", flush=True)
    print(f"In-Domain NFCorpus LightGBM Recall@5:{id_lgb_metrics.recall_5:.4f} ({id_lgb_metrics.recall_5 / candidate_recall_ceiling * 100:.1f}% of ceiling)", flush=True)

    delta_ndcg5 = id_lgb_metrics.ndcg_5 - zs_lgb_metrics.ndcg_5
    delta_recall5 = id_lgb_metrics.recall_5 - zs_lgb_metrics.recall_5
    delta_mrr = id_lgb_metrics.mrr_val - zs_lgb_metrics.mrr_val

    print("\n--- DOMAIN-TRANSFER PENALTY (In-Domain - Zero-Shot) ---", flush=True)
    print(f"nDCG@5 Penalty:     {delta_ndcg5:+.4f} ({delta_ndcg5 / max(zs_lgb_metrics.ndcg_5, 1e-6) * 100:+.1f}%)", flush=True)
    print(f"Recall@5 Penalty:   {delta_recall5:+.4f} ({delta_recall5 / max(zs_lgb_metrics.recall_5, 1e-6) * 100:+.1f}%)", flush=True)
    print(f"MRR Penalty:        {delta_mrr:+.4f} ({delta_mrr / max(zs_lgb_metrics.mrr_val, 1e-6) * 100:+.1f}%)", flush=True)

    # Step 10: Feature Importance Shift Diagnostic
    scifact_imp = scifact_lgb.get_feature_importances(importance_type="gain")
    nf_imp = nf_lgb.get_feature_importances(importance_type="gain")

    print("\n--- FEATURE GAIN IMPORTANCE SHIFT ---", flush=True)
    print(f"{'Feature Name':<24} | {'SciFact Model Gain':>18} | {'NFCorpus Model Gain':>19}", flush=True)
    print("-" * 68, flush=True)
    for feat in extractor.get_feature_names():
        sf_g = scifact_imp.get(feat, 0.0)
        nf_g = nf_imp.get(feat, 0.0)
        print(f"{feat:<24} | {sf_g:>18.2f} | {nf_g:>19.2f}", flush=True)


if __name__ == "__main__":
    run_experiment()
