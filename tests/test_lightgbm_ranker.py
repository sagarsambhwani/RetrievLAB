"""
Unit tests for LightGBMRanker and Learning-to-Rank re-ranking.
"""

from pathlib import Path
import numpy as np
import pytest

from retrievlab.features.dataset import LTRDataset
from retrievlab.models import Chunk, SearchResult
from retrievlab.ranking.interface import ReRanker
from retrievlab.ranking.lightgbm import LightGBMRanker
from retrievlab.selection.candidate import Candidate, CandidatePool


@pytest.fixture
def synthetic_ltr_dataset() -> LTRDataset:
    """Create a small, deterministic synthetic LTR dataset with 5 queries."""
    feature_names = [
        "bm25_score",
        "dense_score",
        "bm25_rank",
        "dense_rank",
        "bm25_reciprocal_rank",
        "dense_reciprocal_rank",
        "rrf_score",
        "retrieved_by_both",
        "exact_query_match",
        "token_overlap_ratio",
        "jaccard_similarity",
        "query_token_count",
        "doc_token_count",
        "length_ratio",
        "rank_discrepancy",
        "modality_preference",
    ]

    # 5 queries, 6 candidates each = 30 rows
    np.random.seed(42)
    num_queries = 5
    cands_per_query = 6
    total_rows = num_queries * cands_per_query

    features = np.zeros((total_rows, len(feature_names)), dtype=np.float64)
    labels = np.zeros(total_rows, dtype=np.int64)

    # Make first candidate of each query relevant (label 1 or 2), rest irrelevant (0)
    for q in range(num_queries):
        start_idx = q * cands_per_query
        for i in range(cands_per_query):
            idx = start_idx + i
            if i == 0:
                labels[idx] = 2
                features[idx, 0] = 20.0  # High bm25_score
                features[idx, 1] = 0.90  # High dense_score
                features[idx, 8] = 1.0   # exact_query_match
                features[idx, 9] = 0.8   # token_overlap_ratio
            elif i == 1:
                labels[idx] = 1
                features[idx, 0] = 14.0
                features[idx, 1] = 0.75
                features[idx, 8] = 0.0
                features[idx, 9] = 0.4
            else:
                labels[idx] = 0
                features[idx, 0] = float(np.random.uniform(1.0, 5.0))
                features[idx, 1] = float(np.random.uniform(0.1, 0.4))
                features[idx, 8] = 0.0
                features[idx, 9] = 0.05

    group_sizes = [cands_per_query] * num_queries
    query_ids = [f"q_{i // cands_per_query}" for i in range(total_rows)]
    chunk_ids = [f"c_{i}" for i in range(total_rows)]

    return LTRDataset(
        features=features,
        labels=labels,
        group_sizes=group_sizes,
        query_ids=query_ids,
        chunk_ids=chunk_ids,
        feature_names=feature_names,
    )


def test_lightgbm_ranker_inherits_interface() -> None:
    ranker = LightGBMRanker()
    assert isinstance(ranker, ReRanker)
    assert not ranker.is_fitted
    assert ranker.model_params["objective"] == "lambdarank"


def test_rerank_before_fit_raises() -> None:
    ranker = LightGBMRanker()
    with pytest.raises(RuntimeError, match="must be fitted before"):
        ranker.rerank("Query", candidates=[])


def test_fit_invalid_dataset_raises() -> None:
    ranker = LightGBMRanker()
    empty_ds = LTRDataset(
        features=np.empty((0, 16)),
        labels=np.empty((0,), dtype=int),
        group_sizes=[],
        feature_names=["f" + str(i) for i in range(16)],
    )
    with pytest.raises(ValueError, match="Cannot fit LightGBMRanker on empty dataset"):
        ranker.fit(empty_ds)

    mismatched_ds = LTRDataset(
        features=np.zeros((10, 16)),
        labels=np.zeros(10, dtype=int),
        group_sizes=[5],  # Sum is 5, but 10 rows
        feature_names=["f" + str(i) for i in range(16)],
    )
    with pytest.raises(ValueError, match="Sum of group_sizes"):
        ranker.fit(mismatched_ds)


def test_fit_and_rerank_synthetic(synthetic_ltr_dataset: LTRDataset) -> None:
    ranker = LightGBMRanker(
        model_params={
            "n_estimators": 20,
            "min_child_samples": 2,
            "num_leaves": 7,
            "learning_rate": 0.1,
        }
    )
    ranker.fit(synthetic_ltr_dataset)
    assert ranker.is_fitted
    assert len(ranker.feature_names_) == 16

    # Verify feature importances
    gain_imp = ranker.get_feature_importances(importance_type="gain")
    assert isinstance(gain_imp, dict)
    assert len(gain_imp) == 16
    assert any(val > 0.0 for val in gain_imp.values())

    # Create candidate pool to test rerank
    chunk_rel = Chunk(
        id="c_rel",
        document_id="d_rel",
        text="Aspirin treats cardiovascular diseases and reduces arterial blood clotting.",
    )
    chunk_irrel = Chunk(
        id="c_irrel",
        document_id="d_irrel",
        text="The solar system has planets orbiting around the central star in space.",
    )

    cand_rel = Candidate(
        chunk=chunk_rel,
        sources=["bm25", "dense"],
        retriever_scores={"bm25": 18.5, "dense": 0.88},
        retriever_ranks={"bm25": 1, "dense": 1},
    )
    cand_irrel = Candidate(
        chunk=chunk_irrel,
        sources=["bm25"],
        retriever_scores={"bm25": 2.1, "dense": 0.15},
        retriever_ranks={"bm25": 45},
    )

    query = "cardiovascular disease aspirin"
    pool = CandidatePool(
        query=query,
        candidates=[cand_irrel, cand_rel],
        top_k_per_retriever=10,
    )

    results = ranker.rerank(query, candidates=pool)
    assert len(results) == 2
    assert isinstance(results[0], SearchResult)
    # The relevant candidate with high BM25 and exact match should be ranked #1
    assert results[0].chunk.id == "c_rel"
    assert results[1].chunk.id == "c_irrel"
    assert results[0].score > results[1].score


def test_rerank_top_k_and_empty(synthetic_ltr_dataset: LTRDataset) -> None:
    ranker = LightGBMRanker(
        model_params={"n_estimators": 10, "min_child_samples": 2, "num_leaves": 7}
    )
    ranker.fit(synthetic_ltr_dataset)

    # Empty candidate list
    assert ranker.rerank("Query", candidates=[]) == []

    # Empty CandidatePool
    empty_pool = CandidatePool(query="Query", candidates=[])
    assert ranker.rerank("Query", candidates=empty_pool) == []

    # Top-K slicing
    chunks = [
        Chunk(id=f"c_{i}", document_id=f"d_{i}", text=f"Text content number {i}")
        for i in range(5)
    ]
    results = ranker.rerank("Query text", candidates=chunks, top_k=2)
    assert len(results) == 2


def test_rerank_invalid_type_raises(synthetic_ltr_dataset: LTRDataset) -> None:
    ranker = LightGBMRanker(
        model_params={"n_estimators": 5, "min_child_samples": 2}
    )
    ranker.fit(synthetic_ltr_dataset)

    with pytest.raises(TypeError, match="Expected Chunk, Candidate, or CandidatePool"):
        ranker.rerank("Query", candidates=12345)  # type: ignore[arg-type]


def test_save_and_load_model(synthetic_ltr_dataset: LTRDataset, tmp_path: Path) -> None:
    ranker = LightGBMRanker(
        model_params={"n_estimators": 15, "min_child_samples": 2, "num_leaves": 7}
    )
    ranker.fit(synthetic_ltr_dataset)

    chunk_a = Chunk(id="ca", document_id="da", text="Machine learning model training.")
    chunk_b = Chunk(id="cb", document_id="db", text="Completely unrelated astronomy article.")
    cands = [chunk_a, chunk_b]

    original_results = ranker.rerank("machine learning", candidates=cands)

    save_dir = tmp_path / "lgb_ranker_test"
    ranker.save(save_dir)

    assert (save_dir / "model.txt").exists()
    assert (save_dir / "metadata.json").exists()

    loaded_ranker = LightGBMRanker.load(save_dir)
    assert loaded_ranker.is_fitted
    assert loaded_ranker.feature_names_ == ranker.feature_names_

    loaded_results = loaded_ranker.rerank("machine learning", candidates=cands)
    assert len(loaded_results) == len(original_results)
    for orig, loaded in zip(original_results, loaded_results):
        assert orig.chunk.id == loaded.chunk.id
        assert pytest.approx(orig.score, rel=1e-5) == loaded.score
