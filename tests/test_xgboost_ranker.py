"""
Unit tests for XGBoostRanker and Learning-to-Rank re-ranking.
"""

from pathlib import Path
import numpy as np
import pytest

from retrievlab.features.dataset import LTRDataset
from retrievlab.models import Chunk, SearchResult
from retrievlab.ranking.interface import ReRanker
from retrievlab.ranking.xgboost import XGBoostRanker
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

    # Make first candidate of each query relevant (label 2 or 1), rest irrelevant (0)
    for q in range(num_queries):
        start_idx = q * cands_per_query
        for i in range(cands_per_query):
            idx = start_idx + i
            if i == 0:
                labels[idx] = 2
                features[idx, 0] = 20.0
                features[idx, 1] = 0.90
                features[idx, 8] = 1.0
                features[idx, 9] = 0.8
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


def test_xgboost_ranker_inherits_interface() -> None:
    ranker = XGBoostRanker()
    assert isinstance(ranker, ReRanker)
    assert not ranker.is_fitted
    assert ranker.model_params["objective"] == "rank:ndcg"


def test_rerank_before_fit_raises() -> None:
    ranker = XGBoostRanker()
    with pytest.raises(RuntimeError, match="must be fitted before"):
        ranker.rerank("Query", candidates=[])


def test_fit_invalid_dataset_raises() -> None:
    ranker = XGBoostRanker()
    empty_ds = LTRDataset(
        features=np.empty((0, 5)),
        labels=np.empty(0),
        group_sizes=[],
        query_ids=[],
        chunk_ids=[],
        feature_names=["f1", "f2", "f3", "f4", "f5"],
    )
    with pytest.raises(ValueError, match="Cannot fit XGBoostRanker on empty dataset"):
        ranker.fit(empty_ds)

    mismatched_ds = LTRDataset(
        features=np.ones((10, 2)),
        labels=np.ones(10),
        group_sizes=[5],  # Sum is 5 != 10
        query_ids=["q"] * 10,
        chunk_ids=[f"c_{i}" for i in range(10)],
        feature_names=["f1", "f2"],
    )
    with pytest.raises(ValueError, match="Sum of group_sizes"):
        ranker.fit(mismatched_ds)


def test_fit_and_predict_synthetic(synthetic_ltr_dataset: LTRDataset) -> None:
    ranker = XGBoostRanker(model_params={"n_estimators": 10, "max_depth": 3})
    ranker.fit(synthetic_ltr_dataset)

    assert ranker.is_fitted
    assert ranker.model is not None
    assert ranker.feature_names_ == synthetic_ltr_dataset.feature_names

    preds = ranker.model.predict(synthetic_ltr_dataset.features)
    assert len(preds) == len(synthetic_ltr_dataset)
    # The first candidate of each query should have higher score than the third
    assert preds[0] > preds[2]
    assert preds[6] > preds[8]


def test_pairwise_objective(synthetic_ltr_dataset: LTRDataset) -> None:
    ranker = XGBoostRanker(model_params={"objective": "rank:pairwise", "n_estimators": 10, "max_depth": 3})
    ranker.fit(synthetic_ltr_dataset)
    assert ranker.is_fitted
    preds = ranker.model.predict(synthetic_ltr_dataset.features)
    assert len(preds) == len(synthetic_ltr_dataset)


def test_rerank_candidates_and_pool(synthetic_ltr_dataset: LTRDataset) -> None:
    ranker = XGBoostRanker(model_params={"n_estimators": 15, "max_depth": 3})
    ranker.fit(synthetic_ltr_dataset)

    chunks = [
        Chunk(id=f"c_{i}", document_id=f"d_{i}", text=f"Document text {i} with some content", metadata={})
        for i in range(4)
    ]
    pool = CandidatePool(
        query="test query",
        candidates=[
            Candidate(chunk=chunks[0], sources=["dense"], retriever_scores={"dense": 0.9, "bm25": 10.0}, retriever_ranks={"dense": 1, "bm25": 5}),
            Candidate(chunk=chunks[1], sources=["bm25"], retriever_scores={"dense": 0.5, "bm25": 15.0}, retriever_ranks={"dense": 10, "bm25": 1}),
            Candidate(chunk=chunks[2], sources=["dense"], retriever_scores={"dense": 0.2, "bm25": 1.0}, retriever_ranks={"dense": 20, "bm25": 40}),
            Candidate(chunk=chunks[3], sources=["bm25"], retriever_scores={"dense": 0.1, "bm25": 1.0}, retriever_ranks={"dense": 50, "bm25": 50}),
        ],
    )

    results = ranker.rerank("test query", pool, top_k=2)
    assert len(results) == 2
    assert isinstance(results[0], SearchResult)
    # Results should be strictly descending in score
    assert results[0].score >= results[1].score


def test_rerank_empty_candidates(synthetic_ltr_dataset: LTRDataset) -> None:
    ranker = XGBoostRanker(model_params={"n_estimators": 5, "max_depth": 2})
    ranker.fit(synthetic_ltr_dataset)

    results = ranker.rerank("test query", candidates=[])
    assert results == []


def test_feature_importances(synthetic_ltr_dataset: LTRDataset) -> None:
    ranker = XGBoostRanker(model_params={"n_estimators": 10, "max_depth": 3})
    ranker.fit(synthetic_ltr_dataset)

    importances = ranker.get_feature_importances()
    assert isinstance(importances, dict)
    assert len(importances) == len(synthetic_ltr_dataset.feature_names)
    vals = list(importances.values())
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))


def test_save_and_load(tmp_path: Path, synthetic_ltr_dataset: LTRDataset) -> None:
    save_dir = tmp_path / "xgb_model"
    ranker = XGBoostRanker(model_params={"n_estimators": 10, "max_depth": 3})
    ranker.fit(synthetic_ltr_dataset)

    ranker.save(save_dir)
    assert (save_dir / "model.json").exists()
    assert (save_dir / "metadata.json").exists()

    loaded_ranker = XGBoostRanker.load(save_dir)
    assert loaded_ranker.is_fitted
    assert loaded_ranker.feature_names_ == ranker.feature_names_

    p1 = ranker.model.predict(synthetic_ltr_dataset.features)
    p2 = loaded_ranker.model.predict(synthetic_ltr_dataset.features)
    np.testing.assert_allclose(p1, p2, rtol=1e-5)
