"""
Unit tests for FeatureExtractor and LTRDataset.
"""

import numpy as np
import pytest

from retrievlab.models import Chunk
from retrievlab.selection.candidate import Candidate, CandidatePool
from retrievlab.evaluation.benchmark import BenchmarkCase
from retrievlab.features.extractor import FeatureExtractor
from retrievlab.features.dataset import LTRDataset, build_ltr_dataset


@pytest.fixture
def extractor() -> FeatureExtractor:
    return FeatureExtractor(missing_rank=1000, rrf_k=60)


def test_feature_extractor_feature_names(extractor: FeatureExtractor) -> None:
    names = extractor.get_feature_names()
    assert len(names) == 16
    assert "bm25_score" in names
    assert "dense_score" in names
    assert "rrf_score" in names
    assert "retrieved_by_both" in names
    assert "exact_query_match" in names
    assert "token_overlap_ratio" in names
    assert "jaccard_similarity" in names
    assert "rank_discrepancy" in names
    assert "modality_preference" in names


def test_single_modality_bm25_only(extractor: FeatureExtractor) -> None:
    chunk = Chunk(id="chunk_1", document_id="doc_1", text="Machine learning optimizes ranking models.")
    cand = Candidate(
        chunk=chunk,
        retriever_scores={"bm25": 12.5},
        retriever_ranks={"bm25": 2},
        sources=["bm25"],
    )

    query = "machine learning"
    feats = extractor.extract_candidate_features(query, cand)

    assert feats["bm25_score"] == 12.5
    assert feats["dense_score"] == 0.0
    assert feats["bm25_rank"] == 2.0
    assert feats["dense_rank"] == 1000.0
    assert feats["bm25_reciprocal_rank"] == 0.5
    assert feats["dense_reciprocal_rank"] == 0.0
    assert feats["rrf_score"] == pytest.approx(1.0 / (60 + 2))
    assert feats["retrieved_by_both"] == 0.0
    assert feats["exact_query_match"] == 1.0
    assert feats["rank_discrepancy"] == 998.0
    assert feats["modality_preference"] == 0.0


def test_single_modality_dense_only(extractor: FeatureExtractor) -> None:
    chunk = Chunk(id="chunk_2", document_id="doc_2", text="Deep neural networks perform semantic search.")
    cand = Candidate(
        chunk=chunk,
        retriever_scores={"dense": 0.88},
        retriever_ranks={"dense": 4},
        sources=["dense"],
    )

    query = "semantic search"
    feats = extractor.extract_candidate_features(query, cand)

    assert feats["bm25_score"] == 0.0
    assert feats["dense_score"] == 0.88
    assert feats["bm25_rank"] == 1000.0
    assert feats["dense_rank"] == 4.0
    assert feats["bm25_reciprocal_rank"] == 0.0
    assert feats["dense_reciprocal_rank"] == 0.25
    assert feats["rrf_score"] == pytest.approx(1.0 / (60 + 4))
    assert feats["retrieved_by_both"] == 0.0
    assert feats["exact_query_match"] == 1.0
    assert feats["rank_discrepancy"] == 996.0
    assert feats["modality_preference"] == 1.0


def test_dual_modality_candidate(extractor: FeatureExtractor) -> None:
    chunk = Chunk(id="chunk_3", document_id="doc_3", text="Information retrieval systems combine BM25 and dense embeddings.")
    cand = Candidate(
        chunk=chunk,
        retriever_scores={"bm25": 15.0, "dense": 0.92},
        retriever_ranks={"bm25": 1, "dense": 3},
        sources=["bm25", "dense"],
    )

    query = "information retrieval"
    feats = extractor.extract_candidate_features(query, cand)

    assert feats["bm25_score"] == 15.0
    assert feats["dense_score"] == 0.92
    assert feats["bm25_rank"] == 1.0
    assert feats["dense_rank"] == 3.0
    assert feats["retrieved_by_both"] == 1.0
    expected_rrf = (1.0 / 61) + (1.0 / 63)
    assert feats["rrf_score"] == pytest.approx(expected_rrf)
    assert feats["rank_discrepancy"] == 2.0

    # Modality preference: (1/3) / (1/1 + 1/3) = (0.333) / (1.333) = 0.25
    assert feats["modality_preference"] == pytest.approx(0.25)


def test_lexical_token_overlap_and_jaccard(extractor: FeatureExtractor) -> None:
    chunk = Chunk(id="chunk_4", document_id="doc_4", text="Apples and oranges are healthy fruits.")
    cand = Candidate(chunk=chunk)

    # Query: "apples bananas fruits" (2 matching out of 3 query terms)
    query = "apples bananas fruits"
    feats = extractor.extract_candidate_features(query, cand)

    assert feats["exact_query_match"] == 0.0
    assert feats["query_token_count"] == 3.0
    assert feats["doc_token_count"] == 6.0
    assert feats["length_ratio"] == 2.0

    # Overlap: {apples, fruits} in {apples, bananas, fruits} -> 2/3
    assert feats["token_overlap_ratio"] == pytest.approx(2.0 / 3.0)

    # Jaccard: intersection = {apples, fruits} (2), union = {apples, bananas, fruits, and, oranges, are, healthy} (7) -> 2/7
    assert feats["jaccard_similarity"] == pytest.approx(2.0 / 7.0)


def test_empty_query_and_doc_edge_cases(extractor: FeatureExtractor) -> None:
    chunk = Chunk(id="doc_empty", document_id="doc_e", text="")
    cand = Candidate(chunk=chunk)

    feats = extractor.extract_candidate_features("", cand)
    assert feats["exact_query_match"] == 0.0
    assert feats["token_overlap_ratio"] == 0.0
    assert feats["jaccard_similarity"] == 0.0
    assert feats["query_token_count"] == 0.0
    assert feats["doc_token_count"] == 0.0
    assert feats["modality_preference"] == 0.5


def test_extract_matrix(extractor: FeatureExtractor) -> None:
    chunk1 = Chunk(id="c1", document_id="d1", text="First chunk text")
    chunk2 = Chunk(id="c2", document_id="d2", text="Second chunk text")

    cand1 = Candidate(chunk=chunk1, retriever_scores={"bm25": 10.0}, retriever_ranks={"bm25": 1}, sources=["bm25"])
    cand2 = Candidate(chunk=chunk2, retriever_scores={"dense": 0.8}, retriever_ranks={"dense": 2}, sources=["dense"])

    pool = CandidatePool(query="chunk text", candidates=[cand1, cand2], top_k_per_retriever=10)
    matrix = extractor.extract_matrix(pool)

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (2, 16)
    assert matrix.dtype == np.float64


def test_build_ltr_dataset_graded_and_binary() -> None:
    chunk1 = Chunk(id="c1", document_id="d1", text="Scientific finding on COVID-19 transmission.")
    chunk2 = Chunk(id="c2", document_id="d2", text="Unrelated text on cooking recipes.")
    chunk3 = Chunk(id="c3", document_id="d3", text="Astrophysics discovery of black hole merger.")

    cand1 = Candidate(chunk=chunk1, retriever_scores={"bm25": 15.0}, retriever_ranks={"bm25": 1}, sources=["bm25"])
    cand2 = Candidate(chunk=chunk2, retriever_scores={"bm25": 5.0}, retriever_ranks={"bm25": 2}, sources=["bm25"])
    cand3 = Candidate(chunk=chunk3, retriever_scores={"dense": 0.95}, retriever_ranks={"dense": 1}, sources=["dense"])

    pool1 = CandidatePool(query="COVID transmission", candidates=[cand1, cand2], top_k_per_retriever=10)
    pool2 = CandidatePool(query="black hole", candidates=[cand3], top_k_per_retriever=10)

    # Case 1: Graded relevance
    case1 = BenchmarkCase(query="COVID transmission", relevant_chunk_ids=["c1"], relevance_grades={"c1": 2, "c2": 0})
    # Case 2: Binary relevance
    case2 = BenchmarkCase(query="black hole", relevant_chunk_ids=["c3"])

    dataset = build_ltr_dataset(pools=[pool1, pool2], benchmark_cases=[case1, case2])

    assert isinstance(dataset, LTRDataset)
    assert len(dataset) == 3
    assert dataset.num_queries == 2
    assert dataset.num_features == 16
    assert dataset.group_sizes == [2, 1]
    assert list(dataset.labels) == [2, 0, 1]
    assert dataset.chunk_ids == ["c1", "c2", "c3"]
    assert dataset.features.shape == (3, 16)


def test_build_ltr_dataset_length_mismatch() -> None:
    pool = CandidatePool(query="test", candidates=[])
    case1 = BenchmarkCase(query="test1", relevant_chunk_ids=[])
    case2 = BenchmarkCase(query="test2", relevant_chunk_ids=[])

    with pytest.raises(ValueError, match="Mismatched lengths"):
        build_ltr_dataset([pool], [case1, case2])
