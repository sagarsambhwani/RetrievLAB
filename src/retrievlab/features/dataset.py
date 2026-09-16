"""
Learning-to-Rank (LTR) dataset container and assembly utilities.

Converts candidate pools and ground-truth benchmark cases into structured
tabular datasets with group sizes for pairwise/listwise ranking models (e.g. LightGBM).
"""

from dataclasses import dataclass, field
import numpy as np

from retrievlab.evaluation.benchmark import BenchmarkCase
from retrievlab.features.extractor import FeatureExtractor
from retrievlab.selection.candidate import CandidatePool


@dataclass
class LTRDataset:
    """Structured dataset for Learning-to-Rank algorithms.

    Attributes:
        features: 2D numpy array of shape (N_candidates, D_features).
        labels: 1D numpy array of relevance scores/grades of shape (N_candidates,).
        group_sizes: List of integers representing number of candidates per query group.
        query_ids: List of query identifiers corresponding to each candidate row.
        chunk_ids: List of chunk identifiers corresponding to each candidate row.
        feature_names: Ordered list of feature column names.
    """

    features: np.ndarray
    labels: np.ndarray
    group_sizes: list[int] = field(default_factory=list)
    query_ids: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        """Return the total number of candidate rows in the dataset."""
        return len(self.labels)

    @property
    def num_queries(self) -> int:
        """Return the number of unique query groups."""
        return len(self.group_sizes)

    @property
    def num_features(self) -> int:
        """Return the number of feature columns."""
        return len(self.feature_names)


def build_ltr_dataset(
    pools: list[CandidatePool],
    benchmark_cases: list[BenchmarkCase],
    extractor: FeatureExtractor | None = None,
) -> LTRDataset:
    """Build an LTRDataset from candidate pools and corresponding benchmark cases.

    Args:
        pools: List of CandidatePool objects (one per query).
        benchmark_cases: List of BenchmarkCase objects corresponding to the pools.
        extractor: Optional FeatureExtractor instance. Defaults to standard FeatureExtractor().

    Returns:
        An LTRDataset containing features, labels, and query group metadata.

    Raises:
        ValueError: If length of pools does not match length of benchmark_cases.
    """
    if len(pools) != len(benchmark_cases):
        raise ValueError(
            f"Mismatched lengths: got {len(pools)} candidate pools but {len(benchmark_cases)} benchmark cases."
        )

    if extractor is None:
        extractor = FeatureExtractor()

    feature_names = extractor.get_feature_names()
    all_features: list[list[float]] = []
    all_labels: list[int] = []
    group_sizes: list[int] = []
    query_ids: list[str] = []
    chunk_ids: list[str] = []

    for pool, case in zip(pools, benchmark_cases):
        if not pool.candidates:
            continue

        group_size = len(pool.candidates)
        group_sizes.append(group_size)

        for cand in pool.candidates:
            feat_dict = extractor.extract_candidate_features(pool.query, cand)
            feat_row = [feat_dict[name] for name in feature_names]
            label = case.get_grade(cand.id)

            all_features.append(feat_row)
            all_labels.append(label)
            query_ids.append(pool.query)
            chunk_ids.append(cand.id)

    if not all_features:
        feat_matrix = np.empty((0, len(feature_names)), dtype=np.float64)
        label_array = np.empty((0,), dtype=np.int64)
    else:
        feat_matrix = np.array(all_features, dtype=np.float64)
        label_array = np.array(all_labels, dtype=np.int64)

    return LTRDataset(
        features=feat_matrix,
        labels=label_array,
        group_sizes=group_sizes,
        query_ids=query_ids,
        chunk_ids=chunk_ids,
        feature_names=feature_names,
    )
