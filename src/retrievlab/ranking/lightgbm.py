"""
Learning-to-Rank re-ranker using LightGBM LambdaMART.

Optimizes pairwise and listwise nDCG over multi-signal tabular features
(lexical, dense, reciprocal rank, token overlap, and modality divergence)
for sub-millisecond candidate re-ranking.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, Union

import lightgbm as lgb
import numpy as np

if TYPE_CHECKING:
    from retrievlab.features.dataset import LTRDataset
    from retrievlab.features.extractor import FeatureExtractor

from retrievlab.models import Chunk, SearchResult
from retrievlab.ranking.interface import ReRanker
from retrievlab.selection.candidate import Candidate, CandidatePool

logger = logging.getLogger(__name__)


class LightGBMRanker(ReRanker):
    """Gradient Boosted Decision Tree (GBDT) re-ranker using LightGBM LambdaMART.

    Trained on query-grouped candidate pools using pairwise LambdaMART loss
    to optimize ranking metrics (e.g. nDCG@5, nDCG@10).
    """

    DEFAULT_PARAMS: dict[str, Any] = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [5, 10],
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 10,
        "random_state": 42,
        "verbosity": -1,
        "importance_type": "gain",
        "n_jobs": -1,
    }

    def __init__(
        self,
        model_params: dict[str, Any] | None = None,
        extractor: FeatureExtractor | None = None,
    ) -> None:
        """Initialize LightGBMRanker with hyperparameters and feature extractor.

        Args:
            model_params: Optional dictionary of LightGBM hyperparameters to override defaults.
            extractor: Optional FeatureExtractor instance. Defaults to FeatureExtractor().
        """
        self.model_params = dict(self.DEFAULT_PARAMS)
        if model_params:
            self.model_params.update(model_params)

        if extractor is None:
            from retrievlab.features.extractor import FeatureExtractor as _FE
            self.extractor = _FE()
        else:
            self.extractor = extractor

        self.model: lgb.LGBMRanker | None = None
        self._booster: lgb.Booster | None = None
        self.feature_names_: list[str] = []
        self.is_fitted: bool = False

    def fit(
        self,
        train_dataset: LTRDataset,
        eval_dataset: LTRDataset | None = None,
    ) -> LightGBMRanker:
        """Fit the LambdaMART ranker on an LTRDataset.

        Args:
            train_dataset: Training LTRDataset containing features, labels, and group sizes.
            eval_dataset: Optional validation LTRDataset for early stopping and metric evaluation.

        Returns:
            self (fitted ranker instance).

        Raises:
            ValueError: If train_dataset is empty or group sizes do not sum to total samples.
        """
        if len(train_dataset) == 0:
            raise ValueError("Cannot fit LightGBMRanker on empty dataset.")

        if sum(train_dataset.group_sizes) != len(train_dataset):
            raise ValueError(
                f"Sum of group_sizes ({sum(train_dataset.group_sizes)}) must equal "
                f"number of candidate rows ({len(train_dataset)})."
            )

        self.feature_names_ = list(train_dataset.feature_names)
        
        # Pop eval_at so it is passed directly to fit() without triggering LGBM UserWarning
        init_params = dict(self.model_params)
        eval_at = init_params.pop("eval_at", [5, 10])
        self.model = lgb.LGBMRanker(**init_params)

        eval_set = None
        eval_group = None
        if eval_dataset is not None and len(eval_dataset) > 0:
            eval_set = [(eval_dataset.features, eval_dataset.labels)]
            eval_group = [eval_dataset.group_sizes]

        logger.info(
            f"Fitting LightGBMRanker on {len(train_dataset)} candidates across "
            f"{train_dataset.num_queries} queries with {len(self.feature_names_)} features..."
        )

        self.model.fit(
            X=train_dataset.features,
            y=train_dataset.labels,
            group=train_dataset.group_sizes,
            eval_set=eval_set,
            eval_group=eval_group,
            eval_at=eval_at,
            feature_name=self.feature_names_,
        )


        self._booster = self.model.booster_
        self.is_fitted = True
        logger.info("LightGBMRanker fitting completed successfully.")
        return self

    def rerank(
        self,
        query: str,
        candidates: Union[CandidatePool, Sequence[Union[Chunk, Candidate]]],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Re-rank candidate chunks or CandidatePool using the fitted LightGBM model.

        Args:
            query: The search query string.
            candidates: A CandidatePool or sequence of Chunk/Candidate objects.
            top_k: Optional maximum number of top results to return.

        Returns:
            List of SearchResult objects sorted descending by predicted relevance score.

        Raises:
            RuntimeError: If called before the model has been fitted.
            TypeError: If candidates are not a CandidatePool or sequence of Chunk/Candidate.
        """
        if not self.is_fitted or (self.model is None and self._booster is None):
            raise RuntimeError("LightGBMRanker must be fitted before calling rerank().")

        if isinstance(candidates, CandidatePool):
            raw_cands = list(candidates.candidates)
        elif isinstance(candidates, Sequence):
            raw_cands = list(candidates)
        else:
            raise TypeError(
                f"Expected Chunk, Candidate, or CandidatePool, got {type(candidates).__name__}"
            )

        if not raw_cands:
            return []

        norm_cands: list[Candidate] = []
        for item in raw_cands:
            if isinstance(item, Candidate):
                norm_cands.append(item)
            elif isinstance(item, Chunk):
                norm_cands.append(Candidate(chunk=item, sources=[]))
            else:
                raise TypeError(
                    f"Expected item in candidates to be Chunk or Candidate, got {type(item).__name__}"
                )

        feature_rows: list[list[float]] = []
        for cand in norm_cands:
            feat_dict = self.extractor.extract_candidate_features(query, cand)
            row = [float(feat_dict.get(fname, 0.0)) for fname in self.feature_names_]
            feature_rows.append(row)

        X = np.array(feature_rows, dtype=np.float64)

        if self._booster is not None:
            scores = self._booster.predict(X)
        elif self.model is not None:
            scores = self.model.predict(X)
        else:
            raise RuntimeError("No underlying booster or model found.")

        scored_pairs = list(zip(norm_cands, scores))
        scored_pairs.sort(key=lambda pair: float(pair[1]), reverse=True)

        if top_k is not None and top_k > 0:
            scored_pairs = scored_pairs[:top_k]

        return [
            SearchResult(chunk=cand.chunk, score=float(score))
            for cand, score in scored_pairs
        ]

    def get_feature_importances(self, importance_type: str = "gain") -> dict[str, float]:
        """Return feature importances sorted descending.

        Args:
            importance_type: 'gain' (total gain of splits) or 'split' (number of times used).
                             Defaults to 'gain'.

        Returns:
            Dictionary mapping feature name to importance score.

        Raises:
            RuntimeError: If called before model is fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        booster = self._booster or (self.model.booster_ if self.model else None)
        if booster is None:
            raise RuntimeError("Underlying booster not available.")

        raw_imp = booster.feature_importance(importance_type=importance_type)
        imp_dict = {
            fname: float(val) for fname, val in zip(self.feature_names_, raw_imp)
        }
        return dict(sorted(imp_dict.items(), key=lambda item: item[1], reverse=True))

    def save(self, directory_path: str | Path) -> None:
        """Save fitted model and metadata to directory.

        Args:
            directory_path: Directory where model.txt and metadata.json will be saved.

        Raises:
            RuntimeError: If called before model is fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted LightGBMRanker.")

        booster = self._booster or (self.model.booster_ if self.model else None)
        if booster is None:
            raise RuntimeError("Underlying booster not available to save.")

        save_dir = Path(directory_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        model_file = save_dir / "model.txt"
        metadata_file = save_dir / "metadata.json"

        booster.save_model(str(model_file))

        metadata = {
            "model_params": self.model_params,
            "feature_names": self.feature_names_,
            "is_fitted": self.is_fitted,
        }
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved LightGBMRanker to {save_dir}")

    @classmethod
    def load(
        cls,
        directory_path: str | Path,
        extractor: FeatureExtractor | None = None,
    ) -> LightGBMRanker:
        """Load fitted model and metadata from directory.

        Args:
            directory_path: Directory containing model.txt and metadata.json.
            extractor: Optional FeatureExtractor instance.

        Returns:
            Fitted LightGBMRanker instance.

        Raises:
            FileNotFoundError: If model.txt or metadata.json is missing.
        """
        load_dir = Path(directory_path)
        model_file = load_dir / "model.txt"
        metadata_file = load_dir / "metadata.json"

        if not model_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(f"Missing model.txt or metadata.json in {load_dir}")

        with metadata_file.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        ranker = cls(model_params=metadata.get("model_params"), extractor=extractor)
        ranker.feature_names_ = list(metadata["feature_names"])
        ranker._booster = lgb.Booster(model_file=str(model_file))
        ranker.is_fitted = True

        logger.info(f"Loaded LightGBMRanker from {load_dir}")
        return ranker
