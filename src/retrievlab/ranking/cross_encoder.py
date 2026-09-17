"""
Cross-Encoder neural re-ranker implementation for second-stage candidate re-ranking.
"""

from typing import Any, Sequence, Union
import numpy as np
from sentence_transformers import CrossEncoder

from retrievlab.models import Chunk, SearchResult
from retrievlab.ranking.interface import ReRanker
from retrievlab.selection.candidate import Candidate, CandidatePool


class CrossEncoderReRanker(ReRanker):
    """Neural re-ranker using full-sequence cross-attention Transformer models.

    Concatenates the query and document into a single sequence `(query, document)`
    and performs cross-attention over all tokens to produce fine-grained relevance logits.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
        device: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize CrossEncoderReRanker.

        Args:
            model_name: HuggingFace model hub ID or local path.
                        Defaults to 'cross-encoder/ms-marco-MiniLM-L-6-v2'.
            batch_size: Batch size for model inference. Defaults to 32.
            device: Device to load the model on ('cpu', 'cuda', etc.). Defaults to auto-detect.
            model_kwargs: Additional keyword arguments passed to CrossEncoder.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

        kwargs: dict[str, Any] = model_kwargs or {}
        if device is not None:
            kwargs["device"] = device

        self.model = CrossEncoder(model_name, **kwargs)

    def _extract_chunks(
        self,
        candidates: Union[CandidatePool, Sequence[Union[Chunk, Candidate]]],
    ) -> list[Chunk]:
        """Extract a flat list of Chunk objects from various candidate input types."""
        if isinstance(candidates, CandidatePool):
            return candidates.get_chunks()

        chunks: list[Chunk] = []
        for item in candidates:
            if isinstance(item, Candidate):
                chunks.append(item.chunk)
            elif isinstance(item, Chunk):
                chunks.append(item)
            else:
                raise TypeError(
                    f"Expected Chunk, Candidate, or CandidatePool, got {type(item).__name__}"
                )
        return chunks

    def rerank(
        self,
        query: str,
        candidates: Union[CandidatePool, Sequence[Union[Chunk, Candidate]]],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Re-rank candidate chunks using full transformer cross-attention scoring.

        Args:
            query: The search query string.
            candidates: Candidate pool or list of candidate chunks.
            top_k: Maximum number of ranked results to return. If None, returns all.

        Returns:
            List of SearchResult objects sorted descending by neural score.
        """
        chunks = self._extract_chunks(candidates)
        if not chunks:
            return []

        # Build (query, text) sequence pairs
        pairs = [(query, chunk.text or "") for chunk in chunks]

        # Predict cross-encoder relevance scores
        raw_scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        scores = np.asarray(raw_scores, dtype=np.float64)

        # Pair each chunk with its score and sort descending
        scored_results: list[SearchResult] = [
            SearchResult(chunk=chunk, score=float(score))
            for chunk, score in zip(chunks, scores)
        ]

        scored_results.sort(key=lambda res: res.score, reverse=True)

        if top_k is not None and top_k > 0:
            return scored_results[:top_k]

        return scored_results
