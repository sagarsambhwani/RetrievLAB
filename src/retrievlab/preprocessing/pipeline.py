"""Composable Preprocessing Pipeline Module.

This module provides PipelineTokenizer:
- PipelineTokenizer: Enables chaining multiple preprocessing steps (normalizers,
  tokenizers, filters, transformers) sequentially into a unified execution pipeline.
"""

from typing import Callable
from retrievlab.preprocessing.interface import BaseTokenizer


class PipelineTokenizer(BaseTokenizer):
    """Composable Preprocessing Pipeline Tokenizer.
    
    Chains text normalizers, tokenizers, and post-token filters sequentially.
    """

    def __init__(
        self,
        normalizers: list[Callable[[str], str]] | None = None,
        base_tokenizer: BaseTokenizer | None = None,
        filters: list[Callable[[list[str]], list[str]]] | None = None,
    ) -> None:
        """Initialize PipelineTokenizer.

        Args:
            normalizers: Sequence of string normalization functions (e.g., lowercasing, unicode normalization).
            base_tokenizer: Primary tokenizer implementation.
            filters: Sequence of post-tokenization filter/transformation functions (e.g. stopword filter, stemmer).
        """
        self.normalizers = normalizers or []
        self.base_tokenizer = base_tokenizer
        self.filters = filters or []

    def tokenize(self, text: str) -> list[str]:
        """Process text through normalization stages, tokenize, and apply post-filters.

        Args:
            text: Input text string.

        Returns:
            List of processed tokens.
        """
        raise NotImplementedError("Implement composable pipeline tokenization.")
