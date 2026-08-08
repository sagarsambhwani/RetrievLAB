"""Preprocessing Interface Module for RetrievLab.

This module defines the foundational abstract interface for all tokenizers and
text preprocessors across RetrievLab.

In accordance with RetrievLab Design Principles (docs/design_principles.md):
- Rule 1: One algorithm, one implementation.
- Rule 2: Behavior should be configurable.

All tokenizers operating at different levels (Word-level, N-Gram-level, Subword-level,
or Composable Pipelines) implement this interface to be injected into retrievers (e.g. BM25Retriever).
"""

from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    """Abstract Base Class for all tokenization strategies in RetrievLab.
    
    Implementations of this interface process input text and return a sequence
    of string tokens.
    """

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Convert input text into a list of normalized string tokens.

        Args:
            text: The raw text string to tokenize.

        Returns:
            A list of string tokens ready for indexing or feature extraction.
        """
        raise NotImplementedError("Subclasses must implement tokenize().")
