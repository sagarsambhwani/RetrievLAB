"""Word-Level Preprocessing and Tokenization Module.

This module provides word-level tokenizers implementing the BaseTokenizer interface:
- BasicWordTokenizer: Simple lowercase regex tokenization.
- RegexTokenizer: Custom pattern-based regex tokenization.
- StopwordTokenizer: Word tokenization with customizable stopword filtering.
- StemmedTokenizer: Word tokenization with stemming (e.g. Porter/Snowball).
"""
import re

from retrievlab.preprocessing.interface import BaseTokenizer


class BasicWordTokenizer(BaseTokenizer):
    """Basic lowercase word tokenizer using standard word boundary regex.
    
    Splits text into alphanumeric word tokens converted to lower case.
    """

    def __init__(self, lower: bool = True) -> None:
        """Initialize BasicWordTokenizer.

        Args:
            lower: Whether to lowercase tokens (default: True).
        """
        self.lower = lower

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into basic word tokens.

        Args:
            text: Input text string.

        Returns:
            List of word tokens.
        """
        if self.lower:
            text = text.lower()
        return re.findall(r"\b\w+\b", text)


class RegexTokenizer(BaseTokenizer):
    """Tokenizer using custom regular expression pattern matching."""

    def __init__(self, pattern: str = r"\b\w+\b", lower: bool = True) -> None:
        """Initialize RegexTokenizer with regex pattern.

        Args:
            pattern: Regular expression pattern for token matching.
            lower: Whether to lowercase tokens (default: True).
        """
        self.pattern = pattern
        self.lower = lower

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text using the configured regex pattern.

        Args:
            text: Input text string.

        Returns:
            List of matched tokens.
        """
        tokens = re.findall(self.pattern, text)

        if self.lower:
            tokens = [token.lower() for token in tokens]

        return tokens


class StopwordTokenizer(BaseTokenizer):
    """Word tokenizer that filters out common stop words."""

    def __init__(
        self,
        base_tokenizer: BaseTokenizer | None = None,
        stopwords: set[str] | list[str] | None = None,
    ) -> None:
        """Initialize StopwordTokenizer.

        Args:
            base_tokenizer: Underlying tokenizer to extract initial tokens.
            stopwords: Custom set or list of stopwords to filter out.
        """
        self.base_tokenizer = base_tokenizer or BasicWordTokenizer()
        self.stopwords = set(stopwords) if stopwords is not None else set()

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text and filter out stopwords.

        Args:
            text: Input text string.

        Returns:
            List of non-stopword tokens.
        """
        raise NotImplementedError("Implement stopword filtering tokenization.")


class StemmedTokenizer(BaseTokenizer):
    """Word tokenizer that applies stemming (e.g., Porter Stemmer) to tokens."""

    def __init__(
        self,
        base_tokenizer: BaseTokenizer | None = None,
        algorithm: str = "porter",
    ) -> None:
        """Initialize StemmedTokenizer.

        Args:
            base_tokenizer: Underlying tokenizer to extract initial tokens.
            algorithm: Stemming algorithm ('porter', 'snowball', etc.).
        """
        self.base_tokenizer = base_tokenizer or BasicWordTokenizer()
        self.algorithm = algorithm

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text and stem each token.

        Args:
            text: Input text string.

        Returns:
            List of stemmed tokens.
        """
        raise NotImplementedError("Implement stemmed word tokenization.")
