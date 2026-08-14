"""Word-Level Preprocessing and Tokenization Module.

This module provides word-level tokenizers implementing the BaseTokenizer interface:
- BasicWordTokenizer: Simple lowercase regex tokenization.
- RegexTokenizer: Custom pattern-based regex tokenization.
- StopwordTokenizer: Word tokenization with customizable stopword filtering.
- StemmedTokenizer: Word tokenization with stemming using mature libraries (Snowball/Porter).
"""
from __future__ import annotations

import re
from typing import Callable

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


DEFAULT_STOPWORDS: set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or",
    "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your",
    "yours", "yourself", "yourselves",
}


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
            stopwords: Custom set or list of stopwords to filter out. Defaults to common English stopwords.
        """
        self.base_tokenizer = base_tokenizer or BasicWordTokenizer()
        if stopwords is None:
            self.stopwords = DEFAULT_STOPWORDS
        else:
            self.stopwords = {w.lower() for w in stopwords}

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text and filter out stopwords.

        Args:
            text: Input text string.

        Returns:
            List of non-stopword tokens.
        """
        tokens = self.base_tokenizer.tokenize(text)
        return [token for token in tokens if token.lower() not in self.stopwords]


class StemmedTokenizer(BaseTokenizer):
    """Word tokenizer that applies morphological stemming using mature stemmer libraries.

    Adheres to RetrievLab Design Principle: Use established, optimized libraries
    for mature linguistic algorithms while embedding first-principles understanding
    of over/under-stemming assumptions in experiments.
    """

    def __init__(
        self,
        base_tokenizer: BaseTokenizer | None = None,
        algorithm: str = "english",
    ) -> None:
        """Initialize StemmedTokenizer.

        Args:
            base_tokenizer: Underlying tokenizer to extract initial tokens.
            algorithm: Stemming algorithm ('english', 'porter', 'snowball', etc.).
        """
        self.base_tokenizer = base_tokenizer or BasicWordTokenizer()
        self.algorithm = algorithm.lower()
        self._stem_func: Callable[[str], str]
        self._stem_words_func: Callable[[list[str]], list[str]]

        # Normalized language name for Snowball/Porter algorithms
        canonical_lang = "english" if self.algorithm in ("english", "porter", "snowball", "porter2") else self.algorithm

        try:
            from py_rust_stemmers import SnowballStemmer
            self._stemmer = SnowballStemmer(canonical_lang)
            self._stem_func = self._stemmer.stem_word
            self._stem_words_func = self._stemmer.stem_words
        except (ImportError, Exception):
            try:
                import snowballstemmer
                self._stemmer = snowballstemmer.stemmer(canonical_lang)
                self._stem_func = self._stemmer.stemWord
                self._stem_words_func = self._stemmer.stemWords
            except ImportError:
                raise ImportError(
                    "A stemming backend (py_rust_stemmers or snowballstemmer) is required "
                    "for StemmedTokenizer. Ensure the virtual environment is active."
                )

    def stem(self, word: str) -> str:
        """Stem a single word token.

        Args:
            word: Input word string.

        Returns:
            Stemmed token string.
        """
        if not word:
            return ""
        return self._stem_func(word.lower())

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text and stem each token using the configured stemmer.

        Args:
            text: Input text string.

        Returns:
            List of stemmed tokens.
        """
        tokens = self.base_tokenizer.tokenize(text)
        if not tokens:
            return []
        if hasattr(self, "_stem_words_func"):
            return self._stem_words_func(tokens)
        return [self.stem(token) for token in tokens]
