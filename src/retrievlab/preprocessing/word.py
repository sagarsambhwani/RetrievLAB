"""Word-level preprocessing and tokenization for RetrievLab.

This module provides configurable word-level tokenization strategies:

- BasicWordTokenizer: Standard lowercase word tokenization.
- RegexTokenizer: Configurable regex-based tokenization.
- StopwordTokenizer: Tokenization with stopword filtering.
- StemmedTokenizer: Tokenization with NLTK stemming.
"""

import re

from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer

from retrievlab.preprocessing.interface import BaseTokenizer


class BasicWordTokenizer(BaseTokenizer):
    """Basic word tokenizer using a standard word-boundary regex."""

    def __init__(self, lower: bool = True) -> None:
        """Initialize the tokenizer.

        Args:
            lower: Whether to lowercase tokens.
        """
        self.lower = lower

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into word tokens.

        Args:
            text: Input text.

        Returns:
            List of word tokens.
        """
        if self.lower:
            text = text.lower()

        return re.findall(r"\b\w+\b", text)


class RegexTokenizer(BaseTokenizer):
    """Tokenizer using a configurable regular expression."""

    def __init__(
        self,
        pattern: str = r"\b\w+\b",
        lower: bool = True,
    ) -> None:
        """Initialize the tokenizer.

        Args:
            pattern: Regular expression used to identify tokens.
            lower: Whether to lowercase extracted tokens.
        """
        self.pattern = pattern
        self.lower = lower

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text using the configured regex.

        Args:
            text: Input text.

        Returns:
            List of extracted tokens.
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
    """Tokenizer that removes configured stopwords."""

    def __init__(
        self,
        base_tokenizer: BaseTokenizer | None = None,
        stopwords: set[str] | list[str] | None = None,
    ) -> None:
        """Initialize the tokenizer.

        Args:
            base_tokenizer: Tokenizer used to produce initial tokens.
            stopwords: Custom stopwords. If omitted, standard English
                stopwords are used.
        """
        self.base_tokenizer = base_tokenizer or BasicWordTokenizer()

        if stopwords is None:
            self.stopwords = set(DEFAULT_STOPWORDS)
        else:
            self.stopwords = {word.lower() for word in stopwords}

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text and remove stopwords.

        Args:
            text: Input text.

        Returns:
            Tokens excluding configured stopwords.
        """
        tokens = self.base_tokenizer.tokenize(text)

        return [
            token
            for token in tokens
            if token.lower() not in self.stopwords
        ]


class StemmedTokenizer(BaseTokenizer):
    """Tokenizer that applies an NLTK stemming algorithm to tokens.

    Supported algorithms:

    - ``porter``
    - ``snowball``
    - ``lancaster``
    """

    def __init__(
        self,
        base_tokenizer: BaseTokenizer | None = None,
        algorithm: str = "porter",
    ) -> None:
        """Initialize the tokenizer.

        Args:
            base_tokenizer: Tokenizer used to produce initial tokens.
            algorithm: Stemming algorithm to use.

        Raises:
            ValueError: If the requested algorithm is unsupported.
        """
        self.base_tokenizer = base_tokenizer or BasicWordTokenizer()

        algorithm = algorithm.lower()

        if algorithm == "porter":
            self._stemmer = PorterStemmer()
        elif algorithm == "snowball":
            self._stemmer = SnowballStemmer("english")
        elif algorithm == "lancaster":
            self._stemmer = LancasterStemmer()
        else:
            raise ValueError(
                f"Unsupported stemming algorithm: {algorithm!r}. "
                "Choose from: porter, snowball, lancaster."
            )

        self.algorithm = algorithm

    def stem(self, word: str) -> str:
        """Stem a single word.

        Args:
            word: Input word.

        Returns:
            Stemmed word.
        """
        return self._stemmer.stem(word.lower())

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text and apply the configured stemmer.

        Args:
            text: Input text.

        Returns:
            List of stemmed tokens.
        """
        tokens = self.base_tokenizer.tokenize(text)

        return [self.stem(token) for token in tokens]