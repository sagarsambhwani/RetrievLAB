"""N-Gram Level Tokenization Module.

This module provides N-Gram tokenization strategies implementing BaseTokenizer:
- CharNGramTokenizer: Extracts character-level n-grams across words or sentences.
- WordNGramTokenizer: Extracts word-level n-grams (bigrams, trigrams, etc.).
"""

from retrievlab.preprocessing.interface import BaseTokenizer


class CharNGramTokenizer(BaseTokenizer):
    """Character-level N-Gram tokenizer.
    
    Generates character n-grams of specified min and max length. Useful for handling
    typos, spelling variations, sub-word morphology, and fuzzy matching in lexical search.
    """

    def __init__(self, min_n: int = 3, max_n: int = 5, lower: bool = True) -> None:
        """Initialize CharNGramTokenizer.

        Args:
            min_n: Minimum character n-gram length (default: 3).
            max_n: Maximum character n-gram length (default: 5).
            lower: Whether to lowercase input text before n-gram extraction (default: True).
        """
        self.min_n = min_n
        self.max_n = max_n
        self.lower = lower

    def tokenize(self, text: str) -> list[str]:
        """Generate character n-gram tokens from input text.

        Args:
            text: Input text string.

        Returns:
            List of character n-gram tokens.
        """
        raise NotImplementedError("Implement character n-gram tokenization.")


class WordNGramTokenizer(BaseTokenizer):
    """Word-level N-Gram tokenizer.
    
    Generates word n-grams (unigrams, bigrams, trigrams) from tokenized words.
    """

    def __init__(
        self,
        base_tokenizer: BaseTokenizer | None = None,
        min_n: int = 1,
        max_n: int = 2,
    ) -> None:
        """Initialize WordNGramTokenizer.

        Args:
            base_tokenizer: Underlying word tokenizer.
            min_n: Minimum word n-gram range (default: 1).
            max_n: Maximum word n-gram range (default: 2).
        """
        self.base_tokenizer = base_tokenizer
        self.min_n = min_n
        self.max_n = max_n

    def tokenize(self, text: str) -> list[str]:
        """Generate word n-gram tokens from input text.

        Args:
            text: Input text string.

        Returns:
            List of word n-gram tokens (e.g. ['information', 'retrieval', 'information retrieval']).
        """
        raise NotImplementedError("Implement word n-gram tokenization.")
