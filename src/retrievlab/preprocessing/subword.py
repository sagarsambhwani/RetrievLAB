"""Subword Level Tokenization Module.

This module provides subword tokenization adapters implementing BaseTokenizer:
- SubwordTokenizer: Adapter wrapping BPE/tiktoken or HuggingFace tokenizers to align
  lexical vocabulary with dense embedding subword models.
"""

from retrievlab.preprocessing.interface import BaseTokenizer


class SubwordTokenizer(BaseTokenizer):
    """Subword tokenizer adapter wrapping external BPE / Tiktoken / HuggingFace tokenizers.
    
    Generates subword tokens for vocabulary alignment with neural dense retrievers.
    """

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        model_name: str | None = None,
    ) -> None:
        """Initialize SubwordTokenizer.

        Args:
            encoding_name: Tiktoken encoding scheme name (e.g. 'cl100k_base' or 'o200k_base').
            model_name: Optional HuggingFace model name / path if using HF transformers tokenizer.
        """
        self.encoding_name = encoding_name
        self.model_name = model_name

    def tokenize(self, text: str) -> list[str]:
        """Convert input text into subword token strings.

        Args:
            text: Input text string.

        Returns:
            List of subword token strings.
        """
        raise NotImplementedError("Implement subword / BPE tokenization.")
