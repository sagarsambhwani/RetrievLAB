import math
import re

from retrievlab.models import Chunk, SearchResult
from retrievlab.preprocessing import BaseTokenizer, BasicWordTokenizer
from retrievlab.retrieval.interface import Retriever


class BM25Retriever(Retriever):
    """BM25 lexical retriever.

    This retriever builds an inverted index over the provided chunks and
    ranks chunks using the BM25 scoring function. In accordance with
    design_principles.md Rule 2, preprocessing is injected via BaseTokenizer.
    """

    def __init__(self, tokenizer: BaseTokenizer | None = None) -> None:
        """Initialize BM25 corpus statistics.

        Args:
            tokenizer: Configurable tokenizer implementing BaseTokenizer.
                       Defaults to BasicWordTokenizer if None.
        """
        self.tokenizer = tokenizer or BasicWordTokenizer()
        self.term_frequencies: dict[str, dict[str, int]] = {}
        self.chunk_lengths: dict[str, int] = {}
        self.average_chunk_length: float = 0.0
        self.idf: dict[str, float] = {}

    def _tokenize(self, text: str) -> list[str]:
        """Convert text into lowercase word tokens.

        Args:
            text: Input text.

        Returns:
            A list of normalized word tokens.
        """
        return re.findall(r"\b\w+\b", text.lower())

    def index(self, chunks: list[Chunk]) -> None:
        """Build BM25 corpus statistics.

        This computes:

        - term frequencies for every token in every chunk
        - chunk lengths
        - average chunk length
        - inverse document frequency (IDF) for every token

        Args:
            chunks: Chunks to index.
        """
        self.term_frequencies.clear()
        self.chunk_lengths.clear()
        self.idf.clear()
        self.average_chunk_length = 0.0

        for chunk in chunks:
            tokens = self._tokenize(chunk.text)

            self.chunk_lengths[chunk.id] = len(tokens)

            for token in tokens:
                if token not in self.term_frequencies:
                    self.term_frequencies[token] = {}

                self.term_frequencies[token][chunk.id] = (
                    self.term_frequencies[token].get(chunk.id, 0) + 1
                )

        if self.chunk_lengths:
            self.average_chunk_length = (
                sum(self.chunk_lengths.values())
                / len(self.chunk_lengths)
            )

        total_chunks = len(self.chunk_lengths)

        for token, postings in self.term_frequencies.items():
            document_frequency = len(postings)
            self.idf[token] = math.log(total_chunks / document_frequency)

    def _inverse_document_frequency(self, token: str) -> float:
        """Return the precomputed IDF for a token.

        Args:
            token: Token whose IDF should be returned.

        Returns:
            The token's inverse document frequency.
        """
        return self.idf.get(token, 0.0)

    def bm25_score(
        self,
        query_tokens: list[str],
        chunk: Chunk,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        """Compute the BM25 relevance score for a chunk.

        Args:
            query_tokens: Tokenized query.
            chunk: Chunk to score.
            k1: BM25 term-frequency saturation parameter.
            b: BM25 length-normalization parameter.

        Returns:
            BM25 relevance score.
        """
        score = 0.0

        chunk_length = self.chunk_lengths.get(chunk.id, 0)

        length_norm = (
            1 - b + b * chunk_length / self.average_chunk_length
            if self.average_chunk_length > 0
            else 1.0
        )

        for token in query_tokens:
            tf = self.term_frequencies.get(token, {}).get(chunk.id, 0)

            if tf == 0:
                continue

            idf = self._inverse_document_frequency(token)

            score += idf * (
                (tf * (k1 + 1))
                / (tf + k1 * length_norm)
            )

        return score

    def retrieve(
        self,
        query: str,
        top_k: int,
        chunks: list[Chunk],
    ) -> list[SearchResult]:
        """Retrieve the top-k most relevant chunks for a query.

        Args:
            query: User query.
            top_k: Number of chunks to return.
            chunks: Candidate chunks to rank.

        Returns:
            Ranked search results sorted by descending BM25 score.
        """
        query_tokens = self._tokenize(query)

        scores = [
            (self.bm25_score(query_tokens, chunk), chunk)
            for chunk in chunks
        ]

        top_scores = sorted(
            scores,
            key=lambda item: item[0],
            reverse=True,
        )[:top_k]

        return [
            SearchResult(chunk=chunk, score=score)
            for score, chunk in top_scores
        ]