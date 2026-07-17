import re

from retrievlab.retrieval.interface import Retriever
from retrievlab.models import Chunk, SearchResult

class BM25Retriever(Retriever):
    def __init__(self):
        self.term_frequencies: dict[str, dict[str, int]] = {}
        self.chunk_lengths: dict[str, int] = {}
        self.average_chunk_length: float = 0.0

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def index(self, chunks: list[Chunk]) -> None:
        """Build BM25 corpus statistics from the provided chunks."""
        self.term_frequencies.clear()
        self.chunk_lengths.clear()
        self.average_chunk_length = 0
        
        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            self.chunk_lengths[chunk.id] = len(tokens)
            for token in tokens:
                if token not in self.term_frequencies:
                    self.term_frequencies[token] = {}
                if chunk.id not in self.term_frequencies[token]:
                    self.term_frequencies[token][chunk.id] = 0
                self.term_frequencies[token][chunk.id]+=1
        self.average_chunk_length = (sum(self.chunk_lengths.values())/ len(self.chunk_lengths) if self.chunk_lengths
        else 0)
                
    def retrieve(self, query: str, top_k: int, chunks: list[Chunk]) -> list[SearchResult]:

        return []
