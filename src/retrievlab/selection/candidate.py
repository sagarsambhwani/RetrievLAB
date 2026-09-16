"""
Candidate data models for first-stage candidate generation.
"""

from pydantic import BaseModel, Field
from retrievlab.models import Chunk


class Candidate(BaseModel):
    """A retrieved candidate chunk with multi-retriever provenance metadata.
    
    Attributes:
        chunk: The underlying text Chunk.
        retriever_scores: Mapping of retriever name to raw retrieval score.
        retriever_ranks: Mapping of retriever name to 1-indexed retrieval rank.
        sources: List of retriever names that retrieved this chunk.
    """

    chunk: Chunk
    retriever_scores: dict[str, float] = Field(default_factory=dict)
    retriever_ranks: dict[str, int] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)

    @property
    def id(self) -> str:
        """Convenience property returning the chunk ID."""
        return self.chunk.id


class CandidatePool(BaseModel):
    """A deduplicated pool of candidates retrieved for a query across one or more retrievers.
    
    The pool is deduplicated by chunk ID and holds candidate provenance.
    It is not ranked; downstream feature extractors and re-rankers perform scoring.
    """

    query: str
    candidates: list[Candidate] = Field(default_factory=list)
    top_k_per_retriever: int = 0

    def __len__(self) -> int:
        return len(self.candidates)

    def __iter__(self):
        return iter(self.candidates)

    def get_candidate(self, chunk_id: str) -> Candidate | None:
        """Look up a candidate by its chunk ID.
        
        Args:
            chunk_id: The ID of the chunk to find.
            
        Returns:
            The Candidate instance if found, None otherwise.
        """
        for cand in self.candidates:
            if cand.id == chunk_id:
                return cand
        return None

    def get_chunks(self) -> list[Chunk]:
        """Return the raw Chunk objects in the pool.
        
        Returns:
            List of Chunk objects.
        """
        return [cand.chunk for cand in self.candidates]

    def get_chunk_ids(self) -> list[str]:
        """Return the list of chunk IDs in the pool.
        
        Returns:
            List of string chunk IDs.
        """
        return [cand.id for cand in self.candidates]
