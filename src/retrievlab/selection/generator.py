"""
Candidate generator implementations for single and multi-retriever setups.
"""

from retrievlab.models import Chunk
from retrievlab.retrieval.interface import Retriever
from retrievlab.selection.interface import CandidateGenerator
from retrievlab.selection.candidate import Candidate, CandidatePool


class MultiRetrieverCandidateGenerator(CandidateGenerator):
    """Generates candidate pools by querying multiple retrievers and merging results.

    Retrieves top_k_per_retriever results from each configured retriever, unions the
    results by chunk.id, and tracks all retriever provenance (scores, ranks, sources).
    """

    def __init__(self, retrievers: dict[str, Retriever]) -> None:
        """Initialize MultiRetrieverCandidateGenerator.

        Args:
            retrievers: Mapping of retriever names to Retriever instances,
                        e.g. {'bm25': bm25_retriever, 'dense': faiss_retriever}.

        Raises:
            ValueError: If retrievers dictionary is empty.
        """
        if not retrievers:
            raise ValueError("Must provide at least one retriever.")
        self.retrievers = retrievers

    def generate(
        self,
        query: str,
        top_k_per_retriever: int,
        chunks: list[Chunk],
    ) -> CandidatePool:
        """Generate a deduplicated candidate pool for a query across all configured retrievers.

        Args:
            query: The search query string.
            top_k_per_retriever: Number of top results to fetch from each retriever.
            chunks: Candidate corpus of chunks to retrieve from.

        Returns:
            A deduplicated CandidatePool tracking all retriever provenance.

        Raises:
            ValueError: If top_k_per_retriever <= 0.
        """
        if top_k_per_retriever <= 0:
            raise ValueError(
                f"top_k_per_retriever must be positive, got {top_k_per_retriever}"
            )

        pool_map: dict[str, Candidate] = {}

        for name, retriever in self.retrievers.items():
            results = retriever.retrieve(
                query=query, top_k=top_k_per_retriever, chunks=chunks
            )
            for rank, res in enumerate(results, start=1):
                cid = res.chunk.id
                if cid not in pool_map:
                    pool_map[cid] = Candidate(
                        chunk=res.chunk,
                        retriever_scores={name: res.score},
                        retriever_ranks={name: rank},
                        sources=[name],
                    )
                else:
                    cand = pool_map[cid]
                    cand.retriever_scores[name] = res.score
                    cand.retriever_ranks[name] = rank
                    if name not in cand.sources:
                        cand.sources.append(name)

        return CandidatePool(
            query=query,
            candidates=list(pool_map.values()),
            top_k_per_retriever=top_k_per_retriever,
        )


class SingleRetrieverCandidateGenerator(CandidateGenerator):
    """Convenience candidate generator wrapping a single retriever."""

    def __init__(self, retriever: Retriever, name: str = "primary") -> None:
        """Initialize SingleRetrieverCandidateGenerator.

        Args:
            retriever: The underlying Retriever instance.
            name: Label to use for provenance tracking. Defaults to 'primary'.
        """
        self.delegate = MultiRetrieverCandidateGenerator({name: retriever})

    def generate(
        self,
        query: str,
        top_k_per_retriever: int,
        chunks: list[Chunk],
    ) -> CandidatePool:
        """Generate a candidate pool from the single retriever.

        Args:
            query: The search query string.
            top_k_per_retriever: Number of top results to fetch from the retriever.
            chunks: Candidate corpus of chunks.

        Returns:
            A CandidatePool holding the retrieved candidates and provenance.
        """
        return self.delegate.generate(
            query=query,
            top_k_per_retriever=top_k_per_retriever,
            chunks=chunks,
        )
