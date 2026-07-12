"""
Interface for retrieval backends.
"""
from abc import ABC, abstractmethod

class Retriever(ABC):
    """Interface for retrieval backends."""
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[dict]:
        """
        Retrieve relevant documents based on a query.

        Args:
            query (str): The query string to search for.
            top_k (int): The number of top results to return.

        Returns:
            list[dict]: A list of dictionaries containing the retrieved documents and their metadata.
        """