"""
Vector indexing subsystem for RetrievLab.
"""

from retrievlab.indexing.interface import VectorIndex
from retrievlab.indexing.faiss import FAISSIndex, FAISSRetriever

__all__ = ["VectorIndex", "FAISSIndex", "FAISSRetriever"]
