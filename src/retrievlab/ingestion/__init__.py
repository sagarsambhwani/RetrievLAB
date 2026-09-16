"""
Document and benchmark ingestion subsystem.
"""

from retrievlab.ingestion.loader import DocumentLoader
from retrievlab.ingestion.beir import BEIRLoader

__all__ = ["DocumentLoader", "BEIRLoader"]
