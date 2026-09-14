"""
Candidate generation and selection components for two-stage retrieval systems.
"""

from retrievlab.selection.candidate import Candidate, CandidatePool
from retrievlab.selection.interface import CandidateGenerator
from retrievlab.selection.generator import (
    MultiRetrieverCandidateGenerator,
    SingleRetrieverCandidateGenerator,
)

__all__ = [
    "Candidate",
    "CandidatePool",
    "CandidateGenerator",
    "MultiRetrieverCandidateGenerator",
    "SingleRetrieverCandidateGenerator",
]
