import json
from pathlib import Path
from pydantic import BaseModel, Field


class BenchmarkCase(BaseModel):
    query: str
    relevant_chunk_ids: list[str]
    relevance_grades: dict[str, int] = Field(default_factory=dict)

    def get_grade(self, chunk_id: str) -> int:
        """Return the numerical relevance grade for a given chunk ID.
        
        If relevance_grades is populated, returns the explicit integer grade (or 0 if unrated).
        Otherwise falls back to binary judgment: returns 1 if in relevant_chunk_ids, else 0.
        """
        if self.relevance_grades:
            return self.relevance_grades.get(chunk_id, 0)
        return 1 if chunk_id in self.relevant_chunk_ids else 0

    def get_all_grades(self) -> dict[str, int]:
        """Return a mapping of all rated chunk IDs to their numerical relevance grades."""
        if self.relevance_grades:
            return self.relevance_grades
        return {cid: 1 for cid in self.relevant_chunk_ids}


class Benchmark(BaseModel):
    cases: list[BenchmarkCase]


def load_benchmark(path: str | Path) -> Benchmark:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    cases = [BenchmarkCase(**item) for item in benchmark_data]
    return Benchmark(cases=cases)