import json
from pathlib import Path
from pydantic import BaseModel


class BenchmarkCase(BaseModel):
    query: str
    relevant_chunk_ids: list[str]


class Benchmark(BaseModel):
    cases: list[BenchmarkCase]


def load_benchmark(path: str | Path) -> Benchmark:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    cases = [BenchmarkCase(**item) for item in benchmark_data]
    return Benchmark(cases=cases)