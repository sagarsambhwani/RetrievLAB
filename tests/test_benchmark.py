from retrievlab.evaluation.benchmark import Benchmark, BenchmarkCase, load_benchmark


def test_benchmark_models():
    case = BenchmarkCase(query="test query", relevant_chunk_ids=["chunk1", "chunk2"])
    benchmark = Benchmark(cases=[case])
    assert len(benchmark.cases) == 1
    assert benchmark.cases[0].query == "test query"
    assert benchmark.cases[0].relevant_chunk_ids == ["chunk1", "chunk2"]


def test_load_benchmark():
    benchmark = load_benchmark("data/benchmarks/simple.json")
    assert isinstance(benchmark, Benchmark)
    assert isinstance(benchmark.cases, list)
    assert len(benchmark.cases) > 0
    for case in benchmark.cases:
        assert isinstance(case, BenchmarkCase)
        assert isinstance(case.query, str)
        assert isinstance(case.relevant_chunk_ids, list)
