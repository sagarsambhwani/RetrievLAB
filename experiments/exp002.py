from retrievlab.evaluation.benchmark import load_benchmark


benchmark = load_benchmark("data/benchmarks/simple.json")

print(type(benchmark.cases))
print(type(benchmark.cases[0]))
print(len(benchmark.cases))