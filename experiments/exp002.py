import json
from pathlib import Path

from retrievlab.evaluation.benchmark import BenchmarkCase


path = Path("data/benchmarks/simple.json")

with path.open("r") as file:
    data = json.load(file)

cases = []

for item in data:
    case = BenchmarkCase(**item)
    cases.append(case)

print(type(cases))
print(type(cases[0]))
print(len(cases))