"""
Experiment 015

Question:
How sensitive is BM25 retrieval quality to k1 and b on our benchmark?

Expected Result:
- Sweeps a 5x5 grid of (k1, b) configuration parameters over simple2.json:
  k1 in [0.5, 0.9, 1.2, 1.5, 2.0]
  b  in [0.1, 0.3, 0.5, 0.75, 0.9]
- Measures Recall@5, Precision@5, and MRR for each configuration.
- Identifies the best configuration per metric and compares against default (k1=1.5, b=0.75).
- Quantifies whether parameter tuning materially impacts retrieval performance on this corpus.
"""

from pathlib import Path

from retrievlab.chunking.markdown import MarkdownChunker
from retrievlab.evaluation import evaluate_retriever, load_benchmark
from retrievlab.ingestion.loader import DocumentLoader
from retrievlab.preprocessing import BasicWordTokenizer
from retrievlab.retrieval.bm25 import BM25Retriever


def run_experiment() -> None:
    print("=" * 118)
    print("Experiment 015: BM25 Parameter Sensitivity Sweep (k1 and b)")
    print("=" * 118)

    # 1. Load and chunk documents
    loader = DocumentLoader()
    chunker = MarkdownChunker()
    raw_path = Path("data/raw")
    documents = loader.load(raw_path)

    chunks = []
    for doc in documents:
        chunks.extend(chunker.chunk(doc))

    print("\n1. Corpus Loading:")
    print(f"   Loaded {len(documents)} document(s) producing {len(chunks)} chunk(s).")

    # 2. Load benchmark dataset
    benchmark_path = "data/benchmarks/simple2.json"
    benchmark = load_benchmark(benchmark_path)
    print("\n2. Benchmark Dataset:")
    print(f"   Loaded {len(benchmark.cases)} benchmark cases from '{benchmark_path}'.\n")

    # 3. Define parameter grid
    k1_values = [0.5, 0.9, 1.2, 1.5, 2.0]
    b_values = [0.1, 0.3, 0.5, 0.75, 0.9]

    tokenizer = BasicWordTokenizer(lower=True)
    grid_results = []

    print("3. Running Parameter Grid Sweep (25 Configurations)...")
    print(f"{'Config #':<10} | {'k1':<6} | {'b':<6} | {'Recall@5':<10} | {'Precision@5':<14} | {'MRR':<8} | {'Notes'}")
    print("-" * 80)

    config_idx = 1
    default_result = None

    for k1 in k1_values:
        for b in b_values:
            retriever = BM25Retriever(tokenizer=tokenizer, k1=k1, b=b)
            retriever.index(chunks)

            res = evaluate_retriever(
                retriever=retriever,
                benchmark=benchmark,
                chunks=chunks,
                k=5,
                retriever_name=f"BM25(k1={k1}, b={b})",
            )

            is_default = (k1 == 1.5 and b == 0.75)
            note = "Default" if is_default else ""

            if is_default:
                default_result = (k1, b, res.recall_at_k, res.precision_at_k, res.mrr)

            grid_results.append({
                "k1": k1,
                "b": b,
                "recall": res.recall_at_k,
                "precision": res.precision_at_k,
                "mrr": res.mrr,
            })

            print(f"{config_idx:<10} | {k1:<6.2f} | {b:<6.2f} | {res.recall_at_k:<10.4f} | {res.precision_at_k:<14.4f} | {res.mrr:<8.4f} | {note}")
            config_idx += 1

    print("-" * 80)

    # 4. Identify Best Configuration Per Metric
    best_recall = max(r["recall"] for r in grid_results)
    best_precision = max(r["precision"] for r in grid_results)
    best_mrr = max(r["mrr"] for r in grid_results)

    best_recall_configs = [r for r in grid_results if r["recall"] == best_recall]
    best_precision_configs = [r for r in grid_results if r["precision"] == best_precision]
    best_mrr_configs = [r for r in grid_results if r["mrr"] == best_mrr]

    print("\n4. Best Parameter Configurations Per Metric:")
    print(f"   A) Best for Recall@5:      {best_recall:.4f}")
    for c in best_recall_configs[:3]:
        print(f"      - k1={c['k1']}, b={c['b']}")
    if len(best_recall_configs) > 3:
        print(f"      - (and {len(best_recall_configs) - 3} other configurations tied)")

    print(f"\n   B) Best for Precision@5:   {best_precision:.4f}")
    for c in best_precision_configs[:3]:
        print(f"      - k1={c['k1']}, b={c['b']}")
    if len(best_precision_configs) > 3:
        print(f"      - (and {len(best_precision_configs) - 3} other configurations tied)")

    print(f"\n   C) Best for MRR:           {best_mrr:.4f}")
    for c in best_mrr_configs[:3]:
        print(f"      - k1={c['k1']}, b={c['b']}")
    if len(best_mrr_configs) > 3:
        print(f"      - (and {len(best_mrr_configs) - 3} other configurations tied)")

    # 5. Comparison against Default
    if default_result:
        def_k1, def_b, def_rec, def_prec, def_mrr = default_result
        print("\n5. Default vs Best Comparison:")
        print(f"   {'Configuration':<25} | {'k1':<6} | {'b':<6} | {'Recall@5':<10} | {'Precision@5':<14} | {'MRR':<8}")
        print(f"   {'-' * 76}")
        print(f"   {'Default Okapi BM25':<25} | {def_k1:<6.2f} | {def_b:<6.2f} | {def_rec:<10.4f} | {def_prec:<14.4f} | {def_mrr:<8.4f}")
        print(f"   {'Best MRR Configuration':<25} | {best_mrr_configs[0]['k1']:<6.2f} | {best_mrr_configs[0]['b']:<6.2f} | {best_mrr_configs[0]['recall']:<10.4f} | {best_mrr_configs[0]['precision']:<14.4f} | {best_mrr_configs[0]['mrr']:<8.4f}")

    # 6. Sensitivity Analysis Summary
    print("\n6. Sensitivity Analysis Findings:")
    all_recalls = [r["recall"] for r in grid_results]
    all_mrrs = [r["mrr"] for r in grid_results]
    min_rec, max_rec = min(all_recalls), max(all_recalls)
    min_mrr, max_mrr = min(all_mrrs), max(all_mrrs)

    print(f"   - Recall@5 Range:  [{min_rec:.4f}, {max_rec:.4f}] (delta = {max_rec - min_rec:.4f})")
    print(f"   - MRR Range:       [{min_mrr:.4f}, {max_mrr:.4f}] (delta = {max_mrr - min_mrr:.4f})")
    if max_mrr == min_mrr and max_rec == min_rec:
        print("   - Observation: BM25 metrics are flat/robust across parameter ranges on this corpus.")
    else:
        print(f"   - Observation: Parameter tuning yields up to +{max_mrr - min_mrr:.4f} MRR variance across the grid.")


if __name__ == "__main__":
    run_experiment()
