"""
Experiment 021: Out-of-Domain BEIR SciFact Baselines & Hybrid Rank Fusion.

Evaluates BM25, FAISS Dense, Hybrid (RRF 1:1), and Hybrid (RRF 1:2) across all 300
SciFact test queries at K=5 and K=10, performs query-level complementarity analysis,
and outputs results/sprint_3/exp021_beir_scifact_baselines.md with strictly data-driven prose.
"""

from pathlib import Path
import numpy as np

from retrievlab.ingestion import BEIRLoader
from retrievlab.retrieval import BM25Retriever, HybridRetriever
from retrievlab.indexing import FAISSRetriever
from retrievlab.embeddings.fastembed import FastEmbedClient
from retrievlab.evaluation import evaluate_retriever, hit_at_k
from retrievlab.models import Chunk


def load_cached_embeddings(chunks: list[Chunk], cache_path: Path) -> list[Chunk]:
    """Load precomputed embedding vectors from disk into chunks."""
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Embeddings cache not found at {cache_path}. "
            "Run exp020_beir_scifact_loader.py first to generate the cache."
        )
    print(f"Loading cached embeddings from {cache_path}...")
    emb_matrix = np.load(cache_path)
    for i, chunk in enumerate(chunks):
        chunk.embedding = emb_matrix[i].tolist()
    print(f"Loaded {len(emb_matrix)} cached vectors (dim: {emb_matrix.shape[1]}).")
    return chunks


def run_experiment() -> None:
    print("=== Step 1: Loading SciFact BEIR Dataset ===")
    loader = BEIRLoader("scifact")
    chunks = loader.load_corpus()
    benchmark = loader.load_benchmark(split="test")
    total_queries = len(benchmark.cases)
    print(f"Loaded {len(chunks)} documents, {total_queries} test queries.")

    print("\n=== Step 2: Loading Cached Embeddings ===")
    cache_path = Path("data/processed/scifact_embeddings.npy")
    chunks = load_cached_embeddings(chunks, cache_path)

    print("\n=== Step 3: Initializing Retrievers ===")
    client = FastEmbedClient()

    # 1. BM25
    bm25 = BM25Retriever()
    bm25.index(chunks)

    # 2. FAISS Dense
    faiss_retriever = FAISSRetriever(client)

    # 3. Hybrid RRF 1:1 (Balanced)
    hybrid_1_1 = HybridRetriever(
        retrievers=[bm25, faiss_retriever],
        weights=[1.0, 1.0],
    )

    # 4. Hybrid RRF 1:2 (Dense-Biased)
    hybrid_1_2 = HybridRetriever(
        retrievers=[bm25, faiss_retriever],
        weights=[1.0, 2.0],
    )

    retrievers = [
        ("BM25", bm25),
        ("FAISS Dense", faiss_retriever),
        ("Hybrid RRF 1:1", hybrid_1_1),
        ("Hybrid RRF 1:2", hybrid_1_2),
    ]

    print("\n=== Step 4: Evaluating Retrievers at K=5 and K=10 ===")
    results_k5 = {}
    results_k10 = {}

    for name, ret in retrievers:
        print(f"Evaluating {name} at K=5...")
        eval_5 = evaluate_retriever(ret, benchmark, chunks, k=5, retriever_name=name)
        results_k5[name] = eval_5

        print(f"Evaluating {name} at K=10...")
        eval_10 = evaluate_retriever(ret, benchmark, chunks, k=10, retriever_name=name)
        results_k10[name] = eval_10

    print("\n=== Step 5: Complementarity Analysis (K=5) ===")
    both_succeed = 0
    dense_only_succeed = 0
    bm25_only_succeed = 0
    both_fail = 0

    for case in benchmark.cases:
        bm25_res = bm25.retrieve(case.query, top_k=5, chunks=chunks)
        dense_res = faiss_retriever.retrieve(case.query, top_k=5, chunks=chunks)

        bm25_hit = hit_at_k(bm25_res, case, k=5) == 1.0
        dense_hit = hit_at_k(dense_res, case, k=5) == 1.0

        if bm25_hit and dense_hit:
            both_succeed += 1
        elif dense_hit and not bm25_hit:
            dense_only_succeed += 1
        elif bm25_hit and not dense_hit:
            bm25_only_succeed += 1
        else:
            both_fail += 1

    print(f"Both Succeed:       {both_succeed} ({both_succeed/total_queries*100:.1f}%)")
    print(f"Dense Wins Only:    {dense_only_succeed} ({dense_only_succeed/total_queries*100:.1f}%)")
    print(f"BM25 Wins Only:     {bm25_only_succeed} ({bm25_only_succeed/total_queries*100:.1f}%)")
    print(f"Both Fail:          {both_fail} ({both_fail/total_queries*100:.1f}%)")

    print("\n=== Step 6: Generating Markdown Report ===")
    out_dir = Path("results/sprint_3")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "exp021_beir_scifact_baselines.md"

    b_k5 = results_k5["BM25"]
    d_k5 = results_k5["FAISS Dense"]
    h1_k5 = results_k5["Hybrid RRF 1:1"]
    h2_k5 = results_k5["Hybrid RRF 1:2"]

    b_k10 = results_k10["BM25"]
    d_k10 = results_k10["FAISS Dense"]
    h1_k10 = results_k10["Hybrid RRF 1:1"]
    h2_k10 = results_k10["Hybrid RRF 1:2"]

    delta_recall_k5 = (d_k5.recall_at_k - b_k5.recall_at_k) * 100
    delta_ndcg_k5 = d_k5.ndcg_at_k - b_k5.ndcg_at_k
    union_ceiling = (both_succeed + dense_only_succeed + bm25_only_succeed) / total_queries * 100

    report_content = f"""# Experiment 021 — BEIR SciFact Baselines & Hybrid Rank Fusion

**Date:** 2026-09-15  
**Status:** ✅ Completed

---

## 1. Research Question

> **Does hybrid retrieval improve out-of-domain retrieval quality on SciFact compared with standalone BM25 and Dense retrieval?**

We also investigate whether RRF successfully combines lexical precision with semantic recall, using both binary and graded relevance metrics.

---

## 2. Experiment Setup

| Property | Configuration |
|---|---|
| **Benchmark** | BEIR SciFact |
| **Queries** | 300 test queries |
| **Corpus** | 5,183 scientific abstracts |
| **Location** | `data/beir/scifact/` |
| **Evaluation Cutoffs** | @5, @10 |
| **Retrievers** | BM25, FAISS Dense |
| **Hybrid Methods** | RRF 1:1, RRF 1:2 Dense-biased |
| **Metrics** | Recall, Precision, MRR, nDCG, MAP, Hit |
| **Relevance** | Graded |

---

## 3. Results

### @5 Results

| System | Recall@5 | MRR | Precision@5 | nDCG@5 | MAP@5 | Hit@5 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | {b_k5.recall_at_k:.4f} | {b_k5.mrr:.4f} | {b_k5.precision_at_k:.4f} | {b_k5.ndcg_at_k:.4f} | {b_k5.map_at_k:.4f} | {b_k5.hit_at_k:.4f} |
| FAISS Dense | {d_k5.recall_at_k:.4f} | {d_k5.mrr:.4f} | {d_k5.precision_at_k:.4f} | {d_k5.ndcg_at_k:.4f} | {d_k5.map_at_k:.4f} | {d_k5.hit_at_k:.4f} |
| Hybrid RRF 1:1 | {h1_k5.recall_at_k:.4f} | {h1_k5.mrr:.4f} | {h1_k5.precision_at_k:.4f} | {h1_k5.ndcg_at_k:.4f} | {h1_k5.map_at_k:.4f} | {h1_k5.hit_at_k:.4f} |
| Hybrid RRF 1:2 | {h2_k5.recall_at_k:.4f} | {h2_k5.mrr:.4f} | {h2_k5.precision_at_k:.4f} | {h2_k5.ndcg_at_k:.4f} | {h2_k5.map_at_k:.4f} | {h2_k5.hit_at_k:.4f} |

### @10 Results

| System | Recall@10 | MRR | Precision@10 | nDCG@10 | MAP@10 | Hit@10 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | {b_k10.recall_at_k:.4f} | {b_k10.mrr:.4f} | {b_k10.precision_at_k:.4f} | {b_k10.ndcg_at_k:.4f} | {b_k10.map_at_k:.4f} | {b_k10.hit_at_k:.4f} |
| FAISS Dense | {d_k10.recall_at_k:.4f} | {d_k10.mrr:.4f} | {d_k10.precision_at_k:.4f} | {d_k10.ndcg_at_k:.4f} | {d_k10.map_at_k:.4f} | {d_k10.hit_at_k:.4f} |
| Hybrid RRF 1:1 | {h1_k10.recall_at_k:.4f} | {h1_k10.mrr:.4f} | {h1_k10.precision_at_k:.4f} | {h1_k10.ndcg_at_k:.4f} | {h1_k10.map_at_k:.4f} | {h1_k10.hit_at_k:.4f} |
| Hybrid RRF 1:2 | {h2_k10.recall_at_k:.4f} | {h2_k10.mrr:.4f} | {h2_k10.precision_at_k:.4f} | {h2_k10.ndcg_at_k:.4f} | {h2_k10.map_at_k:.4f} | {h2_k10.hit_at_k:.4f} |

---

## 4. What Happened?

### Lexical vs. Semantic Retrieval

This experiment evaluates how BM25 and FAISS Dense retrieval behave on scientific literature (SciFact).

**Observed result:**

> FAISS Dense achieved **{d_k5.recall_at_k*100:.1f}% Recall@5** compared with **{b_k5.recall_at_k*100:.1f}%** for BM25, a difference of **{delta_recall_k5:+.1f} percentage points**. In graded ranking, Dense achieved **{d_k5.ndcg_at_k:.4f} nDCG@5** versus **{b_k5.ndcg_at_k:.4f}** for BM25 ({delta_ndcg_k5:+.4f} difference).

### Hybrid Retrieval

RRF combines the rankings produced by BM25 and Dense retrieval.

We compare:

- **1:1 Balanced RRF**
- **1:2 Dense-biased RRF**

**Observed result:**

> Hybrid RRF 1:1 achieved **{h1_k5.recall_at_k*100:.1f}% Recall@5** ({h1_k5.ndcg_at_k:.4f} nDCG@5). Hybrid RRF 1:2 achieved **{h2_k5.recall_at_k*100:.1f}% Recall@5** ({h2_k5.ndcg_at_k:.4f} nDCG@5) and **{h2_k10.recall_at_k*100:.1f}% Recall@10** ({h2_k10.ndcg_at_k:.4f} nDCG@10).

---

## 5. Complementarity Analysis

The aggregate metrics tell us **what** happened. Query-level analysis helps explain **why**.

### Binary retrieval complementarity at K=5 (based on Hit@5):

- **Both Succeed**: {both_succeed} queries ({both_succeed/total_queries*100:.1f}%)
- **Dense Succeeds & BM25 Fails**: {dense_only_succeed} queries ({dense_only_succeed/total_queries*100:.1f}%)
- **BM25 Succeeds & Dense Fails**: {bm25_only_succeed} queries ({bm25_only_succeed/total_queries*100:.1f}%)
- **Both Fail**: {both_fail} queries ({both_fail/total_queries*100:.1f}%)

**Key observation:**

> Across the 300 test queries, Dense uniquely retrieved a relevant document in the top 5 for {dense_only_succeed} queries where BM25 missed, while BM25 uniquely retrieved a relevant document for {bm25_only_succeed} queries where Dense missed. Combining both candidate streams yields a theoretical union Hit@5 ceiling of **{union_ceiling:.1f}%** ({both_succeed + dense_only_succeed + bm25_only_succeed} / {total_queries} queries).

---

## 6. Graded Relevance

Unlike our original binary benchmark, SciFact provides graded relevance judgments.

This allows us to distinguish:

> finding a relevant document

from:

> ranking the **most relevant** documents highly.

Therefore:

- **Recall@K** measures retrieval coverage.
- **nDCG@K** measures ranking quality with graded relevance.
- **MAP@K** measures precision across relevant-document positions.
- **MRR** emphasizes the first relevant result.
- **Precision@K** measures relevant results within the cutoff.
- **Hit@K** measures whether at least one relevant result was retrieved.

**Observed result:**

> At K=5, Dense nDCG@5 is **{d_k5.ndcg_at_k:.4f}** and MAP@5 is **{d_k5.map_at_k:.4f}**, compared with BM25 nDCG@5 of **{b_k5.ndcg_at_k:.4f}** and MAP@5 of **{b_k5.map_at_k:.4f}**. The ranking metrics track the relative order observed in Recall@5 ({d_k5.recall_at_k*100:.1f}% vs {b_k5.recall_at_k*100:.1f}%).

---

## 7. Key Findings

### Finding 1 — BM25
> On SciFact, BM25 achieved {b_k5.recall_at_k*100:.1f}% Recall@5, {b_k5.ndcg_at_k:.4f} nDCG@5, and {b_k5.mrr:.4f} MRR.

### Finding 2 — Dense Retrieval
> FAISS Dense retrieval (BGE-small) achieved {d_k5.recall_at_k*100:.1f}% Recall@5, {d_k5.ndcg_at_k:.4f} nDCG@5, and {d_k5.mrr:.4f} MRR.

### Finding 3 — Hybrid Retrieval
> Hybrid RRF 1:1 scored {h1_k5.recall_at_k*100:.1f}% Recall@5 ({h1_k5.ndcg_at_k:.4f} nDCG@5), while Hybrid RRF 1:2 scored {h2_k5.recall_at_k*100:.1f}% Recall@5 ({h2_k5.ndcg_at_k:.4f} nDCG@5).

### Finding 4 — RRF Weighting
> Increasing the Dense retriever weight from 1.0 to 2.0 shifted Recall@5 from {h1_k5.recall_at_k*100:.1f}% to {h2_k5.recall_at_k*100:.1f}% and nDCG@5 from {h1_k5.ndcg_at_k:.4f} to {h2_k5.ndcg_at_k:.4f}.

---

## 8. What This Means for Two-Stage Retrieval

This experiment establishes the **Stage-1 retrieval baseline** for the upcoming two-stage system.

The architecture is:

```text
Query
  ↓
BM25 + FAISS Dense
  ↓
Hybrid Candidate Generation
  ↓
Candidate Pool
  ↓
Stage-2 Reranker
```
"""
    report_path.write_text(report_content, encoding="utf-8")
    print(f"\nSaved Experiment 021 Report to: {report_path}")


if __name__ == "__main__":
    run_experiment()
