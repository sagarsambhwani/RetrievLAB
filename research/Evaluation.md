# Evaluation Roadmap & Progress

> **Goal**: Build a rigorous, extensible evaluation framework for retrieval systems.

---

# Philosophy

Evaluation exists to answer one empirical question:

> **How good is a retrieval pipeline, and why does it succeed or fail?**

RetrievLab evaluates retrieval algorithms independently of embedding providers, vector databases, or user interfaces.

---

# Guiding Principles

- Start with the simplest possible evaluation.
- Build research infrastructure and simple algorithms yourself; use established libraries for mature algorithms.
- Evaluation should be independent of retrieval implementation details.
- Every experiment should be reproducible.
- Optimize for understanding before benchmark performance.

---

# Phase Progress & Status

| Phase | Description | Status | Deliverables |
| :--- | :--- | :---: | :--- |
| **Phase 1 — Foundation** | Benchmark data model, JSON loader, Recall@K baseline | ✅ Complete | `models.py`, `benchmark.py`, `simple2.json` |
| **Phase 2 — Core Metrics** | Recall@K, Precision@K, MRR, Evaluation Reports | ✅ Complete | `metrics.py`, `reports.py`, `evaluate.py` |
| **Phase 3 — Benchmark Support** | Handcrafted + Local JSON benchmarks | 🟡 Active | `data/benchmarks/simple2.json`, BEIR adapters planned |
| **Phase 4 — Experiment Framework**| One-command CLI/runner for reproducible experiments | 🟡 Active | `run_sprint1_experiments.py`, Sprint 2 runners |
| **Phase 5 — Comparative Studies**| Lexical vs. Dense vs. Hybrid evaluation | 🟡 Active | `results/sprint_1/`, `results/sprint_2/` |
| **Phase 6 — Feature Engineering** | Evaluating individual retrieval signals (BM25, Cosine, Heading) | ⚪ Planned | `retrievlab/features/` |
| **Phase 7 — Learning-to-Rank** | Ranking models (LambdaMART / LightGBM) | ⚪ Planned | `retrievlab/ranking/` |
| **Phase 8 — Adaptive LTR** | Runtime query intent classification & dynamic ranker | ⚪ Planned | `retrievlab/retrieval/adaptive.py` |

---

# Classical Metrics Supported

- [x] **Recall@K** ($K \in \{1, 3, 5, 10\}$)
- [x] **Precision@K** ($K \in \{1, 5, 10\}$)
- [x] **Mean Reciprocal Rank (MRR)**
- [ ] **Normalized Discounted Cumulative Gain (nDCG@K)**
- [ ] **Mean Average Precision (MAP)**
- [ ] **Latency & Throughput Profiling**