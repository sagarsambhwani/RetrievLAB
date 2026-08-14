---
trigger: always_on
---

# Git Workflow & Commit Guidelines for RetrievLab

## 1. Guiding Light: Refactoring & Safety Invariants
When modifying the repository, always follow:
1. **PRESERVE FIRST.** (Protect working baselines and proven patterns).
2. **INSPECT SECOND.** (Verify `git status` and `git diff` before making changes).
3. **PROPOSE THIRD.** (Formulate a clear plan / ADR).
4. **ASK FOURTH.** (Seek user feedback and approval).
5. **CHANGE LAST.** (Execute changes only after confirmation).

- **Main Protection**: Nothing gets merged, rebased, or deleted on `main` without explicit approval.
- **Baseline Checkpoints**: Create permanent tags (e.g., `immersa-v1-baseline`, `sprint-1-baseline`) before touching core architecture.

---

## 2. Granular Commits Policy (One File at a Time)
- **Stage & Commit Individually**: When committing multiple changes, stage (`git add <file>`) and commit (`git commit -m "..."`) each file or logical unit one-by-one.
- **Defer Push to the End ("Push in the Last")**: Never chain `git push` after each commit. Execute a single `git push` only after all granular commits have been successfully made and verified locally.
- **PowerShell Separators**: On Windows PowerShell, never use bash `&&` syntax. Use `;` or execute separate commands.

---

## 3. RetrievLab (RLB) Semantic Commit Specification

Format: `<prefix>(<scope>): <short description>`

### Prefixes
- `feat:` — New retrieval algorithms, chunkers, tokenizers, vector indices, or ranking models
- `exp:` — Experiment runners, parameter sweeps, and benchmark execution scripts
- `eval:` — Evaluation metrics, benchmark dataset schemas, and report generators
- `docs:` — Architecture Decision Records (ADRs), research hypotheses, failures taxonomy, and documentation
- `fix:` — Bug fixes in scoring formulas, boundary edge-cases, or tensor dimensions
- `test:` — Unit tests, metric assertions, and baseline regression tests
- `perf:` — Vector indexing throughput, parallel tokenization, and latency optimizations
- `chore:` — Dependencies (`uv`), environment configs, linter rules, or directory cleanups

### Standard Scopes
- `ingestion` — Document loaders (Markdown, PDF, HTML)
- `chunking` — Markdown, recursive, semantic chunking
- `preprocessing` — Tokenizers, stopwords, stemming, n-grams
- `embeddings` — FastEmbed, SentenceTransformers, embedding clients
- `indexing` — Brute-force linear search, FAISS, Qdrant
- `retrieval` — BM25, Dense, Hybrid (RRF), Adaptive
- `ranking` — Feature engineering, Cross-Encoders, LightGBM / LTR
- `evaluation` — Metrics (Recall, MRR, nDCG), benchmarks, reports
- `adr` — Architecture Decision Records (`docs/adr/`)
- `hypotheses` / `failures` / `observations` — Research logs (`research/`)
- `principles` — Design principles and agent guidelines
- `repo` / `deps` — Repository structure, dependencies, toolchains
