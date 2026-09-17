# Project Guidelines for RetrievLab

## Core Principle: Build vs Use (Infrastructure & Algorithms)

- **Implement research infrastructure and simple algorithms yourself**: Build evaluation harnesses, benchmark loaders, chunkers, and baseline algorithms from scratch to ensure complete transparency, modularity, and first-principles understanding.
- **Use established libraries for mature algorithms and models**: Use battle-tested, optimized libraries (e.g., FAISS, FastEmbed, LightGBM, Hugging Face) for mature models and production-grade indexing/ranking algorithms.
- **Understand library assumptions**: When utilizing external libraries, understand what the library is doing under the hood, what assumptions it makes (e.g., vector normalization, metric spaces, tokenization defaults), and how those assumptions impact experimental results.

## Design Principle: Algorithms vs Configurations

RetrievLab compares **retrieval strategies**, not duplicate implementations.

- **Rule 1 — One algorithm, one implementation**: Each retrieval algorithm should have a single implementation (e.g. `BM25Retriever`, `DenseRetriever`, `HybridRetriever`, `FAISSRetriever`). Do not create multiple files/classes for variations like `bm25_stemmed.py`.
- **Rule 2 — Behavior should be configurable**: Preprocessing or parameter tuning should be injected as configuration (e.g., passing `tokenizer` to `BM25Retriever`), leaving the core retrieval algorithm untouched.
- **Rule 3 — Experiments compare configurations**: Compare different configurations of the same algorithm (e.g. Basic vs +Stopwords vs +Stemming vs +Tuned Parameters).
- **Rule 4 — Introduce a new retriever only for fundamentally different approaches**: Create new retrievers only when the paradigm itself changes (e.g. BM25 vs Dense vs Hybrid/RRF vs FAISS vs SPLADE vs ColBERT).
- **Rule 5 — Preserve experimental reproducibility**: Never overwrite existing configurations. Always store configuration, benchmark, evaluation metrics, and experiment reports.

> **Guiding Principle**: Implement algorithms once. Explore behavior through configuration. Validate improvements through experiments.

## User Interaction & Implementation Rule

- **Mandatory Approval Before Writing Code**: Never write or modify non-test code (features, data loaders, algorithms, experiments, models) without showing the proposed implementation first and obtaining explicit user approval.
- **Exception for Test Cases**: Writing, updating, or running unit test cases (`tests/test_*.py`) to verify functionality is exempt from requiring prior approval.
- **Propose & Show First**: Present a clear breakdown of the proposed sub-steps, classes, and file paths before asking for approval.
- **Granular Execution**: Break complex tasks into small, bite-sized sub-steps so the user can easily track progress.

## Git Workflow & Safety Guidelines

- **Guiding Light**: Preserve First $\rightarrow$ Inspect Second $\rightarrow$ Propose Third $\rightarrow$ Ask Fourth $\rightarrow$ Change Last.
- **Granular Commits**: Commit individual files/units one-by-one using RLB semantic prefixes (`feat`, `exp`, `eval`, `docs`, `fix`, `test`, `perf`, `chore`).
- **Push at the End**: Defer `git push` until all individual local commits are completed and verified.
- **PowerShell Syntax**: Always use `;` or separate commands on Windows PowerShell; avoid `&&`.

## Research Experiment Reporting Guidelines

- **Results-First Invariant**: Never write or draft research reports before the experiment script has completed execution and generated actual verified benchmark numbers.
- **8-Part Research Schema**: All reports in `results/` must adhere strictly to the 8-part empirical research structure (`.agents/rules/experiment-reports.md`).
- **Data-Driven Prose**: Never hard-code subjective outcomes or biased conclusions in report generators; always state exact measured deltas.
- **Explicit Metric Labeling**: Always specify the exact metric used for diagnostics and complementarity analysis.

## System Hardware & Resource Protection Rule

- **Hardware Profile**: 8 GB System RAM (tight), NVIDIA GeForce GTX 1650 (4 GB VRAM).
- **Mandatory Safety Rule**: Never overwhelm system RAM. Follow `.agents/rules/system-resources.md`.
- **Offload to GPU**: Route neural models to dedicated GPU VRAM to protect system RAM.
- **Batch & Bound**: Micro-batch tensors, enforce sequence length limits (`max_length <= 256`), and cache heavy neural outputs to disk.

## Code Quality & Linter Invariants

- **Zero Lint Policy**: Every file must pass `uv run ruff check .` with zero errors and zero warnings before completing tasks or committing. Follow `.agents/rules/code-quality.md`.
- **F-String Invariant (F541)**: Never prefix a string with `f` unless it contains `{}` variable or expression placeholders. Use standard strings for static titles, prints, and headers.
- **No Dead Code (F401, F841)**: Never leave unused imports or dead local variable assignments.



