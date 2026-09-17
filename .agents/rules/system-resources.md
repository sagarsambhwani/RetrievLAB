# System Hardware Constraints & Resource Safety Guidelines

## 1. Hardware Profile
- **System RAM:** 8 GB Total (Tight budget; susceptible to severe OS disk paging/swapping if RAM exceeds ~6 GB).
- **Dedicated GPU:** NVIDIA GeForce GTX 1650 (4 GB GDDR5/GDDR6 VRAM).
- **OS:** Windows.

---

## 2. Mandatory Rules to Protect System Stability

### Rule 1 — Never Overwhelm System RAM
- Never execute memory-hungry algorithms, unchunked tensor passes, or unbatched neural loops on CPU.
- When loading or generating embeddings/matrices, always use **micro-batching** (`batch_size <= 64`) with periodic garbage collection.
- Free intermediate arrays and references immediately after use.

### Rule 2 — Prioritize Dedicated GPU VRAM for Neural Models
- When executing neural models (Cross-Encoders, Sentence-Transformers, Embeddings), route execution to the **GTX 1650 GPU** where possible.
- Keep GPU batch sizes conservative (`batch_size=32` to `64`) to comfortably fit within the 4 GB VRAM budget without out-of-memory (OOM) crashes.
- Offloading to GPU VRAM directly protects the limited 8 GB system RAM.

### Rule 3 — Bound Transformer Sequence Lengths
- Never allow transformer inputs to default to unconstrained `512` tokens unless strictly required.
- Enforce domain-appropriate sequence length bounds (e.g., `max_length=192` or `256`) to avoid quadratic $O(L^2)$ memory and computation spikes.

### Rule 4 — Cache Heavy Intermediate Results
- Always persist precomputed embeddings, cross-encoder scores, and candidate pools to disk (`.npy`, `.parquet`).
- Reload from disk rather than recomputing heavy neural workloads from scratch.

### Rule 5 — Lightweight Real-Time Paths (Tree Models Over Heavy Neural Ensembles)
- Favor fast, lightweight models (e.g., LightGBM LambdaMART $< 0.2\text{ ms}$, single-pass vector indexing) for repeated sweeps and interactive workflows.
