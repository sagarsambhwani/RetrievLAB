# 🧠 Adaptive Reranker Architecture & Research Strategy (v1)

**Version:** 1.0  
**Date:** 2026-08-28  
**Status:** 📋 Design Blueprint  
**Authors / Contributors:** RetrievLab Research Team  

---

## 1. Executive Problem Statement

### The "Static Fusion" Ceiling
Static rank fusion (e.g., standard Reciprocal Rank Fusion with $k=60$ and fixed $1:1$ or $1:2$ weights) applies identical blending logic to all queries:
- **Exact Code / Identifiers** (`uvicorn.run(app)`, `recall_at_k`): Lexical matches should dominate ($>85\%$ weight), but static weights dilute high-precision keyword signals.
- **Conceptual Paraphrases** (*"how do I deploy a web server on the cloud"*): Semantic embeddings should dominate ($>85\%$ weight), but static weights can allow noisy boilerplate lexical matches to compete.
- **Ambiguous Queries**: Require deep cross-attention reranking rather than shallow rank blending.

**Objective of Adaptive Reranking:** Dynamically determine ranking policy, retriever weights, or reranking models at runtime based on query characteristics and retrieval confidence margins.

---

## 2. Multi-Stage Architectural Blueprint

```mermaid
graph TD
    UserQuery["🔍 Query (q)"] --> LexicalGen["Lexical Generator (BM25Retriever)"]
    UserQuery --> DenseGen["Dense Generator (FastEmbed/FAISS)"]
    
    LexicalGen -->|Top 20-50 Candidates| Pool["Union Candidate Pool (C_bm25 ∪ C_dense)"]
    DenseGen -->|Top 20-50 Candidates| Pool
    
    Pool --> SignalExtractor["1. Query & Score Signal Extractor"]
    SignalExtractor --> PolicyEngine{"2. Adaptive Policy Engine"}
    
    PolicyEngine -->|High BM25 Margin (Δ_bm25 > τ)| LexicalBoost["Boost BM25 Weight (w=[3.0, 1.0])"]
    PolicyEngine -->|High Dense Margin & Flat BM25| DenseBoost["Boost Dense Weight (w=[1.0, 3.0])"]
    PolicyEngine -->|Ambiguous / Conflicting Scores| CrossEncoder["Trigger Zero-Shot Cross-Encoder"]
    
    LexicalBoost --> FinalTopK["🎯 Final Top-K Search Results"]
    DenseBoost --> FinalTopK
    CrossEncoder --> FinalTopK
```

---

## 3. Candidate Pool Specifications

To ensure high recall without compute bottlenecks:

| Dimension | Specification | Rationale |
| :--- | :--- | :--- |
| **Candidate Sources** | Lexical ($C_{\text{bm25}}$) + Dense ($C_{\text{dense}}$) | Captures orthogonal keyword and semantic matches. |
| **Candidate Depth ($K_{\text{cand}}$)** | **$20 \text{ to } 50$ items** per retriever | Prevents true positive candidates ranked in positions #6–#15 from being truncated before reranking. |
| **Union Pool Size ($|C|$)** | **$\approx 20 \text{ to } 35$ unique chunks** | Compact enough for sub-millisecond heuristic scoring or $<30\text{ ms}$ cross-encoder passes. |
| **Candidate Payload** | `chunk.id`, `chunk.text`, raw BM25 score, raw dense cosine score, 1-based ranks ($r_m$) | Provides complete feature representation for downstream rankers. |

---

## 4. Feasibility & Model Training Analysis

### Why Supervised Training from Scratch is Infeasible Today
With our current benchmark size (**22 queries, 9 chunks**):
- Any supervised ML model (LambdaMART, LightGBM GBDT, or fine-tuned Transformer) will **severely overfit**, memorizing exact query strings rather than generalizable ranking distributions.
- A standard $70/15/15$ train/val/test split leaves only **15 training queries**, making statistical convergence impossible.

### Minimum Scale Requirements:
| Architecture | Training Data Needed | What We Have | Verdict |
| :--- | :---: | :---: | :---: |
| **Learned GBDT (LightGBM / LambdaMART)** | **500 – 2,000+ queries** | 22 queries | ❌ Infeasible for v1 (deferred to Phase 3) |
| **Fine-Tuned Cross-Encoder** | **10,000 – 100,000+ pairs** | ~440 pairs | ❌ Infeasible for v1 |
| **Zero-Shot Pretrained Cross-Encoder** | **0 queries** *(Inference only)* | — | ✅ **Feasible for v1.2** |
| **Score-Margin Dynamic Gating (Heuristic)** | **0 queries** *(Rule-based)* | — | ✅ **Feasible for v1.0** |

---

## 5. Phased Implementation Roadmap

### Phase 1: Score-Margin Dynamic Gating (`AdaptiveReranker` v1.0)
- **Mechanism:** Zero-training query-time heuristic in [`src/retrievlab/ranking/adaptive.py`](file:///e:/Downloads/RetrievLab/src/retrievlab/ranking/adaptive.py).
- **Core Signals:**
  - **BM25 Score Margin:** $\Delta_{\text{bm25}} = S_{\text{bm25}}^{(1)} - S_{\text{bm25}}^{(2)}$
  - **Dense Score Margin:** $\Delta_{\text{dense}} = S_{\text{dense}}^{(1)} - S_{\text{dense}}^{(2)}$
  - **Keyword Specificity:** Ratio of query tokens present in inverted index with high IDF.
- **Decision Logic:**
  $$\text{If } \Delta_{\text{bm25}} \ge \tau_{\text{lex}} \implies w = [2.5, 1.0] \quad (\text{Lexical Dominance})$$
  $$\text{If } \Delta_{\text{bm25}} < \tau_{\text{flat}} \text{ and } \Delta_{\text{dense}} \ge \tau_{\text{sem}} \implies w = [1.0, 2.5] \quad (\text{Semantic Dominance})$$
  $$\text{Otherwise} \implies w = [1.0, 1.0] \quad (\text{Balanced RRF})$$
- **Latency Overhead:** $< 0.1 \text{ ms}$.

### Phase 2: Zero-Shot Pretrained Cross-Encoder (`CrossEncoderReranker` v1.2)
- **Mechanism:** Integrate lightweight pretrained cross-encoder (e.g., `BAAI/bge-reranker-small` or `cross-encoder/ms-marco-MiniLM-L-6-v2`) via FastEmbed or ONNX.
- **Cascade Trigger:** Only invoke cross-encoder when Phase 1 detects high ambiguity ($\Delta_{\text{bm25}} < \tau$ and $\Delta_{\text{dense}} < \tau$).
- **Latency Overhead:** $15 \text{ to } 35 \text{ ms}$ on CPU for top 20 candidates.

### Phase 3: Learned GBDT / LambdaMART (Future Scale)
- **Prerequisite:** Automated Golden Dataset Generation ($500+$ queries).
- **Model:** LightGBM ranker trained on 8 features (BM25 score, dense cosine, heading level, term coverage ratio, chunk length, position penalty).

---

## 6. Evaluation Metrics & Success Criteria

1. **Recall@5:** Must match or exceed Dense baseline ($1.0000$).
2. **MRR:** Must exceed static balanced RRF ($> 0.9545$).
3. **Zero Degradations:** Zero queries where adaptive ranking performs worse than both individual baselines.
4. **Latency Budget:** Phase 1 execution time $< 1 \text{ ms}$ total query overhead.
