Layer 1 – Corpus

Support multiple datasets:

PDFs
HTML
Markdown
Code
Tables
Images (OCR)
Multi-modal documents
Layer 2 – Chunking Research

Implement and compare:

Fixed-size chunking
Recursive chunking
Semantic chunking
Hierarchical chunking
Parent-child retrieval
Late chunking

Evaluate:

Recall
Precision
Latency
Token cost
Layer 3 – Embedding Research

Compare models like:

OpenAI embeddings
Gemini embeddings
BGE
E5
Nomic
Jina

Evaluate:

Retrieval quality
Index size
Inference speed
Cost
Layer 4 – Vector Indexes

Compare:

FAISS
HNSW
IVF
DiskANN
ScaNN
Qdrant
Milvus
pgvector

Questions to answer:

Which index is fastest?
Which uses the least memory?
Which maintains the highest recall?
Layer 5 – Retrieval Strategies

Implement:

Dense retrieval
BM25
Hybrid search
Metadata filtering
Multi-query retrieval
HyDE
Query rewriting
Parent-child retrieval
GraphRAG
Ensemble retrieval
Layer 6 – Reranking

Compare:

Cross-encoders
Cohere Rerank
BGE Reranker
Jina Reranker
LLM reranking

Measure:

nDCG
MRR
Recall@k
Latency
Layer 7 – Evaluation

Go beyond "it feels better."

Use:

Precision@k
Recall@k
MAP
MRR
nDCG
Hit Rate
Context precision
Context recall
Faithfulness
Answer relevance

Include automated and human evaluation.

Layer 8 – Production

Add observability:

Latency breakdown
Embedding cache hit rate
Query distributions
Cost per query
Failure analysis
Layer 9 – Adaptive Retrieval

Experiment with:

Automatic top-k selection
Dynamic chunk size
Confidence-based retrieval
Multi-stage retrieval pipelines

---

# 🏛️ RetrievLab Architecture Analysis & Comparative Review

## 1. Architecture & Design Patterns Identified

The architecture built in Sprint 1 follows four foundational design patterns:

```text
 Documents ──► [DocumentLoader] ──► [MarkdownChunker] ──► [FastEmbed/Embedder]
                                                                │
                                                                ▼
[EvaluationReport] ◄── [evaluate_retriever] ◄── [BM25Retriever / DenseRetriever]
```

### 1. Strategy Pattern (`Retriever` Interface)
- **Implementation:** `BM25Retriever` and `DenseRetriever` share identical call signatures (`index()`, `retrieve()`).
- **Benefit:** Allows the evaluation runner (`evaluate_retriever`) to consume any retrieval engine polymorphically without knowing its internal indexing mechanics.

### 2. Pipeline / Filter Pattern (Dataflow Pipeline)
- **Implementation:** Unidirectional dataflow: `Document` -> `Chunker` -> `Embedder` -> `Retriever` -> `Evaluator`.
- **Benefit:** Components do not share mutable global state; each stage consumes clean domain objects (`Document`, `Chunk`) and outputs enriched data objects (`EmbeddedChunk`, `SearchResult`).

### 3. Adapter Pattern (`FastEmbedClient`)
- **Implementation:** Embeddings are encapsulated behind `FastEmbedClient` and `Embedder`.
- **Benefit:** Decouples vector generation from the retriever. Swapping embedding providers (e.g., to OpenAI, Voyage AI, or HuggingFace) requires zero code changes inside `DenseRetriever`.

### 4. Domain-Driven Core Models (Value Objects)
- **Implementation:** Strongly-typed dataclass/Pydantic-style objects (`Document`, `Chunk`, `BenchmarkCase`, `RetrieverEvaluationResult`).
- **Benefit:** Prevents dictionary string-key bugs (`chunk["text"]` vs `chunk.text`) and guarantees schema validity across experiments.

---

## 2. Architectural Evaluation & Opinions

### Strengths of Current Design
1. **Zero-Abstraction Overhead (First Principles):** Unlike heavy frameworks (LangChain / LlamaIndex) that wrap retrieval in multi-layered abstractions, RetrievLab is explicit and inspectable.
2. **Evaluation-First Philosophy:** Benchmarking (`Recall@K`, `Precision@K`, `MRR`) is integrated directly into the pipeline rather than tacked on at the end.
3. **High Scientific Reproducibility:** Fixed benchmark schemas (`simple2.json`) ensure every algorithm comparison is deterministic and apples-to-apples.

### Current Architectural Boundaries / Limitations
1. **Synchronous & In-Memory Execution:** All chunking, embedding, and retrieval happen synchronously in local process memory.
2. **Linear Cosine Search ($O(N \cdot D)$):** `DenseRetriever` currently calculates brute-force dot products across all vector arrays. While fine for thousands of chunks, it does not scale to millions without indexing (which FAISS solves in Sprint 2).
3. **Uncoupled Tokenization:** Tokenization logic in BM25 is currently internal rather than a standalone configurable pipeline module (addressed in `RLB-201`).

---

## 3. Comparison: RetrievLab vs Production Architecture

Production retrieval systems (e.g., enterprise search at Netflix, Pinecone, Elastic, or Notion) require defensive scalability and strict opinionatedness. Here is how RetrievLab compares:

| Architectural Dimension | RetrievLab (Current Architecture) | Production Enterprise RAG Architecture | Architectural Gap / Next Steps |
|---|---|---|---|
| **System Pattern** | **Monolithic / In-Process Pipeline:** Embedded Python library running in a single process thread. | **Microservices & Asynchronous Event Pipelines:** Distributed workers (Celery/Temporal) processing document queues. | RetrievLab can add an optional REST API / Async wrapper (FastAPI). |
| **Vector Storage** | **In-Memory NumPy Arrays:** Raw vector arrays held in RAM (`DenseRetriever`). | **Distributed Vector Databases:** Partitioned HNSW indexes (Qdrant, Pinecone, Milvus, FAISS) with shard replication. | Sprint 2 ticket `RLB-230` introduces FAISS to bridge this gap. |
| **Defensive Scalability** | **Minimal:** Assumes valid input files and bounded corpus sizes. | **High Defensive Mechanics:** Rate limiting, circuit breakers, Redis semantic caching, fallback retrievers, backpressure management. | Defensive error boundaries, batching thresholds, and memory caps. |
| **Opinionatedness** | **Low Opinionatedness:** Flexible research laboratory designed to test and compare any approach. | **High Opinionatedness:** Enforces strict chunk size limits, mandatory telemetry, system guardrails (LlamaGuard), and fixed SLA response times. | Keep RetrievLab unopinionated for research, but add strict validation schemas. |
| **Evaluation Mechanics** | **Offline Benchmarks:** Batch metrics over static ground-truth datasets. | **Hybrid Offline + Online Telemetry:** A/B testing, user CTR tracking, online relevance feedback, LLM-as-a-Judge telemetry. | Log online query telemetry alongside offline benchmarks. |

---

## 4. Summary & Strategic Roadmap

- **Sprint 1 (Completed):** Established clean, un-abstracted baselines (`BM25Retriever`, `DenseRetriever`) and a reproducible evaluation engine.
- **Sprint 2 (Planned):** Evolves the architecture from single-retriever baselines to **Hybrid Candidate Generation** (`HybridRRFRetriever`), configurable preprocessing pipelines, and **FAISS index scaling**.
- **Sprint 3 (Future):** Introduces **Feature Extraction** (calculating dense similarity, BM25 score, heading overlap) and **Learning-to-Rank (LTR)** models to adaptively select optimal rankings per query.