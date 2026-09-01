"""Diagnostic and failure analysis tooling for hybrid retrieval systems.

This module provides automated query-by-query failure detection, rank disagreement
metrics, and outcome categorization across BM25 lexical, Dense semantic, and
Hybrid (RRF/Fused) retrieval strategies.
"""

from enum import Enum
from pydantic import BaseModel, Field

from retrievlab.models import Chunk, SearchResult
from retrievlab.retrieval.interface import Retriever
from retrievlab.evaluation.benchmark import Benchmark, BenchmarkCase
from retrievlab.evaluation.metrics import recall_at_k, reciprocal_rank


class QueryOutcomeCategory(str, Enum):
    """Categorical classification of a query's retrieval behavior across channels.

    Attributes:
        JOINT_HIT: Both BM25 and Dense succeeded in retrieving relevant chunk(s).
        DENSE_WIN_HYBRID_RECOVERED: BM25 failed (Recall=0), Dense succeeded, and Hybrid recovered.
        BM25_WIN_HYBRID_RECOVERED: Dense failed (Recall=0), BM25 succeeded, and Hybrid recovered.
        DENSE_WIN_HYBRID_MISSED: Dense succeeded, but Hybrid failed to pull it into Top-K.
        BM25_WIN_HYBRID_MISSED: BM25 succeeded, but Hybrid failed to pull it into Top-K.
        HYBRID_DEGRADATION: At least one baseline succeeded, but Hybrid missed entirely.
        JOINT_MISS: Neither BM25, Dense, nor Hybrid found any relevant chunk.
    """

    JOINT_HIT = "joint_hit"
    DENSE_WIN_HYBRID_RECOVERED = "dense_win_hybrid_recovered"
    BM25_WIN_HYBRID_RECOVERED = "bm25_win_hybrid_recovered"
    DENSE_WIN_HYBRID_MISSED = "dense_win_hybrid_missed"
    BM25_WIN_HYBRID_MISSED = "bm25_win_hybrid_missed"
    HYBRID_DEGRADATION = "hybrid_degradation"
    JOINT_MISS = "joint_miss"


class QueryDiagnostic(BaseModel):
    """Detailed query-level diagnosis comparing BM25, Dense, and Hybrid retrievers."""

    query_index: int = Field(default=1, description="1-based numerical index of the benchmark query.")
    query: str = Field(description="Raw user query string.")
    
    # Lexical BM25 performance metrics
    bm25_recall: float = Field(description="Recall@K for BM25.")
    bm25_mrr: float = Field(description="MRR for BM25.")
    bm25_rank: int | None = Field(default=None, description="1-based rank of first hit in BM25, or None.")
    
    # Semantic Dense performance metrics
    dense_recall: float = Field(description="Recall@K for Dense vector retriever.")
    dense_mrr: float = Field(description="MRR for Dense vector retriever.")
    dense_rank: int | None = Field(default=None, description="1-based rank of first hit in Dense, or None.")
    
    # Hybrid performance metrics
    hybrid_recall: float = Field(description="Recall@K for Hybrid retriever.")
    hybrid_mrr: float = Field(description="MRR for Hybrid retriever.")
    hybrid_rank: int | None = Field(default=None, description="1-based rank of first hit in Hybrid, or None.")
    
    # Classification and disagreement indicators
    category: QueryOutcomeCategory = Field(description="Categorical diagnosis of query outcome.")
    rank_shift: int = Field(default=0, description="Divergence in rank positions between Dense and BM25.")
    is_recovered: bool = Field(default=False, description="True if Hybrid recovered an isolated single-channel miss.")
    is_degradation: bool = Field(default=False, description="True if Hybrid dropped a query that succeeded on a baseline.")


def _find_first_hit_rank(results: list[SearchResult], expected_ids: set[str], k: int) -> int | None:
    """Find the 1-based rank of the first relevant chunk in the result list.

    Args:
        results: Ranked search results.
        expected_ids: Set of relevant chunk IDs for the query.
        k: Evaluation depth cutoff.

    Returns:
        1-based integer rank of first relevant match within top-K, or None.
    """
    for rank_idx, res in enumerate(results[:k], start=1):
        if res.chunk.id in expected_ids:
            return rank_idx
    return None


def categorize_query_outcome(
    bm25_recall: float,
    dense_recall: float,
    hybrid_recall: float,
) -> QueryOutcomeCategory:
    """Classify retrieval behavior based on binary recall across the three channels.

    Args:
        bm25_recall: Recall@K for BM25 (0.0 to 1.0).
        dense_recall: Recall@K for Dense (0.0 to 1.0).
        hybrid_recall: Recall@K for Hybrid (0.0 to 1.0).

    Returns:
        QueryOutcomeCategory enum value.
    """
    bm25_hit = bm25_recall > 0.0
    dense_hit = dense_recall > 0.0
    hybrid_hit = hybrid_recall > 0.0

    # 1. Both baselines succeeded
    if bm25_hit and dense_hit:
        if hybrid_hit:
            return QueryOutcomeCategory.JOINT_HIT
        return QueryOutcomeCategory.HYBRID_DEGRADATION

    # 2. Dense succeeded, BM25 missed
    if dense_hit and not bm25_hit:
        if hybrid_hit:
            return QueryOutcomeCategory.DENSE_WIN_HYBRID_RECOVERED
        return QueryOutcomeCategory.DENSE_WIN_HYBRID_MISSED

    # 3. BM25 succeeded, Dense missed
    if bm25_hit and not dense_hit:
        if hybrid_hit:
            return QueryOutcomeCategory.BM25_WIN_HYBRID_RECOVERED
        return QueryOutcomeCategory.BM25_WIN_HYBRID_MISSED

    # 4. Both baselines missed
    if hybrid_hit:
        # Rare case: neither had it in individual Top-K, but fusion pooled it in
        return QueryOutcomeCategory.JOINT_HIT
    return QueryOutcomeCategory.JOINT_MISS


def analyze_query_outcome(
    case: BenchmarkCase,
    bm25_results: list[SearchResult],
    dense_results: list[SearchResult],
    hybrid_results: list[SearchResult],
    k: int = 5,
    query_index: int = 1,
) -> QueryDiagnostic:
    """Diagnose a single benchmark case across BM25, Dense, and Hybrid retrieval results.

    Args:
        case: Benchmark test case containing query and relevant chunk IDs.
        bm25_results: Ranked list of SearchResult from BM25.
        dense_results: Ranked list of SearchResult from Dense retriever.
        hybrid_results: Ranked list of SearchResult from Hybrid retriever.
        k: Evaluation cutoff depth. Defaults to 5.
        query_index: 1-based sequential index of the benchmark query. Defaults to 1.

    Returns:
        QueryDiagnostic model with complete comparative metrics and category.
    """
    expected_ids = set(case.relevant_chunk_ids)

    # Compute metrics for BM25
    b_recall = recall_at_k(bm25_results, case, k=k)
    b_mrr = reciprocal_rank(bm25_results, case)
    b_rank = _find_first_hit_rank(bm25_results, expected_ids, k=k)

    # Compute metrics for Dense
    d_recall = recall_at_k(dense_results, case, k=k)
    d_mrr = reciprocal_rank(dense_results, case)
    d_rank = _find_first_hit_rank(dense_results, expected_ids, k=k)

    # Compute metrics for Hybrid
    h_recall = recall_at_k(hybrid_results, case, k=k)
    h_mrr = reciprocal_rank(hybrid_results, case)
    h_rank = _find_first_hit_rank(hybrid_results, expected_ids, k=k)

    # Determine category
    category = categorize_query_outcome(b_recall, d_recall, h_recall)

    # Calculate rank shift (|Dense Rank - BM25 Rank|)
    rank_shift = 0
    if b_rank is not None and d_rank is not None:
        rank_shift = abs(d_rank - b_rank)
    elif b_rank is not None:
        rank_shift = (k + 1) - b_rank
    elif d_rank is not None:
        rank_shift = (k + 1) - d_rank

    is_recovered = category in (
        QueryOutcomeCategory.DENSE_WIN_HYBRID_RECOVERED,
        QueryOutcomeCategory.BM25_WIN_HYBRID_RECOVERED,
    )
    is_degradation = category in (
        QueryOutcomeCategory.HYBRID_DEGRADATION,
        QueryOutcomeCategory.DENSE_WIN_HYBRID_MISSED,
        QueryOutcomeCategory.BM25_WIN_HYBRID_MISSED,
    )

    return QueryDiagnostic(
        query_index=query_index,
        query=case.query,
        bm25_recall=b_recall,
        bm25_mrr=b_mrr,
        bm25_rank=b_rank,
        dense_recall=d_recall,
        dense_mrr=d_mrr,
        dense_rank=d_rank,
        hybrid_recall=h_recall,
        hybrid_mrr=h_mrr,
        hybrid_rank=h_rank,
        category=category,
        rank_shift=rank_shift,
        is_recovered=is_recovered,
        is_degradation=is_degradation,
    )


class HybridFailureAnalysisReport(BaseModel):
    """Comprehensive failure and recovery analysis report across an entire benchmark."""

    total_queries: int
    category_counts: dict[str, int] = Field(default_factory=dict)
    diagnostics: list[QueryDiagnostic] = Field(default_factory=list)

    @property
    def recoveries(self) -> list[QueryDiagnostic]:
        """Return all queries where Hybrid recovered a single-channel failure."""
        return [d for d in self.diagnostics if d.is_recovered]

    @property
    def degradations(self) -> list[QueryDiagnostic]:
        """Return all queries where Hybrid degraded relative to baselines."""
        return [d for d in self.diagnostics if d.is_degradation]

    @property
    def disagreements(self) -> list[QueryDiagnostic]:
        """Return queries with significant rank disagreements between BM25 and Dense."""
        return [d for d in self.diagnostics if d.rank_shift > 0]

    def to_markdown(self) -> str:
        """Render the failure analysis as a formatted Markdown summary and case table.

        Returns:
            Formatted Markdown report string.
        """
        lines = [
            "# Hybrid Failure Analysis & Recovery Diagnosis",
            "",
            "## 1. Outcome Distribution Summary",
            "",
            f"- **Total Benchmark Queries:** {self.total_queries}",
            f"- **Joint Hits (Both BM25 & Dense Succeeded):** {self.category_counts.get(QueryOutcomeCategory.JOINT_HIT.value, 0)}",
            f"- **Dense Wins Recovered by Hybrid:** {self.category_counts.get(QueryOutcomeCategory.DENSE_WIN_HYBRID_RECOVERED.value, 0)}",
            f"- **BM25 Wins Recovered by Hybrid:** {self.category_counts.get(QueryOutcomeCategory.BM25_WIN_HYBRID_RECOVERED.value, 0)}",
            f"- **Hybrid Degradations (Recall Loss):** {self.category_counts.get(QueryOutcomeCategory.HYBRID_DEGRADATION.value, 0)}",
            f"- **Joint Misses (All Failed):** {self.category_counts.get(QueryOutcomeCategory.JOINT_MISS.value, 0)}",
            "",
            "## 2. Query-by-Query Diagnostic Breakdown",
            "",
            "| # | Query | BM25 Rank | Dense Rank | Hybrid Rank | Category | Status |",
            "| :---: | :--- | :---: | :---: | :---: | :--- | :---: |",
        ]

        for d in self.diagnostics:
            b_rank_str = f"#{d.bm25_rank}" if d.bm25_rank else "-"
            d_rank_str = f"#{d.dense_rank}" if d.dense_rank else "-"
            h_rank_str = f"#{d.hybrid_rank}" if d.hybrid_rank else "-"

            status = "[OK]"
            if d.is_recovered:
                status = "[RECOVERED]"
            elif d.is_degradation:
                status = "[DEGRADED]"
            elif d.category == QueryOutcomeCategory.JOINT_MISS:
                status = "[MISS]"

            query_snippet = d.query[:50] + "..." if len(d.query) > 50 else d.query
            row = [
                str(d.query_index),
                query_snippet.replace("|", "\\|"),
                b_rank_str,
                d_rank_str,
                h_rank_str,
                d.category.value,
                status,
            ]
            lines.append(f"| {' | '.join(row)} |")

        return "\n".join(lines)


def analyze_hybrid_failures(
    bm25_retriever: Retriever,
    dense_retriever: Retriever,
    hybrid_retriever: Retriever,
    benchmark: Benchmark,
    chunks: list[Chunk],
    k: int = 5,
) -> HybridFailureAnalysisReport:
    """Execute end-to-end failure diagnosis across BM25, Dense, and Hybrid retrievers.

    Args:
        bm25_retriever: BM25 lexical retriever instance.
        dense_retriever: Dense semantic vector retriever instance.
        hybrid_retriever: Hybrid (BM25 + Dense) fusion retriever instance.
        benchmark: Loaded benchmark test cases.
        chunks: Corpus chunk list.
        k: Top-K evaluation cutoff. Defaults to 5.

    Returns:
        HybridFailureAnalysisReport with complete diagnostic classifications.
    """
    diagnostics: list[QueryDiagnostic] = []
    category_counts: dict[str, int] = {cat.value: 0 for cat in QueryOutcomeCategory}

    for idx, case in enumerate(benchmark.cases, start=1):
        # Retrieve from all three channels
        b_res = bm25_retriever.retrieve(query=case.query, top_k=k, chunks=chunks)
        d_res = dense_retriever.retrieve(query=case.query, top_k=k, chunks=chunks)
        h_res = hybrid_retriever.retrieve(query=case.query, top_k=k, chunks=chunks)

        diag = analyze_query_outcome(
            case=case,
            bm25_results=b_res,
            dense_results=d_res,
            hybrid_results=h_res,
            k=k,
            query_index=idx,
        )

        diagnostics.append(diag)
        category_counts[diag.category.value] += 1

    return HybridFailureAnalysisReport(
        total_queries=len(benchmark.cases),
        category_counts=category_counts,
        diagnostics=diagnostics,
    )
