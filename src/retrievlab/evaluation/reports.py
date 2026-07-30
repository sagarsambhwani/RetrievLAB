from pydantic import BaseModel, Field


class RetrieverEvaluationResult(BaseModel):
    """Evaluation result metrics for a single retriever."""

    retriever_name: str
    recall_at_k: float
    precision_at_k: float
    mrr: float
    k: int = 5
    num_cases: int = 0


class EvaluationReport(BaseModel):
    """Aggregate report collecting evaluation results across multiple retrievers."""

    results: list[RetrieverEvaluationResult] = Field(default_factory=list)

    def add_result(self, result: RetrieverEvaluationResult) -> None:
        """Add a retriever evaluation result to the report.

        Args:
            result: A RetrieverEvaluationResult instance.
        """
        self.results.append(result)

    def to_markdown(self) -> str:
        """Render the evaluation results as a formatted Markdown table.

        Returns:
            A Markdown table string displaying Retriever, Recall@K, MRR, and Precision@K.
        """
        if not self.results:
            return "No evaluation results available."

        k_val = self.results[0].k if self.results else 5

        headers = ["Retriever", f"Recall@{k_val}", "MRR", f"Precision@{k_val}"]
        lines = [
            f"| {' | '.join(headers)} |",
            f"| {' | '.join([':---'] + [':---:'] * (len(headers) - 1))} |",
        ]

        for res in self.results:
            row = [
                res.retriever_name,
                f"{res.recall_at_k:.4f}",
                f"{res.mrr:.4f}",
                f"{res.precision_at_k:.4f}",
            ]
            lines.append(f"| {' | '.join(row)} |")

        return "\n".join(lines)

    def to_dict(self) -> list[dict]:
        """Return raw dictionaries for all retriever evaluation results.

        Returns:
            A list of dictionary representations of the results.
        """
        return [res.model_dump() for res in self.results]
