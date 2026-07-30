from retrievlab.evaluation.reports import RetrieverEvaluationResult, EvaluationReport


def test_retriever_evaluation_result_model():
    result = RetrieverEvaluationResult(
        retriever_name="DenseRetriever",
        recall_at_k=0.82,
        precision_at_k=0.76,
        mrr=0.71,
        k=5,
        num_cases=10,
    )
    assert result.retriever_name == "DenseRetriever"
    assert result.recall_at_k == 0.82
    assert result.precision_at_k == 0.76
    assert result.mrr == 0.71
    assert result.k == 5
    assert result.num_cases == 10


def test_evaluation_report_empty():
    report = EvaluationReport()
    assert report.to_markdown() == "No evaluation results available."
    assert report.to_dict() == []


def test_evaluation_report_markdown_and_dict():
    report = EvaluationReport()
    res1 = RetrieverEvaluationResult(
        retriever_name="Dense",
        recall_at_k=0.82,
        precision_at_k=0.76,
        mrr=0.71,
        k=5,
        num_cases=14,
    )
    res2 = RetrieverEvaluationResult(
        retriever_name="BM25",
        recall_at_k=0.79,
        precision_at_k=0.72,
        mrr=0.74,
        k=5,
        num_cases=14,
    )

    report.add_result(res1)
    report.add_result(res2)

    assert len(report.results) == 2
    raw_dicts = report.to_dict()
    assert len(raw_dicts) == 2
    assert raw_dicts[0]["retriever_name"] == "Dense"
    assert raw_dicts[1]["retriever_name"] == "BM25"

    md = report.to_markdown()
    lines = md.splitlines()
    assert "| Retriever | Recall@5 | MRR | Precision@5 |" in lines[0]
    assert "| Dense | 0.8200 | 0.7100 | 0.7600 |" in lines[2]
    assert "| BM25 | 0.7900 | 0.7400 | 0.7200 |" in lines[3]
