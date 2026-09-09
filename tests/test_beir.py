"""
Unit tests for BEIRLoader (corpus, query, and qrels parsing).
"""

from pathlib import Path
import pytest

from retrievlab.ingestion.beir import BEIRLoader
from retrievlab.models import Chunk
from retrievlab.evaluation.benchmark import Benchmark


@pytest.fixture
def mock_beir_dataset(tmp_path: Path) -> Path:
    """Create a mock BEIR dataset directory structure with sample data."""
    dataset_dir = tmp_path / "mock_dataset"
    dataset_dir.mkdir(parents=True)
    qrels_dir = dataset_dir / "qrels"
    qrels_dir.mkdir(parents=True)

    # 1. corpus.jsonl
    corpus_content = (
        '{"_id": "doc1", "title": "FastAPI Guide", "text": "FastAPI is a modern web framework.", "metadata": {}}\n'
        '{"_id": "doc2", "title": "", "text": "Docker provides containerization.", "metadata": {}}\n'
        '{"_id": "doc3", "title": "Python Overview", "text": "Python is a multi-paradigm language.", "metadata": {}}\n'
    )
    (dataset_dir / "corpus.jsonl").write_text(corpus_content, encoding="utf-8")

    # 2. queries.jsonl
    queries_content = (
        '{"_id": "q1", "text": "What is FastAPI?", "metadata": {}}\n'
        '{"_id": "q2", "text": "Tell me about containers", "metadata": {}}\n'
        '{"_id": "q3", "text": "Unused query without qrels", "metadata": {}}\n'
    )
    (dataset_dir / "queries.jsonl").write_text(queries_content, encoding="utf-8")

    # 3. qrels/test.tsv
    qrels_content = (
        "query-id\tcorpus-id\tscore\n"
        "q1\tdoc1\t2\n"
        "q1\tdoc3\t0\n"
        "q2\tdoc2\t1\n"
    )
    (qrels_dir / "test.tsv").write_text(qrels_content, encoding="utf-8")

    return tmp_path


def test_load_corpus(mock_beir_dataset: Path) -> None:
    """Test loading and parsing corpus.jsonl into Chunk models."""
    loader = BEIRLoader(
        dataset_name="mock_dataset",
        cache_dir=mock_beir_dataset,
        auto_download=False,
    )
    chunks = loader.load_corpus()

    assert len(chunks) == 3
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].id == "doc1"
    assert chunks[0].document_id == "doc1"
    assert "FastAPI Guide\nFastAPI is a modern web framework." in chunks[0].text

    # Test title-less document
    assert chunks[1].id == "doc2"
    assert chunks[1].text == "Docker provides containerization."

    # Test max_docs limit
    limited_chunks = loader.load_corpus(max_docs=2)
    assert len(limited_chunks) == 2


def test_load_queries(mock_beir_dataset: Path) -> None:
    """Test loading and parsing queries.jsonl."""
    loader = BEIRLoader(
        dataset_name="mock_dataset",
        cache_dir=mock_beir_dataset,
        auto_download=False,
    )
    queries = loader.load_queries()

    assert len(queries) == 3
    assert queries["q1"] == "What is FastAPI?"
    assert queries["q2"] == "Tell me about containers"
    assert queries["q3"] == "Unused query without qrels"


def test_load_raw_qrels(mock_beir_dataset: Path) -> None:
    """Test loading and parsing qrels/test.tsv."""
    loader = BEIRLoader(
        dataset_name="mock_dataset",
        cache_dir=mock_beir_dataset,
        auto_download=False,
    )
    qrels = loader.load_raw_qrels(split="test")

    assert "q1" in qrels
    assert qrels["q1"]["doc1"] == 2
    assert qrels["q1"]["doc3"] == 0
    assert qrels["q2"]["doc2"] == 1


def test_load_benchmark(mock_beir_dataset: Path) -> None:
    """Test loading queries and qrels into standard Benchmark object."""
    loader = BEIRLoader(
        dataset_name="mock_dataset",
        cache_dir=mock_beir_dataset,
        auto_download=False,
    )
    benchmark = loader.load_benchmark(split="test", min_relevance_score=1)

    assert isinstance(benchmark, Benchmark)
    assert len(benchmark.cases) == 2

    # q1 has doc1 relevant (score 2 >= 1), doc3 ignored (score 0 < 1)
    case1 = next(c for c in benchmark.cases if c.query == "What is FastAPI?")
    assert case1.relevant_chunk_ids == ["doc1"]

    # q2 has doc2 relevant (score 1 >= 1)
    case2 = next(c for c in benchmark.cases if c.query == "Tell me about containers")
    assert case2.relevant_chunk_ids == ["doc2"]

    # Test max_queries limit
    limited_bench = loader.load_benchmark(split="test", max_queries=1)
    assert len(limited_bench.cases) == 1


def test_missing_files_raise_error(tmp_path: Path) -> None:
    """Test that missing files raise FileNotFoundError when auto_download is False."""
    empty_dir = tmp_path / "empty_dataset"
    empty_dir.mkdir(parents=True)

    loader = BEIRLoader(
        dataset_name="empty_dataset",
        cache_dir=tmp_path,
        auto_download=False,
    )

    with pytest.raises(FileNotFoundError, match="Corpus file not found"):
        loader.load_corpus()

    with pytest.raises(FileNotFoundError, match="Queries file not found"):
        loader.load_queries()

    with pytest.raises(FileNotFoundError, match="Qrels file not found"):
        loader.load_raw_qrels(split="test")
