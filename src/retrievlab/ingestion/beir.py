"""
BEIR (Benchmarking Information Retrieval) dataset loader and schema adapter.

Supports automated downloading, caching, and conversion of standard BEIR datasets
(e.g., SciFact, NFCorpus, FiQA) into RetrievLab's Chunk and Benchmark data models.
"""

from pathlib import Path
import json
import zipfile
import urllib.request
import logging

from retrievlab.models import Chunk
from retrievlab.evaluation.benchmark import Benchmark, BenchmarkCase

logger = logging.getLogger(__name__)


class BEIRLoader:
    """Automated downloader, parser, and adapter for BEIR benchmark datasets.
    
    Standard BEIR datasets contain:
    - corpus.jsonl: {"_id": str, "title": str, "text": str, "metadata": dict}
    - queries.jsonl: {"_id": str, "text": str, "metadata": dict}
    - qrels/{split}.tsv: query-id \\t corpus-id \\t score
    """

    BEIR_BASE_URL: str = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"

    def __init__(
        self,
        dataset_name: str,
        cache_dir: str | Path = "data/beir",
        auto_download: bool = True,
    ) -> None:
        """Initialize BEIRLoader for a specific BEIR dataset.
        
        Args:
            dataset_name: Name of the BEIR dataset (e.g., 'scifact', 'nfcorpus', 'fiqa').
            cache_dir: Directory where datasets will be downloaded and cached.
            auto_download: Whether to automatically download the dataset if missing.
        """
        self.dataset_name = dataset_name.lower().strip()
        self.cache_dir = Path(cache_dir)
        self.dataset_dir = self.cache_dir / self.dataset_name
        self.auto_download = auto_download

        if self.auto_download:
            self.download_if_missing()

    def download_if_missing(self) -> Path:
        """Download and extract the BEIR dataset if it does not already exist locally.
        
        Returns:
            Path to the local dataset directory.
            
        Raises:
            RuntimeError: If download or extraction fails.
        """
        corpus_path = self.dataset_dir / "corpus.jsonl"
        queries_path = self.dataset_dir / "queries.jsonl"

        if corpus_path.exists() and queries_path.exists():
            return self.dataset_dir

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        zip_url = f"{self.BEIR_BASE_URL}/{self.dataset_name}.zip"
        zip_dest = self.cache_dir / f"{self.dataset_name}.zip"

        logger.info(f"Downloading BEIR dataset '{self.dataset_name}' from {zip_url}...")
        try:
            import requests
            response = requests.get(zip_url, stream=True, timeout=60)
            response.raise_for_status()
            with open(zip_dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            if zip_dest.exists():
                zip_dest.unlink()
            raise RuntimeError(
                f"Failed to download BEIR dataset '{self.dataset_name}' from {zip_url}: {e}"
            ) from e

        logger.info(f"Extracting '{self.dataset_name}.zip' to {self.cache_dir}...")
        try:
            with zipfile.ZipFile(zip_dest, "r") as zip_ref:
                zip_ref.extractall(self.cache_dir)
        except Exception as e:
            raise RuntimeError(
                f"Failed to extract BEIR archive '{zip_dest}': {e}"
            ) from e
        finally:
            if zip_dest.exists():
                zip_dest.unlink()

        if not corpus_path.exists() or not queries_path.exists():
            raise RuntimeError(
                f"BEIR dataset '{self.dataset_name}' extracted, but corpus.jsonl or queries.jsonl not found."
            )

        return self.dataset_dir

    def load_corpus(self, max_docs: int | None = None) -> list[Chunk]:
        """Load and parse corpus.jsonl into a list of standard RetrievLab Chunks.
        
        Args:
            max_docs: Optional maximum number of documents to load (useful for smoke testing).
            
        Returns:
            List of Chunk objects.
            
        Raises:
            FileNotFoundError: If corpus.jsonl is not found in dataset directory.
        """
        corpus_path = self.dataset_dir / "corpus.jsonl"
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

        chunks: list[Chunk] = []
        with corpus_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if max_docs is not None and line_idx >= max_docs:
                    break
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                doc_id = str(doc["_id"])
                title = doc.get("title", "").strip()
                text = doc.get("text", "").strip()

                full_text = f"{title}\n{text}".strip() if title else text

                chunk = Chunk(
                    id=doc_id,
                    document_id=doc_id,
                    text=full_text,
                    embedding=None,
                )
                chunks.append(chunk)

        return chunks

    def load_queries(self) -> dict[str, str]:
        """Load and parse queries.jsonl into a mapping of query_id -> query_text.
        
        Returns:
            Dictionary mapping query_id to query_text string.
            
        Raises:
            FileNotFoundError: If queries.jsonl is not found.
        """
        queries_path = self.dataset_dir / "queries.jsonl"
        if not queries_path.exists():
            raise FileNotFoundError(f"Queries file not found: {queries_path}")

        queries: dict[str, str] = {}
        with queries_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                queries[str(item["_id"])] = item.get("text", "").strip()

        return queries

    def load_raw_qrels(self, split: str = "test") -> dict[str, dict[str, int]]:
        """Load raw relevance judgments from qrels/{split}.tsv.
        
        Args:
            split: Dataset split ('test', 'dev', or 'train'). Defaults to 'test'.
            
        Returns:
            Nested dictionary mapping query_id -> {doc_id: relevance_score}.
            
        Raises:
            FileNotFoundError: If qrels/{split}.tsv is not found.
        """
        qrels_path = self.dataset_dir / "qrels" / f"{split}.tsv"
        if not qrels_path.exists():
            raise FileNotFoundError(f"Qrels file not found: {qrels_path}")

        qrels: dict[str, dict[str, int]] = {}
        with qrels_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("query-id"):  # Header guard
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    qid, doc_id, score_str = parts[0], parts[1], parts[2]
                    try:
                        score = int(score_str)
                    except ValueError:
                        score = int(float(score_str))

                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][doc_id] = score

        return qrels

    def load_benchmark(
        self,
        split: str = "test",
        min_relevance_score: int = 1,
        max_queries: int | None = None,
    ) -> Benchmark:
        """Load queries and qrels into a standard RetrievLab Benchmark.
        
        Args:
            split: Dataset split to load ('test', 'dev', 'train'). Defaults to 'test'.
            min_relevance_score: Minimum relevance score to consider a document relevant (>= 1).
            max_queries: Optional limit on the number of query test cases to load.
            
        Returns:
            Benchmark object containing BenchmarkCases with query text and relevant chunk IDs.
        """
        queries = self.load_queries()
        qrels = self.load_raw_qrels(split=split)

        cases: list[BenchmarkCase] = []
        for qid, judgments in qrels.items():
            if qid not in queries:
                continue

            relevant_ids = [
                doc_id for doc_id, score in judgments.items()
                if score >= min_relevance_score
            ]

            if not relevant_ids:
                continue

            cases.append(
                BenchmarkCase(
                    query=queries[qid],
                    relevant_chunk_ids=relevant_ids,
                )
            )

            if max_queries is not None and len(cases) >= max_queries:
                break

        return Benchmark(cases=cases)
