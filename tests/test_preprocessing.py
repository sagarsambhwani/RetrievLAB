import pytest
from retrievlab.models import Chunk
from retrievlab.preprocessing import (
    BasicWordTokenizer,
    RegexTokenizer,
    StemmedTokenizer,
    StopwordTokenizer,
)
from retrievlab.retrieval.bm25 import BM25Retriever


def test_basic_word_tokenizer():
    tokenizer = BasicWordTokenizer()
    tokens = tokenizer.tokenize("Hello, World! Tokenization 123.")
    assert tokens == ["hello", "world", "tokenization", "123"]


def test_regex_tokenizer():
    tokenizer = RegexTokenizer(pattern=r"[a-zA-Z]+")
    tokens = tokenizer.tokenize("Hello, World! Tokenization 123.")
    assert tokens == ["hello", "world", "tokenization"]


def test_stopword_tokenizer():
    tokenizer = StopwordTokenizer()
    tokens = tokenizer.tokenize("This is a simple test of the stopword filter.")
    assert "this" not in tokens
    assert "is" not in tokens
    assert "a" not in tokens
    assert "of" not in tokens
    assert "the" not in tokens
    assert "simple" in tokens
    assert "test" in tokens
    assert "stopword" in tokens
    assert "filter" in tokens


def test_stemmed_tokenizer_english_words():
    tokenizer = StemmedTokenizer(algorithm="snowball")

    assert tokenizer.stem("caresses") == "caress"
    assert tokenizer.stem("ponies") == "poni"
    assert tokenizer.stem("cats") == "cat"
    assert tokenizer.stem("motoring") == "motor"
    assert tokenizer.stem("sing") == "sing"
    assert tokenizer.stem("hopping") == "hop"
    assert tokenizer.stem("happy") == "happi"
    assert tokenizer.stem("relational") == "relat"
    assert tokenizer.stem("goodness") == "good"
    assert tokenizer.stem("replacement") == "replac"


def test_stemmed_tokenizer_ir_terms():
    tokenizer = StemmedTokenizer(algorithm="porter")

    assert tokenizer.stem("retrieval") == "retriev"
    assert tokenizer.stem("retrieve") == "retriev"
    assert tokenizer.stem("retrieved") == "retriev"
    assert tokenizer.stem("retrieving") == "retriev"
    assert tokenizer.stem("retrievers") == "retriev"
    assert tokenizer.stem("queries") == "queri"
    assert tokenizer.stem("querying") == "queri"
    assert tokenizer.stem("indexing") == "index"
    assert tokenizer.stem("indexes") == "index"
    assert tokenizer.stem("indexed") == "index"


def test_stemmed_tokenizer_lancaster():
    tokenizer = StemmedTokenizer(algorithm="lancaster")

    # Lancaster is aggressive: maximum -> maxim, continuous -> continu, ability -> abl
    assert tokenizer.stem("maximum") == "maxim"
    assert tokenizer.stem("continuous") == "continu"
    assert tokenizer.stem("ability") == "abl"


def test_stemmed_tokenizer_unsupported_algorithm():
    with pytest.raises(ValueError, match="Unsupported stemming algorithm"):
        StemmedTokenizer(algorithm="unknown_algorithm")


def test_stemmed_tokenizer_edge_cases():
    tokenizer = StemmedTokenizer(algorithm="porter")

    assert tokenizer.stem("") == ""
    assert tokenizer.stem("a") == "a"
    assert tokenizer.stem("to") == "to"
    assert tokenizer.stem("123") == "123"


def test_stemmed_tokenizer_full_sentence():
    tokenizer = StemmedTokenizer(algorithm="snowball")
    tokens = tokenizer.tokenize("Information retrieval systems are retrieving documents.")
    assert tokens == ["inform", "retriev", "system", "are", "retriev", "document"]


def test_bm25_with_stemmed_tokenizer():
    stemmed_tokenizer = StemmedTokenizer(algorithm="porter")
    retriever = BM25Retriever(tokenizer=stemmed_tokenizer)

    c1 = Chunk(id="c1", document_id="doc1", text="Information retrieval systems and algorithms")
    c2 = Chunk(id="c2", document_id="doc1", text="Database transactions and locking protocols")

    retriever.index([c1, c2])

    # Query uses inflection 'retrieved' which stems to 'retriev' and matches chunk c1
    results = retriever.retrieve("retrieved", top_k=2, chunks=[c1, c2])
    assert len(results) == 2
    assert results[0].chunk.id == "c1"
    assert results[0].score > 0.0
    assert results[1].chunk.id == "c2"
    assert results[1].score == 0.0
