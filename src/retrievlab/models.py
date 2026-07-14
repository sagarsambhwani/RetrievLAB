"""
Data models for representing documents, chunks, queries, and search results.
"""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    title: str
    source: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str

    text: str

    metadata: dict[str, Any] = Field(default_factory=dict)

    embedding: list[float] | None = None


class Query(BaseModel):
    text: str

    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    chunk: Chunk

    score: float