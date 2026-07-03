from retrievlab.models import Document, Chunk, Query, SearchResult

doc = Document(
    title="FastAPI",
    source="fastapi.md",
    content="FastAPI is a modern, high-performance web framework."
)

chunk = Chunk(
    document_id=doc.id,
    text=doc.content,
)

query = Query(
    text="What is FastAPI?"
)

result = SearchResult(
    chunk=chunk,
    score=0.98
)

print("=" * 50)
print(doc.model_dump())
print("=" * 50)
print(chunk.model_dump())
print("=" * 50)
print(query.model_dump())
print("=" * 50)
print(result.model_dump())