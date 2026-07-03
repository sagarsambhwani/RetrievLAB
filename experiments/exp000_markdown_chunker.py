"""
Experiment 002

Question:
Does the MarkdownChunker correctly split Markdown documents into
heading-based chunks while preserving preamble text?

Expected Result:
- Every heading starts a new chunk.
- The heading is included in the chunk text.
- Text before the first heading becomes a "Preamble" chunk.
"""

from pathlib import Path

from retrievlab.ingestion.loader import DocumentLoader
from retrievlab.chunking.markdown import MarkdownChunker
loader = DocumentLoader()
chunker = MarkdownChunker()

documents = loader.load(Path("data/raw/big_fastapi.md"))
chunks = chunker.chunk(documents[0])  # Chunk the first document for demonstration

print(f"\nLoaded {len(documents)} documents")
print(f"Chunked into {len(chunks)} chunks\n")

for document in documents:
    chunks = chunker.chunk(document)

    print(f"\n{'=' * 80}")
    print(f"Document: {document.title}")
    print(f"Produced {len(chunks)} chunks")

    for i, chunk in enumerate(chunks, start=1):
        print(f"\nChunk {i}")
        print(f"Heading: {chunk.metadata['heading']}")
        print(chunk.text)
