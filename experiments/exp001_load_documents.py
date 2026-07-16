"""
Experiment: 001

Question: Do the documents load correctly from the specified path?

Expected Result:
- The DocumentLoader should load all documents from the specified path.
"""

from pathlib import Path

from retrievlab.ingestion.loader import DocumentLoader

loader = DocumentLoader()

documents = loader.load(Path("data/raw"))

print(f"\nLoaded {len(documents)} documents\n")

for i, document in enumerate(documents, start=1):
    print("=" * 60)
    print(f"Document {i}")
    print("=" * 60)
    print(f"ID      : {document.id}")
    print(f"Title   : {document.title}")
    print(f"Source  : {document.source}")
    print(f"Content :\n{document.content[:50]}...")  # Print first 50 characters of content
    print()