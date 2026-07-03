# Experiment 000

## Objective

Verify the core domain models and project packaging.

## Hypothesis

The retrieval system should have stable data models that can be shared across all future components.

## Procedure

- Created Document
- Created Chunk
- Created Query
- Created SearchResult
- Executed exp000_models.py

## Result

PASS

All objects instantiated correctly.
Package imports correctly.
Editable installation works.

## Next

Implement the Document Loader.

# Experiment 002 — Markdown Chunker

Status: ✅ Passed

Dataset:
- Docker
- FastAPI
- Python

Observation:
Each document produced a single chunk because the dataset only contains one top-level heading.

Conclusion:
The parser behaves as designed.
A larger Markdown document is needed to evaluate heading-based chunking.

## Experiment 002

Status: Partial Success

Observation:
The parser correctly splits on H1 headings but ignores H2-H6 headings.

Next Action:
Generalize heading detection to all Markdown heading levels.