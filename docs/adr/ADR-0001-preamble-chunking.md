# ADR-0001: Markdown Preamble Chunking

**Status**: Accepted  
**Deciders**: RetrievLab Team  
**Date**: 2026-08-14  

---

## Context
When chunking structured Markdown documents by heading levels (H1-H6), documents frequently contain introductory text, metadata, or summary paragraphs preceding the first explicit `# Heading` tag. 

If chunking solely relies on splitting on `#` markers, this preamble content would either be dropped or inadvertently prepended to the first section.

## Decision
Create a dedicated `Chunk` for any non-empty preamble content appearing before the first detected Markdown heading.

The preamble chunk is assigned:
- Chunk ID: `doc_id:0` (or `doc_id:preamble`)
- Metadata: `heading=None`, `heading_level=0`

## Consequences
### Positive
- Zero information loss during Markdown document ingestion.
- Explicit visibility of document intros in the retrieval candidate pool.
- Clean heading hierarchy preserved for all subsequent chunks.

### Negative / Tradeoffs
- Preamble chunks may be smaller than standard section chunks and might require minimum token length validation in future chunking iterations.

## Alternatives Considered
- **Drop preamble**: Rejected because introductory overviews often contain high-value semantic summaries.
- **Merge into first section**: Rejected because it pollutes the semantics and heading metadata of the first topic.
