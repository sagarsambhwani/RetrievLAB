"""
Algorithm

1. Read the document line by line.

2. If a heading is encountered:
   - Save the previous chunk.
   - Start a new chunk with this heading.

3. Otherwise:
   - If we've seen a heading, append the line to the current chunk.
   - Otherwise, append the line to the preamble.

4. At the end of the document, save the remaining chunk.
"""

from retrievlab.models import Document, Chunk

class MarkdownChunker:
    """
    A class for chunking Markdown documents into heading aware pieces.

    """
    def chunk(self, document: Document) -> list[Chunk]:
        """
        Chunk a Markdown document into heading aware pieces.

        Args:
            document: The Document object to be chunked.

        Returns:
            A list of Chunk objects.
        """
        chunks = []
        current_chunk = []
        current_heading = None

        for line in document.content.splitlines():
            if line.lstrip().startswith("#"):
                # If we encounter a new heading, save the current chunk
                if current_chunk:
                    chunks.append(Chunk(
                        id=f"{document.id}:{len(chunks)+1}",
                        document_id=document.id,
                        text="\n".join(current_chunk),
                        metadata={"heading": current_heading or "Preamble"}
                    ))
                current_chunk = [line]
                current_heading = line.strip()
            else:
                # If we have seen a heading, append the line to the current chunk
                current_chunk.append(line)


        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(Chunk(
                id=f"{document.id}:{len(chunks)+1}",
                document_id=document.id,
                text="\n".join(current_chunk),
                metadata={"heading": current_heading or "Preamble"}
            ))

        return chunks
