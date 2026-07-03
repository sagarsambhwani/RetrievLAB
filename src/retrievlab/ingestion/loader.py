from pathlib import Path
from retrievlab.models import Document

class DocumentLoader:

    def load(self, path: Path) -> list[Document]:
        """
        Load all Markdown documents from a directory.

        Args:
            path: Directory containing Markdown files.

        Returns:
            A list of Document objects.
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        documents = []
        
        for file in path.iterdir():
            if file.is_file() and file.suffix == ".md":
                content = file.read_text(encoding="utf-8")
                doc = Document(
                    title=self._extract_title(content, file.stem),
                    source=file.name,
                    content=content
                )
                documents.append(doc)
        return documents
    
    def _extract_title(self, content: str, fallback: str) -> str:
        """
        Extract the title from the specified content.

        Args:
            content (str): The content of the document.
            fallback (str): The fallback title to use if the title cannot be extracted.
        """
        # Example implementation - replace with actual title extraction logic
        for line in content.splitlines():
                if line.startswith("# "):
                    return line[2:].strip()
        return fallback