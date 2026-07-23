"""Extract plain text from documents (PDF and TXT)."""

from pathlib import Path

import pymupdf

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def extract_text(file_path: str | Path) -> str:
    """Extract full text content from a PDF or TXT file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    if suffix == ".txt":
        return path.read_text(encoding="utf-8")

    with pymupdf.open(path) as doc:
        pages = [page.get_text() for page in doc]
    return "\n".join(pages)