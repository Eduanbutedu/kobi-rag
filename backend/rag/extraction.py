"""Extract plain text from documents (PDF and TXT)."""

import re
from pathlib import Path

import pymupdf

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

# Tek başına bir satır olarak "References" / "Bibliography" / "Kaynakça" vb.
# (başında bölüm numarası olabilir: "5 References", "7. Kaynakça")
_REFERENCES_HEADING = re.compile(
    r"^\s*(?:\d+\.?\s+)?(references|bibliography|kaynak(?:ça|lar))\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def strip_references(text: str) -> str:
    """Drop a trailing references/bibliography section from academic text.

    Reference lists produce many chunks that are useless for Q&A and can
    outrank real content. The cut is only applied when the heading sits in
    the second half of the document, so an early mention like a table of
    contents entry never truncates the body.
    """
    matches = list(_REFERENCES_HEADING.finditer(text))
    if not matches:
        return text
    cut = matches[-1].start()
    if cut > len(text) * 0.5:
        return text[:cut].rstrip()
    return text


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
    return strip_references("\n".join(pages))