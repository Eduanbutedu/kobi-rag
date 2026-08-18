"""Parsing and validation for the evaluation corpus source list.

Kept free of network and disk access beyond reading the list itself, so the
rules that decide what is a valid source -- and what is really a PDF -- can
be unit tested.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from eval.textio import read_text_utf8

# Slug dosya adına dönüştüğü için dar tutuluyor: dizin ayracı ve ".." giremez
SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
KINDS = ("mevzuat", "rehber")
PDF_MAGIC = b"%PDF"
# Sihirli bayt dosyanın hemen başında olmayabilir; PDF spec'i ilk 1KB'ı kabul eder
MAGIC_WINDOW = 1024


class SourcesError(ValueError):
    """Raised when the source list is malformed."""


@dataclass(frozen=True)
class Source:
    """One downloadable corpus document."""

    url: str
    slug: str
    kind: str

    @property
    def filename(self) -> str:
        return f"{self.slug}.pdf"

    @property
    def is_redistributable(self) -> bool:
        """Whether the document may be committed to the repository."""
        return self.kind == "mevzuat"


def parse_source_line(line: str, line_number: int = 0) -> Source | None:
    """Parse one line into a Source. Returns None for blank and comment lines."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    parts = [part.strip() for part in stripped.split("|")]
    if len(parts) != 3:
        raise SourcesError(
            f"line {line_number}: expected 'URL | slug | kind', got {len(parts)} field(s)"
        )

    url, slug, kind = parts
    if not url.startswith(("http://", "https://")):
        raise SourcesError(f"line {line_number}: URL must start with http:// or https://")
    if not SLUG_PATTERN.match(slug):
        raise SourcesError(
            f"line {line_number}: invalid slug '{slug}' "
            "(use lowercase letters, digits and hyphens)"
        )
    if kind not in KINDS:
        raise SourcesError(
            f"line {line_number}: unknown kind '{kind}' (expected one of {', '.join(KINDS)})"
        )
    return Source(url=url, slug=slug, kind=kind)


def load_sources(path: str | Path) -> list[Source]:
    """Read the source list. Slugs must be unique because they become filenames."""
    file_path = Path(path)
    if not file_path.exists():
        raise SourcesError(f"source list not found: {file_path}")

    sources: list[Source] = []
    seen: dict[str, int] = {}
    for number, line in enumerate(read_text_utf8(file_path).splitlines(), start=1):
        source = parse_source_line(line, number)
        if source is None:
            continue
        if source.slug in seen:
            raise SourcesError(
                f"line {number}: duplicate slug '{source.slug}', "
                f"first seen on line {seen[source.slug]}"
            )
        seen[source.slug] = number
        sources.append(source)

    if not sources:
        raise SourcesError(f"no sources listed in {file_path}")
    return sources


def looks_like_pdf(data: bytes) -> bool:
    """Whether the bytes carry the %PDF magic number near the start."""
    return PDF_MAGIC in data[:MAGIC_WINDOW]


def looks_like_html(data: bytes) -> bool:
    """Whether the bytes look like an HTML page, the usual shape of a soft 404."""
    head = data[:MAGIC_WINDOW].lstrip().lower()
    return head.startswith((b"<!doctype html", b"<html")) or b"<html" in head


def describe_payload(data: bytes) -> str:
    """Short description of what arrived instead of a PDF, for error reporting."""
    if not data:
        return "empty response"
    if looks_like_html(data):
        return "HTML page (site returned an error page instead of the file)"
    return f"not a PDF (starts with {data[:8]!r})"
