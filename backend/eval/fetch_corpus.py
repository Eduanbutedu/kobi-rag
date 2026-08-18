"""Download the evaluation corpus listed in eval/corpus_sources.txt.

Downloads are sequential and politely spaced. A source that fails is
reported and skipped -- one dead link never stops the run.

    python -m eval.fetch_corpus
    python -m eval.fetch_corpus --force --only is-kanunu
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import pymupdf
import requests

from eval.sources import Source, SourcesError, describe_payload, load_sources, looks_like_pdf

DEFAULT_SOURCES = Path("eval/corpus_sources.txt")
DEFAULT_TARGET_DIR = Path("data/corpus")
TIMEOUT_SECONDS = 60
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 2.0
DELAY_BETWEEN_REQUESTS = 1.0

# mevzuat.gov.tr requests'in varsayılan User-Agent'ıyla gelen istekleri reddedebiliyor
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
}

DOWNLOADED = "downloaded"
SKIPPED = "skipped"
FAILED = "ERROR"


@dataclass
class FetchResult:
    """Outcome of one source: what happened and what we know about the file."""

    source: Source
    status: str
    size_bytes: int = 0
    pages: int | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status != FAILED


def page_count(path: Path) -> int | None:
    """Number of pages in a PDF, or None if it cannot be opened."""
    try:
        with pymupdf.open(path) as doc:
            return doc.page_count
    except Exception:  # noqa: BLE001 - bozuk PDF'in türü önemli değil, sayfa sayısı yok
        return None


def download(url: str, session: requests.Session) -> bytes:
    """Fetch a URL with retries and exponential backoff. Raises on final failure."""
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = session.get(url, headers=BROWSER_HEADERS, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            last_error = exc
            if attempt < MAX_ATTEMPTS:
                wait = BACKOFF_SECONDS * (2 ** (attempt - 1))
                print(f"      attempt {attempt}/{MAX_ATTEMPTS} failed ({exc}); retrying in {wait}s")
                time.sleep(wait)
    raise RuntimeError(str(last_error))


def fetch_source(
    source: Source, target_dir: Path, session: requests.Session, force: bool
) -> FetchResult:
    """Download one source into target_dir, validating that it really is a PDF."""
    target = target_dir / source.filename
    if target.exists() and not force:
        return FetchResult(
            source, SKIPPED, size_bytes=target.stat().st_size, pages=page_count(target)
        )

    try:
        data = download(source.url, session)
    except RuntimeError as exc:
        return FetchResult(source, FAILED, detail=str(exc)[:120])

    # Site 404 yerine 200 + HTML hata sayfası dönebiliyor; içeriği doğrula
    if not looks_like_pdf(data):
        return FetchResult(source, FAILED, size_bytes=len(data), detail=describe_payload(data))

    target.write_bytes(data)
    pages = page_count(target)
    if pages is None:
        target.unlink(missing_ok=True)
        return FetchResult(
            source, FAILED, size_bytes=len(data), detail="PDF header present but unreadable"
        )
    return FetchResult(source, DOWNLOADED, size_bytes=len(data), pages=pages)


def format_table(results: list[FetchResult]) -> str:
    """Render the run as an aligned table."""
    header = f"{'slug':<34}{'kind':<10}{'size':>10}{'pages':>7}  status"
    lines = ["", header, "-" * (len(header) + 20)]
    for result in results:
        size = f"{result.size_bytes / 1024:,.0f} KB" if result.size_bytes else "-"
        pages = str(result.pages) if result.pages is not None else "-"
        line = (
            f"{result.source.slug:<34}{result.source.kind:<10}"
            f"{size:>10}{pages:>7}  {result.status}"
        )
        if result.detail:
            line += f": {result.detail}"
        lines.append(line)
    return "\n".join(lines)


def summarise(results: list[FetchResult]) -> str:
    """One-line-per-fact summary of the whole run."""
    downloaded = [r for r in results if r.status == DOWNLOADED]
    skipped = [r for r in results if r.status == SKIPPED]
    failed = [r for r in results if r.status == FAILED]
    usable = downloaded + skipped
    total_pages = sum(r.pages or 0 for r in usable)

    lines = [
        "",
        f"{len(downloaded)} downloaded, {len(skipped)} already present, {len(failed)} failed.",
        f"{len(usable)} usable PDF(s), {total_pages} pages in total.",
    ]
    if failed:
        lines.append("")
        lines.append("Failed sources (left out of the corpus):")
        lines += [f"  {r.source.slug}: {r.detail}" for r in failed]
    if any(not r.source.is_redistributable for r in usable):
        lines.append("")
        lines.append(
            "Note: 'rehber' documents are under copyright. They stay in data/corpus/,\n"
            "      which is gitignored, and must not be committed."
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the evaluation corpus.")
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES, help="source list")
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR, help="download dir")
    parser.add_argument("--force", action="store_true", help="re-download files already present")
    parser.add_argument("--only", nargs="+", metavar="SLUG", help="fetch just these slugs")
    args = parser.parse_args()

    try:
        sources = load_sources(args.sources)
    except SourcesError as exc:
        raise SystemExit(str(exc)) from exc

    if args.only:
        wanted = set(args.only)
        unknown = wanted - {s.slug for s in sources}
        if unknown:
            raise SystemExit(f"unknown slug(s): {', '.join(sorted(unknown))}")
        sources = [s for s in sources if s.slug in wanted]

    args.target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {len(sources)} source(s) into {args.target_dir}\n")

    results: list[FetchResult] = []
    with requests.Session() as session:
        for index, source in enumerate(sources, start=1):
            print(f"[{index:>2}/{len(sources)}] {source.slug}")
            result = fetch_source(source, args.target_dir, session, args.force)
            results.append(result)
            print(f"      {result.status}{': ' + result.detail if result.detail else ''}")
            # Siteyi yormamak için istekler arasında bekle
            if result.status == DOWNLOADED and index < len(sources):
                time.sleep(DELAY_BETWEEN_REQUESTS)

    print(format_table(results))
    print(summarise(results))


if __name__ == "__main__":
    main()
