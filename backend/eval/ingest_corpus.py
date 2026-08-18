"""Upload the downloaded corpus to a running backend via POST /documents.

Documents already present on the server are skipped, so the script is safe
to re-run after adding new PDFs to data/corpus/.

Start the API first (uvicorn app.main:app), then:

    python -m eval.ingest_corpus
    python -m eval.ingest_corpus --base-url http://127.0.0.1:8000
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_CORPUS_DIR = Path("data/corpus")
UPLOAD_TIMEOUT_SECONDS = 600
LIST_TIMEOUT_SECONDS = 30

UPLOADED = "uploaded"
SKIPPED = "skipped"
FAILED = "ERROR"


@dataclass
class IngestResult:
    """Outcome of uploading one file."""

    filename: str
    status: str
    chunks: int = 0
    detail: str = ""


def existing_documents(base_url: str, session: requests.Session) -> dict[str, int]:
    """Return {source: chunk_count} already indexed by the server."""
    response = session.get(f"{base_url}/documents", timeout=LIST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return {doc["source"]: doc["chunks"] for doc in response.json()["documents"]}


def upload(path: Path, base_url: str, session: requests.Session) -> IngestResult:
    """Upload one PDF and report how many chunks it produced."""
    try:
        with path.open("rb") as handle:
            response = session.post(
                f"{base_url}/documents",
                files={"file": (path.name, handle, "application/pdf")},
                timeout=UPLOAD_TIMEOUT_SECONDS,
            )
        response.raise_for_status()
    except requests.RequestException as exc:
        return IngestResult(path.name, FAILED, detail=str(exc)[:150])

    chunks = response.json().get("chunks", 0)
    if chunks == 0:
        return IngestResult(path.name, FAILED, detail="no text extracted (0 chunks)")
    return IngestResult(path.name, UPLOADED, chunks=chunks)


def collect_pdfs(corpus_dir: Path) -> list[Path]:
    """PDFs in the corpus directory, in a stable order.

    The extension is matched case-insensitively rather than with glob, whose
    case sensitivity depends on the platform.
    """
    return sorted(
        (p for p in corpus_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=lambda p: p.name.lower(),
    )


def format_summary(results: list[IngestResult], total_docs: int, total_chunks: int) -> str:
    """Render the per-file outcomes and the resulting store totals."""
    uploaded = [r for r in results if r.status == UPLOADED]
    skipped = [r for r in results if r.status == SKIPPED]
    failed = [r for r in results if r.status == FAILED]

    lines = ["", f"{len(uploaded)} uploaded, {len(skipped)} already indexed, {len(failed)} failed."]
    if uploaded:
        lines.append(f"{sum(r.chunks for r in uploaded)} new chunk(s) from this run.")
    if failed:
        lines.append("")
        lines.append("Failed uploads:")
        lines += [f"  {r.filename}: {r.detail}" for r in failed]
    lines += ["", f"Store now holds {total_docs} document(s) and {total_chunks} chunk(s)."]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload the corpus to a running backend.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR, help="PDF folder")
    parser.add_argument("--force", action="store_true", help="upload even if already indexed")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    if not args.corpus_dir.exists():
        raise SystemExit(f"{args.corpus_dir} does not exist. Run eval.fetch_corpus first.")

    pdfs = collect_pdfs(args.corpus_dir)
    if not pdfs:
        raise SystemExit(f"No PDFs in {args.corpus_dir}. Run eval.fetch_corpus first.")

    with requests.Session() as session:
        try:
            indexed = existing_documents(base_url, session)
        except requests.RequestException as exc:
            raise SystemExit(
                f"Cannot reach the API at {base_url}: {exc}\n"
                "Start it with: uvicorn app.main:app --reload"
            ) from exc

        print(f"Uploading {len(pdfs)} PDF(s) to {base_url}; {len(indexed)} already indexed.\n")

        results: list[IngestResult] = []
        for index, path in enumerate(pdfs, start=1):
            prefix = f"[{index:>2}/{len(pdfs)}] {path.name}"
            if path.name in indexed and not args.force:
                results.append(
                    IngestResult(path.name, SKIPPED, chunks=indexed[path.name])
                )
                print(f"{prefix}: skipped, already indexed ({indexed[path.name]} chunks)")
                continue

            print(f"{prefix}: uploading...")
            result = upload(path, base_url, session)
            results.append(result)
            if result.status == UPLOADED:
                print(f"{prefix}: {result.chunks} chunks")
            else:
                print(f"{prefix}: {result.status} - {result.detail}")

        try:
            final = existing_documents(base_url, session)
        except requests.RequestException as exc:
            raise SystemExit(f"Upload finished but the document list failed: {exc}") from exc

    print(format_summary(results, len(final), sum(final.values())))


if __name__ == "__main__":
    main()
