import pytest
import requests

from eval.ingest_corpus import (
    FAILED,
    SKIPPED,
    UPLOADED,
    IngestResult,
    collect_pdfs,
    existing_documents,
    format_summary,
    upload,
)


class _FakeResponse:
    def __init__(self, payload=None, error=None):
        self._payload = payload or {}
        self._error = error

    def raise_for_status(self):
        if self._error:
            raise self._error

    def json(self):
        return self._payload


class _FakeSession:
    """Stands in for requests.Session, recording what was sent."""

    def __init__(self, get_response=None, post_response=None):
        self._get = get_response
        self._post = post_response
        self.posted_urls = []

    def get(self, url, timeout=None):
        return self._get

    def post(self, url, files=None, timeout=None):
        self.posted_urls.append(url)
        return self._post


def test_existing_documents_maps_source_to_chunk_count():
    session = _FakeSession(
        get_response=_FakeResponse(
            {"documents": [{"source": "a.pdf", "chunks": 12}, {"source": "b.pdf", "chunks": 3}]}
        )
    )
    assert existing_documents("http://x", session) == {"a.pdf": 12, "b.pdf": 3}


def test_existing_documents_is_empty_for_a_fresh_store():
    session = _FakeSession(get_response=_FakeResponse({"documents": []}))
    assert existing_documents("http://x", session) == {}


def test_upload_reports_the_chunk_count(tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.7")
    session = _FakeSession(post_response=_FakeResponse({"filename": "a.pdf", "chunks": 42}))

    result = upload(path, "http://x", session)

    assert result.status == UPLOADED
    assert result.chunks == 42
    assert session.posted_urls == ["http://x/documents"]


def test_upload_treats_zero_chunks_as_a_failure(tmp_path):
    # Taranmış PDF'ten metin çıkmayabilir; sessizce başarılı sayılmamalı
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.7")
    session = _FakeSession(post_response=_FakeResponse({"filename": "a.pdf", "chunks": 0}))

    result = upload(path, "http://x", session)

    assert result.status == FAILED
    assert "0 chunks" in result.detail


def test_upload_reports_a_server_error_without_raising(tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.7")
    session = _FakeSession(
        post_response=_FakeResponse(error=requests.HTTPError("400 Unsupported file type"))
    )

    result = upload(path, "http://x", session)

    assert result.status == FAILED
    assert "Unsupported file type" in result.detail


def test_collect_pdfs_is_sorted_and_ignores_other_files(tmp_path):
    for name in ("b.pdf", "a.pdf", "notes.txt", "readme.md"):
        (tmp_path / name).write_bytes(b"x")
    assert [p.name for p in collect_pdfs(tmp_path)] == ["a.pdf", "b.pdf"]


def test_collect_pdfs_matches_the_extension_case_insensitively(tmp_path):
    # glob'un büyük/küçük harf duyarlılığı platforma göre değişiyor
    for name in ("a.pdf", "B.PDF", "c.Pdf"):
        (tmp_path / name).write_bytes(b"x")
    assert [p.name for p in collect_pdfs(tmp_path)] == ["a.pdf", "B.PDF", "c.Pdf"]


def test_collect_pdfs_ignores_subdirectories(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"x")
    (tmp_path / "nested.pdf").mkdir()
    assert [p.name for p in collect_pdfs(tmp_path)] == ["a.pdf"]


def test_collect_pdfs_is_empty_for_an_empty_dir(tmp_path):
    assert collect_pdfs(tmp_path) == []


def test_summary_counts_outcomes_and_store_totals():
    results = [
        IngestResult("a.pdf", UPLOADED, chunks=100),
        IngestResult("b.pdf", UPLOADED, chunks=50),
        IngestResult("c.pdf", SKIPPED, chunks=10),
        IngestResult("d.pdf", FAILED, detail="HTTP 500"),
    ]
    summary = format_summary(results, total_docs=3, total_chunks=160)

    assert "2 uploaded, 1 already indexed, 1 failed" in summary
    assert "150 new chunk(s)" in summary
    assert "d.pdf: HTTP 500" in summary
    assert "3 document(s) and 160 chunk(s)" in summary


def test_summary_without_uploads_omits_the_new_chunk_line():
    summary = format_summary([IngestResult("a.pdf", SKIPPED, chunks=10)], 1, 10)
    assert "new chunk(s)" not in summary
    assert "0 uploaded, 1 already indexed, 0 failed" in summary


@pytest.mark.parametrize("base", ["http://127.0.0.1:8000/", "http://127.0.0.1:8000"])
def test_trailing_slash_in_base_url_is_harmless(base):
    assert base.rstrip("/") == "http://127.0.0.1:8000"
