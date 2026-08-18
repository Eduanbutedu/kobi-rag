import pymupdf
import pytest

from eval.fetch_corpus import (
    DOWNLOADED,
    FAILED,
    SKIPPED,
    FetchResult,
    fetch_source,
    format_table,
    page_count,
    summarise,
)
from eval.sources import Source

MEVZUAT = Source("https://example.com/a.pdf", "is-kanunu", "mevzuat")
REHBER = Source("https://example.com/b.pdf", "kvkk-rehberi", "rehber")


def _pdf_bytes(pages=2):
    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def fake_download(monkeypatch):
    """Replace the network call with a queue of canned responses."""

    def _install(payload):
        def _download(url, session, chain_repair=None):
            if isinstance(payload, Exception):
                raise payload
            return payload

        monkeypatch.setattr("eval.fetch_corpus.download", _download)

    return _install


def test_downloads_and_records_page_count(tmp_path, fake_download):
    fake_download(_pdf_bytes(pages=3))
    result = fetch_source(MEVZUAT, tmp_path, session=None, force=False)

    assert result.status == DOWNLOADED
    assert result.pages == 3
    assert (tmp_path / "is-kanunu.pdf").exists()


def test_existing_file_is_skipped(tmp_path, fake_download):
    fake_download(_pdf_bytes())
    fetch_source(MEVZUAT, tmp_path, session=None, force=False)
    target = tmp_path / "is-kanunu.pdf"
    target.write_bytes(_pdf_bytes(pages=1))

    result = fetch_source(MEVZUAT, tmp_path, session=None, force=False)

    assert result.status == SKIPPED
    assert result.pages == 1  # yeniden indirilmedi, diskteki dosya okundu


def test_force_redownloads_an_existing_file(tmp_path, fake_download):
    (tmp_path / "is-kanunu.pdf").write_bytes(_pdf_bytes(pages=1))
    fake_download(_pdf_bytes(pages=5))

    result = fetch_source(MEVZUAT, tmp_path, session=None, force=True)

    assert result.status == DOWNLOADED
    assert result.pages == 5


def test_html_error_page_is_rejected_and_not_written(tmp_path, fake_download):
    # Site 404 yerine 200 + HTML hata sayfası dönebiliyor
    fake_download(b"<!DOCTYPE html><html><body>Sayfa bulunamadi</body></html>")

    result = fetch_source(MEVZUAT, tmp_path, session=None, force=False)

    assert result.status == FAILED
    assert "HTML" in result.detail
    assert not (tmp_path / "is-kanunu.pdf").exists()


def test_corrupt_pdf_is_deleted_after_writing(tmp_path, fake_download):
    # %PDF başlığı var ama içerik bozuk: dosya diskte bırakılmamalı
    fake_download(b"%PDF-1.7\nbozuk icerik")

    result = fetch_source(MEVZUAT, tmp_path, session=None, force=False)

    assert result.status == FAILED
    assert "unreadable" in result.detail
    assert not (tmp_path / "is-kanunu.pdf").exists()


def test_network_failure_is_reported_not_raised(tmp_path, fake_download):
    fake_download(RuntimeError("connection reset"))

    result = fetch_source(MEVZUAT, tmp_path, session=None, force=False)

    assert result.status == FAILED
    assert "connection reset" in result.detail


def test_page_count_returns_none_for_a_non_pdf(tmp_path):
    path = tmp_path / "x.pdf"
    path.write_bytes(b"not a pdf at all")
    assert page_count(path) is None


def test_table_lists_every_source():
    results = [
        FetchResult(MEVZUAT, DOWNLOADED, size_bytes=2048, pages=10),
        FetchResult(REHBER, FAILED, detail="HTTP 404"),
    ]
    table = format_table(results)
    assert "is-kanunu" in table
    assert "kvkk-rehberi" in table
    assert "HTTP 404" in table


def test_summary_counts_each_outcome():
    results = [
        FetchResult(MEVZUAT, DOWNLOADED, pages=10),
        FetchResult(REHBER, SKIPPED, pages=5),
        FetchResult(Source("https://e.com/c.pdf", "yok", "mevzuat"), FAILED, detail="HTTP 404"),
    ]
    summary = summarise(results)
    assert "1 downloaded, 1 already present, 1 failed" in summary
    assert "2 usable PDF(s), 15 pages" in summary
    assert "yok: HTTP 404" in summary


def test_summary_warns_about_copyright_only_when_relevant():
    with_rehber = summarise([FetchResult(REHBER, DOWNLOADED, pages=5)])
    only_mevzuat = summarise([FetchResult(MEVZUAT, DOWNLOADED, pages=5)])
    assert "copyright" in with_rehber
    assert "copyright" not in only_mevzuat


def test_failed_source_does_not_count_as_usable():
    summary = summarise([FetchResult(REHBER, FAILED, detail="HTTP 500")])
    assert "0 usable PDF(s), 0 pages" in summary
    assert "copyright" not in summary
