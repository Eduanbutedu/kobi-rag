import pytest

from eval.sources import (
    Source,
    SourcesError,
    describe_payload,
    load_sources,
    looks_like_html,
    looks_like_pdf,
    parse_source_line,
)

VALID_LINE = "https://example.com/a.pdf | is-kanunu | mevzuat"


def _write(tmp_path, *lines):
    path = tmp_path / "sources.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_parses_a_valid_line():
    source = parse_source_line(VALID_LINE)
    assert source == Source("https://example.com/a.pdf", "is-kanunu", "mevzuat")


def test_surrounding_whitespace_is_ignored():
    assert parse_source_line("  https://e.com/a.pdf|slug|mevzuat  ").slug == "slug"


@pytest.mark.parametrize("line", ["", "   ", "# yorum", "   # girintili yorum"])
def test_blank_and_comment_lines_are_skipped(line):
    assert parse_source_line(line) is None


def test_filename_comes_from_the_slug():
    assert parse_source_line(VALID_LINE).filename == "is-kanunu.pdf"


def test_only_mevzuat_is_redistributable():
    assert parse_source_line(VALID_LINE).is_redistributable is True
    assert parse_source_line("https://e.com/a.pdf | r | rehber").is_redistributable is False


@pytest.mark.parametrize(
    "line",
    [
        "https://example.com/a.pdf | is-kanunu",
        "https://example.com/a.pdf",
        "https://example.com/a.pdf | is-kanunu | mevzuat | fazladan",
    ],
)
def test_wrong_field_count_raises(line):
    with pytest.raises(SourcesError, match="expected 'URL"):
        parse_source_line(line, 3)


@pytest.mark.parametrize("url", ["ftp://example.com/a.pdf", "example.com/a.pdf", "/local/a.pdf"])
def test_non_http_url_raises(url):
    with pytest.raises(SourcesError, match="http"):
        parse_source_line(f"{url} | slug | mevzuat")


@pytest.mark.parametrize(
    "slug",
    [
        "../etc/passwd",
        "alt/dizin",
        "BuyukHarf",
        "boşluk lu",
        "alt_cizgi",
        "-bastan-tire",
        "sondan-tire-",
        "cift--tire",
        "",
        "türkçe",
    ],
)
def test_invalid_slug_raises(slug):
    # Slug doğrudan dosya adına gidiyor; dizin ayracı ve ".." kabul edilmemeli
    with pytest.raises(SourcesError, match="slug"):
        parse_source_line(f"https://example.com/a.pdf | {slug} | mevzuat")


def test_unknown_kind_raises():
    with pytest.raises(SourcesError, match="unknown kind"):
        parse_source_line("https://example.com/a.pdf | slug | kitap")


def test_error_message_names_the_line_number():
    with pytest.raises(SourcesError, match="line 7"):
        parse_source_line("bozuk satır", 7)


def test_load_sources_reads_all_entries(tmp_path):
    path = _write(
        tmp_path,
        "# başlık",
        "",
        VALID_LINE,
        "https://example.com/b.pdf | kvkk-kanunu | rehber",
    )
    sources = load_sources(path)
    assert [s.slug for s in sources] == ["is-kanunu", "kvkk-kanunu"]


def test_load_sources_rejects_duplicate_slugs(tmp_path):
    with pytest.raises(SourcesError, match="duplicate slug"):
        load_sources(_write(tmp_path, VALID_LINE, VALID_LINE))


def test_load_sources_reports_missing_file(tmp_path):
    with pytest.raises(SourcesError, match="not found"):
        load_sources(tmp_path / "yok.txt")


def test_load_sources_rejects_an_empty_list(tmp_path):
    with pytest.raises(SourcesError, match="no sources"):
        load_sources(_write(tmp_path, "# yalnızca yorum", ""))


def test_pdf_magic_bytes_are_recognised():
    assert looks_like_pdf(b"%PDF-1.7\n...") is True


def test_pdf_magic_is_accepted_after_leading_junk():
    assert looks_like_pdf(b"\r\n" + b"%PDF-1.4") is True


def test_html_error_page_is_not_a_pdf():
    # Site 404 yerine HTML hata sayfası döndürebiliyor
    assert looks_like_pdf(b"<!DOCTYPE html><html><body>Not found</body></html>") is False
    assert looks_like_html(b"<!DOCTYPE html><html></html>") is True


def test_magic_beyond_the_window_is_rejected():
    assert looks_like_pdf(b"x" * 2000 + b"%PDF") is False


def test_empty_response_is_not_a_pdf():
    assert looks_like_pdf(b"") is False
    assert describe_payload(b"") == "empty response"


def test_describe_payload_names_the_failure():
    assert "HTML" in describe_payload(b"<html><body>hata</body></html>")
    assert "not a PDF" in describe_payload(b"\x00\x01binary")
