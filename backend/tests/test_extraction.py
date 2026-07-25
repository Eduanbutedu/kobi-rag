from pathlib import Path

import pymupdf
import pytest

from rag.extraction import extract_text, strip_references


def _make_pdf(path: Path, text: str) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_extracts_text_from_pdf(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path, "Staj başvurusu için gerekli belgeler")
    result = extract_text(pdf_path)
    assert "Staj" in result


def test_extracts_text_from_txt(tmp_path):
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("KOBİ yönetmeliği", encoding="utf-8")
    assert extract_text(txt_path) == "KOBİ yönetmeliği"


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_text(tmp_path / "yok.pdf")


def test_unsupported_extension_raises(tmp_path):
    bad = tmp_path / "resim.png"
    bad.write_bytes(b"fake")
    with pytest.raises(ValueError):
        extract_text(bad)


def test_strip_references_removes_trailing_section():
    body = "Giriş bölümü. " * 50
    refs = "\nReferences\n[1] Smith, J. (2020). Some paper.\n[2] Doe, A. (2021). Another."
    assert strip_references(body + refs).rstrip() == body.rstrip()


def test_strip_references_ignores_early_mention():
    text = "İçindekiler: References sayfa 9. " + "Gövde metni burada devam ediyor. " * 50
    assert strip_references(text) == text