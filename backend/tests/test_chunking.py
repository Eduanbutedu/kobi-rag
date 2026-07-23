import pytest

from rag.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text_returns_single_chunk():
    text = "KOBİ'ler için doküman asistanı."
    assert chunk_text(text, chunk_size=500) == [text]


def test_long_text_is_split_with_overlap():
    text = "A" * 1200
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) == 3
    assert all(len(c) <= 500 for c in chunks)


def test_overlap_preserves_boundary_content():
    text = "x" * 490 + "önemli cümle" + "y" * 490
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert any("önemli cümle" in c for c in chunks)


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=100, overlap=100)