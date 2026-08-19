"""Turkish stopword filtering on the BM25 query side."""

import pytest

from rag.stopwords_tr import STOPWORDS_TR, is_stopword
from rag.store import DocumentStore
from tests.test_bm25 import IZIN, SAGLIK, SIRKET, _vec

# --- Liste içeriği ----------------------------------------------------------


def test_list_is_a_reasonable_size():
    assert 100 <= len(STOPWORDS_TR) <= 150


@pytest.mark.parametrize(
    "word", ["ve", "ile", "için", "içinde", "bir", "mi", "olan", "olarak", "kadar", "ne"]
)
def test_function_words_are_filtered(word):
    assert is_stopword(word)


@pytest.mark.parametrize(
    "word",
    [
        # Anlamı tersine çeviren kelimeler listede olmamalı
        "değil",
        "yok",
        "olmaz",
        "olmayan",
        # Mevzuatta sayılar anlam taşıyor
        "iki",
        "üç",
        "beş",
        "on",
        "otuz",
        # Sorunun asıl konusu olabilen sıradan adlar
        "süre",
        "gün",
        "yıl",
        "hak",
        "ceza",
        "izin",
    ],
)
def test_meaningful_words_are_kept(word):
    assert not is_stopword(word)


@pytest.mark.parametrize("word", ["İÇİNDE", "İçinde", "IÇINDE", "içinde", "İLE", "Ve"])
def test_case_is_ignored_including_dotted_capital_i(word):
    # "İÇİNDE".lower() birleşik noktalı bir i üretir; düz lower() yetmez
    assert is_stopword(word)


def test_stopwords_are_stored_lowercase():
    assert all(word == word.lower() for word in STOPWORDS_TR)


# --- Sorgu oluşturma --------------------------------------------------------


def _expression(query):
    return DocumentStore._match_expression(query)


def test_function_words_are_dropped_from_the_expression():
    expression = _expression("çalışanların özlük dosyalarını ne kadar süre saklamalıyım")
    assert '"ne"' not in expression
    assert '"kadar"' not in expression
    assert '"özlük"' in expression
    assert '"süre"' in expression  # içerik kelimesi korunur


def test_content_terms_keep_their_order():
    assert _expression("internetten satışta müşteri kaç gün içinde cayabilir") == (
        '"internetten" OR "satışta" OR "müşteri" OR "kaç" OR "gün" OR "cayabilir"'
    )


def test_a_question_of_only_function_words_falls_back_to_them():
    # Boş MATCH ifadesi göndermektense zayıf bir sorgu yollamak yeğdir
    expression = _expression("bunlar ne olarak ve için")
    assert expression
    assert '"olarak"' in expression


def test_fallback_keeps_the_search_working(tmp_path):
    store = DocumentStore(tmp_path / "fallback.db")
    try:
        store.add_document("a.pdf", ["Bunlar ne olarak kabul edilir"], [_vec(1, 0)])
        assert store.bm25_search("bunlar ne olarak", k=3) != []
    finally:
        store.close()


def test_an_empty_query_is_still_empty():
    assert _expression("") == ""
    assert _expression("?! ,") == ""


# --- Aramaya etkisi ---------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    s = DocumentStore(tmp_path / "stop.db")
    s.add_document("izin.pdf", [IZIN], [_vec(1, 0)])
    s.add_document("sirket.pdf", [SIRKET], [_vec(0, 1)])
    s.add_document("saglik.pdf", [SAGLIK], [_vec(1, 1)])
    yield s
    s.close()


def test_a_chunk_no_longer_qualifies_on_a_function_word_alone(store):
    # "ve" üç metinde de geçiyor; yalnızca onun yüzünden aday olmamalılar
    results = store.bm25_search("ve tescil", k=5)
    assert [r["source"] for r in results] == ["sirket.pdf"]


def test_content_words_still_find_their_chunk(store):
    assert [r["source"] for r in store.bm25_search("tescil", k=1)] == ["sirket.pdf"]
    assert [r["source"] for r in store.bm25_search("sağlık raporu", k=1)] == ["saglik.pdf"]


def test_a_natural_question_reaches_the_right_chunk(store):
    question = "şirket unvanı ne zaman ve nasıl tescil edilir"
    assert [r["source"] for r in store.bm25_search(question, k=1)] == ["sirket.pdf"]
