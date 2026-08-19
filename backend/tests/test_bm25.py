"""Keyword search over the FTS5 index, including its Turkish behaviour."""

import pytest

from rag.embedding import EMBEDDING_DIM
from rag.store import DocumentStore

# Gerçek korpustan alınmış cümleler
IZIN = (
    "İşveren, işçinin yıllık ücretli izin hakkını iş sözleşmesinin devamı süresince "
    "kullandırmak zorundadır ve bu haktan vazgeçilemez."
)
SIRKET = (
    "Şirket unvanı ticaret siciline tescil edilir; unvanın seçiminde kanunda "
    "öngörülen kurallara uyulması zorunludur."
)
SAGLIK = (
    "İşyerinde sağlık ve güvenlik önlemleri alınır, ağır ve tehlikeli işlerde "
    "çalışacakların sağlık raporu bulunması gerekir."
)


def _vec(first: float, second: float) -> list[float]:
    v = [0.0] * EMBEDDING_DIM
    v[0], v[1] = first, second
    return v


@pytest.fixture
def store(tmp_path):
    s = DocumentStore(tmp_path / "test.db")
    s.add_document("izin.pdf", [IZIN], [_vec(1, 0)])
    s.add_document("sirket.pdf", [SIRKET], [_vec(0, 1)])
    s.add_document("saglik.pdf", [SAGLIK], [_vec(1, 1)])
    yield s
    s.close()


def _sources(results):
    return [r["source"] for r in results]


# --- Temel davranış ---------------------------------------------------------


def test_finds_the_chunk_containing_the_term(store):
    assert _sources(store.bm25_search("izin", k=3))[0] == "izin.pdf"
    assert _sources(store.bm25_search("unvan tescil", k=3))[0] == "sirket.pdf"


def test_returns_the_same_shape_as_dense_search(store):
    [hit] = store.bm25_search("tescil", k=1)
    assert set(hit) == {"id", "text", "source", "score"}
    assert isinstance(hit["id"], int)


def test_ids_match_the_stored_chunks(store):
    [hit] = store.bm25_search("tescil", k=1)
    by_id = {c["id"]: c["text"] for c in store.all_chunks()}
    assert by_id[hit["id"]] == hit["text"]


def test_respects_k(store):
    assert len(store.bm25_search("ve", k=1)) <= 1
    assert len(store.bm25_search("zorundadır sağlık unvanı", k=2)) <= 2


def test_higher_score_is_better(store):
    results = store.bm25_search("sağlık güvenlik önlemleri", k=3)
    assert results == sorted(results, key=lambda r: -r["score"])


def test_no_match_returns_nothing(store):
    assert store.bm25_search("kuantum kriptografi", k=5) == []


# --- Türkçe karakterler -----------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("sağlık", "saglik.pdf"),
        ("ağır", "saglik.pdf"),
        ("şirket", "sirket.pdf"),
        ("işçinin", "izin.pdf"),
        ("yıllık", "izin.pdf"),
        ("ücretli", "izin.pdf"),
        ("güvenlik", "saglik.pdf"),
        ("öngörülen", "sirket.pdf"),
    ],
)
def test_turkish_letters_are_matched(store, query, expected):
    assert _sources(store.bm25_search(query, k=1)) == [expected]


def test_diacritics_are_not_stripped(store):
    # remove_diacritics=0 olmasaydı "saglik" da "sağlık"a eşleşirdi
    assert store.bm25_search("saglik", k=3) == []
    assert store.bm25_search("agir", k=3) == []


def test_ascii_case_is_folded(store):
    assert _sources(store.bm25_search("TESCIL", k=1)) == ["sirket.pdf"]
    assert _sources(store.bm25_search("tescil", k=1)) == ["sirket.pdf"]


def test_dotted_capital_i_is_a_known_limitation(store):
    """İ (U+0130) is never folded, so a word capitalised with it is missed.

    This bites ordinary sentence-initial words, not only headings: the text
    holds "İşveren, işçinin ...", and only the lowercase one is reachable.
    Fixing it would mean indexing a case-folded second copy of every chunk,
    and on the real corpus that changed BM25 hit rate by one question in
    twenty-five -- not worth the duplication.
    """
    assert store.bm25_search("işveren", k=3) == []
    assert _sources(store.bm25_search("işçinin", k=1)) == ["izin.pdf"]

    assert store.bm25_search("İZİN", k=3) == []
    assert _sources(store.bm25_search("izin", k=1)) == ["izin.pdf"]


# --- Sorgu ayrıştırma -------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        'izin "tırnaklı',
        "izin AND OR NOT",
        "izin*",
        "izin -hak",
        "izin (parantez)",
        "izin: iki nokta",
        "SGK'ya bildirim izin",
    ],
)
def test_fts5_syntax_in_a_question_does_not_break_the_search(store, query):
    # Ham sorgu MATCH'e verilseydi bunlar sözdizimi hatası olurdu
    results = store.bm25_search(query, k=3)
    assert "izin.pdf" in _sources(results)


def test_empty_or_tooshort_queries_return_nothing(store):
    for query in ("", "   ", "?", "a", "!!", "-"):
        assert store.bm25_search(query, k=3) == []


def test_a_long_question_matches_on_its_terms(store):
    question = "çalışanın yıllık ücretli izin hakkı ne zaman kullandırılır"
    assert _sources(store.bm25_search(question, k=1)) == ["izin.pdf"]


# --- İndeksin senkron kalması ------------------------------------------------


def test_new_documents_are_indexed_immediately(store):
    store.add_document("yeni.pdf", ["Kıdem tazminatı hesaplama esasları"], [_vec(0, 1)])
    assert _sources(store.bm25_search("kıdem tazminatı", k=1)) == ["yeni.pdf"]


def test_deleting_a_document_removes_it_from_the_index(store):
    assert store.bm25_search("tescil", k=3) != []
    store.delete_document("sirket.pdf")
    assert store.bm25_search("tescil", k=3) == []


def test_deleting_one_document_leaves_the_others_searchable(store):
    store.delete_document("sirket.pdf")
    assert _sources(store.bm25_search("izin", k=1)) == ["izin.pdf"]
    assert _sources(store.bm25_search("sağlık", k=1)) == ["saglik.pdf"]


def test_reindexing_the_same_source_finds_the_new_text(store):
    store.delete_document("izin.pdf")
    store.add_document("izin.pdf", ["Doğum izni ve süt izni hakları"], [_vec(1, 0)])
    assert store.bm25_search("ücretli", k=3) == []
    assert _sources(store.bm25_search("doğum", k=1)) == ["izin.pdf"]


def test_index_survives_reopening_the_database(tmp_path):
    path = tmp_path / "reopen.db"
    first = DocumentStore(path)
    first.add_document("a.pdf", [IZIN], [_vec(1, 0)])
    first.close()

    second = DocumentStore(path)
    try:
        assert _sources(second.bm25_search("izin", k=1)) == ["a.pdf"]
    finally:
        second.close()


def test_a_database_written_before_the_index_existed_is_backfilled(tmp_path):
    """A store created without FTS5 gets its index built on next open."""
    path = tmp_path / "legacy.db"
    legacy = DocumentStore(path)
    legacy.add_document("a.pdf", [IZIN], [_vec(1, 0)])
    # Eski sürümü taklit et: indeksi ve trigger'ları düşür, sürümü geri al
    legacy.db.execute("DROP TRIGGER chunks_fts_insert")
    legacy.db.execute("DROP TRIGGER chunks_fts_delete")
    legacy.db.execute("DROP TRIGGER chunks_fts_update")
    legacy.db.execute("DROP TABLE chunks_fts")
    legacy.db.execute("PRAGMA user_version = 0")
    legacy.db.commit()
    legacy.close()

    upgraded = DocumentStore(path)
    try:
        assert _sources(upgraded.bm25_search("izin", k=1)) == ["a.pdf"]
    finally:
        upgraded.close()


def test_chunks_table_is_untouched_by_the_index(store):
    columns = [row[1] for row in store.db.execute("PRAGMA table_info(chunks)")]
    assert columns == ["id", "source", "text"]
