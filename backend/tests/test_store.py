import pytest

from rag.embedding import EMBEDDING_DIM
from rag.store import DocumentStore


def _vec(first: float, second: float) -> list[float]:
    """Build a simple 384-dim unit vector controlled by two components."""
    v = [0.0] * EMBEDDING_DIM
    v[0], v[1] = first, second
    return v


@pytest.fixture
def store(tmp_path):
    s = DocumentStore(tmp_path / "test.db")
    yield s
    s.close()


def test_add_document_returns_chunk_count(store):
    n = store.add_document("a.pdf", ["bir", "iki"], [_vec(1, 0), _vec(0, 1)])
    assert n == 2


def test_mismatched_lengths_raise(store):
    with pytest.raises(ValueError):
        store.add_document("a.pdf", ["bir"], [])


def test_search_returns_most_similar_first(store):
    store.add_document("izin.pdf", ["izin metni"], [_vec(1.0, 0.0)])
    store.add_document("futbol.pdf", ["futbol metni"], [_vec(0.0, 1.0)])

    results = store.search(_vec(0.9, 0.1), k=2)

    assert results[0]["source"] == "izin.pdf"
    assert results[0]["score"] > results[1]["score"]


def test_search_respects_k(store):
    store.add_document("a.pdf", ["x", "y", "z"], [_vec(1, 0), _vec(0, 1), _vec(1, 1)])
    assert len(store.search(_vec(1, 0), k=2)) == 2

def test_search_returns_chunk_ids(store):
    store.add_document("a.pdf", ["x", "y"], [_vec(1, 0), _vec(0, 1)])
    results = store.search(_vec(1, 0), k=2)
    ids = [r["id"] for r in results]
    assert sorted(ids) == [1, 2]


def test_search_ids_match_stored_text(store):
    store.add_document("izin.pdf", ["izin metni"], [_vec(1.0, 0.0)])
    store.add_document("futbol.pdf", ["futbol metni"], [_vec(0.0, 1.0)])

    [top] = store.search(_vec(1.0, 0.0), k=1)
    by_id = {c["id"]: c["text"] for c in store.all_chunks()}

    assert by_id[top["id"]] == top["text"] == "izin metni"


def test_all_chunks_returns_every_chunk_with_source(store):
    store.add_document("a.pdf", ["x", "y"], [_vec(1, 0), _vec(0, 1)])
    store.add_document("b.pdf", ["z"], [_vec(1, 1)])

    chunks = store.all_chunks()

    assert [c["id"] for c in chunks] == [1, 2, 3]
    assert [c["source"] for c in chunks] == ["a.pdf", "a.pdf", "b.pdf"]
    assert [c["text"] for c in chunks] == ["x", "y", "z"]


def test_all_chunks_is_empty_for_new_store(store):
    assert store.all_chunks() == []


def test_list_documents_groups_by_source(store):
    store.add_document("a.pdf", ["x", "y"], [_vec(1, 0), _vec(0, 1)])
    store.add_document("b.pdf", ["z"], [_vec(1, 1)])
    docs = store.list_documents()
    assert {"source": "a.pdf", "chunks": 2} in docs
    assert {"source": "b.pdf", "chunks": 1} in docs


def test_delete_document_removes_chunks(store):
    store.add_document("a.pdf", ["x"], [_vec(1, 0)])
    assert store.delete_document("a.pdf") == 1
    assert store.list_documents() == []
    assert store.search(_vec(1, 0), k=1) == []