"""Cross-encoder reranking. The model itself is never loaded here."""

import pytest

from rag.rerank import order_by_score, rerank
from rag.service import HYBRID, RERANK_CANDIDATES, retrieve


def _chunk(chunk_id, score=0.0):
    return {"id": chunk_id, "text": f"metin {chunk_id}", "source": "a.pdf", "score": score}


def _ids(results):
    return [r["id"] for r in results]


# --- Saf sıralama mantığı ---------------------------------------------------


def test_orders_by_score_descending():
    candidates = [_chunk(1), _chunk(2), _chunk(3)]
    assert _ids(order_by_score(candidates, [0.2, 0.9, 0.5], 3)) == [2, 3, 1]


def test_keeps_only_top_k():
    candidates = [_chunk(1), _chunk(2), _chunk(3)]
    assert _ids(order_by_score(candidates, [0.2, 0.9, 0.5], 2)) == [2, 3]


def test_a_low_ranked_candidate_can_be_promoted_to_first():
    # Yeniden sıralamanın bütün amacı bu: 20. aday 1. sıraya çıkabilmeli
    candidates = [_chunk(i) for i in range(1, 21)]
    scores = [0.0] * 19 + [9.9]
    assert _ids(order_by_score(candidates, scores, 10))[0] == 20


def test_the_cross_encoder_score_replaces_the_incoming_score():
    [top] = order_by_score([_chunk(1, score=0.0164)], [3.5], 1)
    assert top["score"] == pytest.approx(3.5)


def test_chunk_fields_are_preserved():
    [top] = order_by_score([_chunk(7)], [1.0], 1)
    assert top["text"] == "metin 7"
    assert top["source"] == "a.pdf"
    assert top["id"] == 7


def test_negative_scores_are_ordered_correctly():
    candidates = [_chunk(1), _chunk(2), _chunk(3)]
    assert _ids(order_by_score(candidates, [-8.0, -1.5, -11.0], 3)) == [2, 1, 3]


def test_ties_break_on_chunk_id_so_the_order_is_stable():
    candidates = [_chunk(9), _chunk(4), _chunk(6)]
    assert _ids(order_by_score(candidates, [1.0, 1.0, 1.0], 3)) == [4, 6, 9]


def test_top_k_larger_than_the_shortlist_is_safe():
    assert len(order_by_score([_chunk(1), _chunk(2)], [1.0, 2.0], 10)) == 2


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError, match="top_k must be positive"):
        order_by_score([_chunk(1)], [1.0], 0)
    with pytest.raises(ValueError, match="same length"):
        order_by_score([_chunk(1), _chunk(2)], [1.0], 2)


# --- Model çağrısı taklit edilerek -----------------------------------------


class _FakeCrossEncoder:
    """Scores pairs from a lookup, and records what it was asked."""

    def __init__(self, scores_by_text):
        self.scores_by_text = scores_by_text
        self.pairs = None

    def predict(self, pairs):
        self.pairs = list(pairs)
        return [self.scores_by_text[text] for _, text in self.pairs]


@pytest.fixture
def fake_model(monkeypatch):
    def _install(scores_by_text):
        model = _FakeCrossEncoder(scores_by_text)
        monkeypatch.setattr("rag.rerank._get_model", lambda: model)
        return model

    return _install


def test_rerank_scores_every_candidate_against_the_query(fake_model):
    model = fake_model({"metin 1": 0.1, "metin 2": 0.7})
    rerank("yıllık izin kaç gün", [_chunk(1), _chunk(2)], 2)

    assert model.pairs == [
        ("yıllık izin kaç gün", "metin 1"),
        ("yıllık izin kaç gün", "metin 2"),
    ]


def test_rerank_returns_the_best_scoring_candidates(fake_model):
    fake_model({"metin 1": 0.1, "metin 2": 0.9, "metin 3": 0.5})
    results = rerank("soru", [_chunk(1), _chunk(2), _chunk(3)], 2)
    assert _ids(results) == [2, 3]


def test_rerank_on_an_empty_shortlist_does_not_call_the_model(monkeypatch):
    def _boom():
        raise AssertionError("model should not be loaded")

    monkeypatch.setattr("rag.rerank._get_model", _boom)
    assert rerank("soru", [], 10) == []


def test_the_model_is_loaded_at_most_once_per_process():
    from rag.rerank import _get_model

    # Yükleme saniyeler sürüyor, puanlama milisaniyeler; her sorguda
    # yeniden yüklenmemeli
    assert _get_model.cache_info().maxsize == 1


# --- retrieve() ile bağlantısı ----------------------------------------------


class _FakeStore:
    def __init__(self):
        self.search_k = None

    def search(self, vector, k=3):
        self.search_k = k
        return [_chunk(i) for i in range(1, 21)]

    def bm25_search(self, query, k=3):
        return [_chunk(i) for i in range(15, 35)]


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr("rag.service.embed_texts", lambda texts: [[0.0, 1.0]])
    return _FakeStore()


def test_retrieve_reranks_by_default(store, monkeypatch):
    """Reranking is the application's normal behaviour, not an opt-in."""
    called = []

    def _rerank(query, candidates, top_k):
        called.append(top_k)
        return list(candidates)[:top_k]

    monkeypatch.setattr("rag.service.rerank_candidates", _rerank)
    assert len(retrieve(store, "soru", k=10, mode=HYBRID)) == 10
    assert called == [10]


def test_reranking_can_be_switched_off(store, monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("rerank should not run when disabled")

    monkeypatch.setattr("rag.service.rerank_candidates", _boom)
    assert len(retrieve(store, "soru", k=10, mode=HYBRID, rerank=False)) == 10


def test_retrieve_reranks_the_shortlist_down_to_k(store, monkeypatch):
    captured = {}

    def _rerank(query, candidates, top_k):
        captured["count"] = len(candidates)
        captured["top_k"] = top_k
        return list(candidates)[:top_k]

    monkeypatch.setattr("rag.service.rerank_candidates", _rerank)
    results = retrieve(store, "soru", k=10, mode=HYBRID, rerank=True)

    assert captured["count"] == RERANK_CANDIDATES
    assert captured["top_k"] == 10
    assert len(results) == 10


def test_reranking_widens_the_dense_shortlist_too(store, monkeypatch):
    monkeypatch.setattr(
        "rag.service.rerank_candidates", lambda q, c, k: list(c)[:k]
    )
    retrieve(store, "soru", k=3, mode="dense", rerank=True)
    assert store.search_k == RERANK_CANDIDATES


def test_the_query_reaches_the_reranker_unchanged(store, monkeypatch):
    captured = {}

    def _rerank(query, candidates, top_k):
        captured["query"] = query
        return list(candidates)[:top_k]

    monkeypatch.setattr("rag.service.rerank_candidates", _rerank)
    retrieve(store, "işten çıkarma bildirim süresi", k=5, mode=HYBRID, rerank=True)
    assert captured["query"] == "işten çıkarma bildirim süresi"
