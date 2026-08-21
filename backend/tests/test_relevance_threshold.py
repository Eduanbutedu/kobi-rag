"""The relevance threshold: what retrieve() does with a question the corpus
cannot answer.

Everything expensive is stubbed. Embedding, the store and the cross-encoder
are all replaced, so these tests measure the filtering rule itself rather
than the models behind it.
"""

import pytest

from rag import service
from rag.service import RELEVANCE_THRESHOLD, retrieve


class FakeStore:
    """Returns the same candidates for any query, dense or keyword."""

    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query_vector, k=3):
        return self.chunks[:k]

    def bm25_search(self, query, k=3):
        return self.chunks[:k]


CHUNKS = [
    {"id": 1, "text": "Yıllık izin en az on dört gündür.", "source": "is.pdf"},
    {"id": 2, "text": "İzin ücreti peşin ödenir.", "source": "is.pdf"},
    {"id": 3, "text": "Şirket merkezi Ankara'dadır.", "source": "esas.pdf"},
]


@pytest.fixture
def scored(monkeypatch):
    """Let a test dictate the cross-encoder score of each chunk by id."""
    monkeypatch.setattr(service, "embed_texts", lambda texts: [[0.0, 0.0] for _ in texts])

    def use(scores):
        def fake_rerank(query, candidates, top_k):
            ranked = sorted(candidates, key=lambda c: -scores[c["id"]])
            return [{**c, "score": scores[c["id"]]} for c in ranked[:top_k]]

        monkeypatch.setattr(service, "rerank_candidates", fake_rerank)
        return FakeStore(CHUNKS)

    return use


def test_chunks_below_the_threshold_are_dropped(scored):
    store = scored({1: 4.0, 2: -3.0, 3: -8.0})

    hits = retrieve(store, "yıllık izin kaç gün", k=3)

    assert [hit["id"] for hit in hits] == [1]


def test_a_question_nothing_answers_comes_back_empty(scored):
    # "merhaba" gibi bir girdide her chunk düşük skor alır
    store = scored({1: -6.0, 2: -7.0, 3: -9.0})

    assert retrieve(store, "merhaba", k=3) == []


def test_a_chunk_exactly_on_the_threshold_survives(scored):
    store = scored({1: RELEVANCE_THRESHOLD, 2: -9.0, 3: -9.0})

    assert [hit["id"] for hit in retrieve(store, "soru", k=3)] == [1]


def test_a_chunk_just_under_the_threshold_does_not(scored):
    store = scored({1: RELEVANCE_THRESHOLD - 0.001, 2: -9.0, 3: -9.0})

    assert retrieve(store, "soru", k=3) == []


def test_relevant_chunks_are_untouched(scored):
    store = scored({1: 5.0, 2: 2.0, 3: 0.5})

    assert [hit["id"] for hit in retrieve(store, "soru", k=3)] == [1, 2, 3]


def test_the_threshold_can_be_turned_off_for_measurement(scored):
    store = scored({1: -6.0, 2: -7.0, 3: -9.0})

    hits = retrieve(store, "merhaba", k=3, min_score=None)

    assert [hit["id"] for hit in hits] == [1, 2, 3]


def test_an_explicit_threshold_overrides_the_default(scored):
    store = scored({1: 4.0, 2: 1.0, 3: -1.0})

    assert [hit["id"] for hit in retrieve(store, "soru", k=3, min_score=0.0)] == [1, 2]


def test_without_reranking_the_threshold_is_not_applied(scored):
    # Yeniden sıralama kapalıyken skor bir RRF değeri; -2,5 orada anlamsız
    store = scored({1: 4.0, 2: 1.0, 3: -1.0})

    hits = retrieve(store, "soru", k=3, rerank=False)

    assert len(hits) == 3
    assert all(hit["score"] > 0 for hit in hits)


def test_the_threshold_matches_the_calibrated_value():
    # Kalibrasyon eval/README.md'de; değişirse orası da güncellenmeli
    assert RELEVANCE_THRESHOLD == -2.5
