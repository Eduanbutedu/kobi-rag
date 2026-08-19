"""Reciprocal Rank Fusion, checked against ranks worked out by hand."""

import pytest

from rag.service import DENSE, HYBRID, RRF_K, reciprocal_rank_fusion, retrieve


def _chunk(chunk_id, score=0.0):
    return {"id": chunk_id, "text": f"metin {chunk_id}", "source": "a.pdf", "score": score}


def _ranking(*ids):
    return [_chunk(i) for i in ids]


def _ids(results):
    return [r["id"] for r in results]


def test_a_single_ranking_keeps_its_order():
    assert _ids(reciprocal_rank_fusion([_ranking(7, 2, 9)])) == [7, 2, 9]


# Skorlar gösterim için yuvarlanıyor; sıralama yuvarlanmamış değerlerle yapılıyor
ROUNDING = 1e-6


def test_scores_follow_the_standard_formula():
    [first, second] = reciprocal_rank_fusion([_ranking(7, 2)])
    assert first["score"] == pytest.approx(1 / (RRF_K + 1), abs=ROUNDING)
    assert second["score"] == pytest.approx(1 / (RRF_K + 2), abs=ROUNDING)


def test_a_chunk_in_both_rankings_sums_its_contributions():
    # 7: her iki listede de 1. sırada -> 2/(60+1)
    [top] = reciprocal_rank_fusion([_ranking(7), _ranking(7)])
    assert top["score"] == pytest.approx(2 / (RRF_K + 1), abs=ROUNDING)


def test_agreement_beats_a_single_first_place():
    # 2: iki listede de 2. sıra -> 2/62 = 0.03226
    # 7: bir listede 1. sıra    -> 1/61 = 0.01639
    fused = reciprocal_rank_fusion([_ranking(7, 2), _ranking(9, 2)])
    assert _ids(fused)[0] == 2
    assert fused[0]["score"] == pytest.approx(2 / (RRF_K + 2), abs=ROUNDING)


def test_one_high_placing_outweighs_two_middling_ones():
    # 1/(k+r) dışbükey: 1/61 + 1/63 = 0.032266 > 2/62 = 0.032258
    # Yani 1. + 3. sıra, iki kez 2. sıradan biraz daha ağır basar
    fused = reciprocal_rank_fusion([_ranking(1, 2, 3), _ranking(3, 2, 1)])
    assert _ids(fused) == [1, 3, 2]


def test_a_chunk_only_one_side_found_still_appears():
    fused = reciprocal_rank_fusion([_ranking(1, 2), _ranking(3, 4)])
    assert sorted(_ids(fused)) == [1, 2, 3, 4]


def test_ranking_is_by_fused_score_descending():
    fused = reciprocal_rank_fusion([_ranking(1, 2, 3), _ranking(3, 2, 1)])
    scores = [r["score"] for r in fused]
    assert scores == sorted(scores, reverse=True)


def test_ties_are_broken_by_chunk_id_so_the_order_is_stable():
    # İki chunk da yalnızca bir listede ve aynı sırada: skorlar tam eşit
    fused = reciprocal_rank_fusion([_ranking(9), _ranking(4)])
    assert _ids(fused) == [4, 9]
    assert fused[0]["score"] == pytest.approx(fused[1]["score"], abs=ROUNDING)
    assert _ids(reciprocal_rank_fusion([_ranking(4), _ranking(9)])) == [4, 9]


def test_deeper_ranks_contribute_less():
    fused = reciprocal_rank_fusion([_ranking(*range(1, 21))])
    assert fused[0]["score"] > fused[-1]["score"]
    assert fused[-1]["score"] == pytest.approx(1 / (RRF_K + 20), abs=ROUNDING)


def test_underlying_scores_are_ignored():
    # BM25 ve kosinüs skorları farklı ölçeklerde; yalnızca sıra sayılmalı
    dense = [_chunk(1, score=0.99), _chunk(2, score=0.98)]
    keyword = [_chunk(2, score=-12.5), _chunk(1, score=-31.0)]
    inflated = [_chunk(1, score=1000.0), _chunk(2, score=999.0)]

    assert _ids(reciprocal_rank_fusion([dense, keyword])) == _ids(
        reciprocal_rank_fusion([inflated, keyword])
    )


def test_the_fused_score_replaces_the_input_score():
    [top] = reciprocal_rank_fusion([[_chunk(7, score=0.99)]])
    assert top["score"] == pytest.approx(1 / (RRF_K + 1), abs=ROUNDING)


def test_chunk_fields_are_preserved():
    [top] = reciprocal_rank_fusion([[_chunk(7)]])
    assert top["text"] == "metin 7"
    assert top["source"] == "a.pdf"


def test_empty_input_is_handled():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_one_empty_ranking_does_not_disturb_the_other():
    assert _ids(reciprocal_rank_fusion([_ranking(5, 6), []])) == [5, 6]


def test_k_controls_how_much_top_ranks_dominate():
    rankings = [_ranking(1, 2), _ranking(2, 1)]
    # Küçük k ilk sırayı çok öne çıkarır, eşitliği bozar
    assert reciprocal_rank_fusion(rankings, k=1)[0]["score"] > reciprocal_rank_fusion(
        rankings, k=1000
    )[0]["score"]


# --- retrieve() kip seçimi --------------------------------------------------


class _FakeStore:
    """Records how it was called and returns fixed rankings."""

    def __init__(self):
        self.search_k = None
        self.bm25_k = None
        self.bm25_query = None

    def search(self, vector, k=3):
        self.search_k = k
        return [_chunk(i) for i in (1, 2, 3)]

    def bm25_search(self, query, k=3):
        self.bm25_k = k
        self.bm25_query = query
        return [_chunk(i) for i in (3, 4, 5)]


@pytest.fixture
def fake_store(monkeypatch):
    monkeypatch.setattr("rag.service.embed_texts", lambda texts: [[0.0, 1.0]])
    return _FakeStore()


def test_dense_mode_does_not_touch_the_keyword_index(fake_store):
    results = retrieve(fake_store, "soru", k=2, mode=DENSE)
    assert fake_store.bm25_k is None
    assert fake_store.search_k == 2
    assert _ids(results) == [1, 2, 3]


def test_hybrid_mode_merges_both_and_cuts_to_k(fake_store):
    results = retrieve(fake_store, "soru", k=3, mode=HYBRID)
    # 3 iki listede de var, en üstte olmalı
    assert _ids(results)[0] == 3
    assert len(results) == 3


def test_hybrid_pulls_more_candidates_than_it_returns(fake_store):
    retrieve(fake_store, "soru", k=3, mode=HYBRID)
    assert fake_store.search_k == 20
    assert fake_store.bm25_k == 20


def test_keyword_side_sees_the_raw_question(fake_store):
    retrieve(fake_store, "yıllık izin kaç gün", k=3, mode=HYBRID)
    assert fake_store.bm25_query == "yıllık izin kaç gün"


def test_hybrid_is_the_default(fake_store):
    assert _ids(retrieve(fake_store, "soru", k=3))[0] == 3


def test_an_unknown_mode_is_rejected(fake_store):
    with pytest.raises(ValueError, match="unknown retrieval mode"):
        retrieve(fake_store, "soru", k=3, mode="sparse")
