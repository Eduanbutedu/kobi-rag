"""Reciprocal Rank Fusion, checked against ranks worked out by hand."""

import pytest

from rag.service import (
    BM25_WEIGHT,
    DENSE,
    DENSE_WEIGHT,
    HYBRID,
    RRF_K,
    reciprocal_rank_fusion,
    retrieve,
)


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


# --- Ağırlıklandırma --------------------------------------------------------


def test_weights_default_to_one_each():
    assert reciprocal_rank_fusion([_ranking(1, 2)]) == reciprocal_rank_fusion(
        [_ranking(1, 2)], [1.0]
    )


def test_a_weight_scales_that_ranking_contribution():
    [top] = reciprocal_rank_fusion([_ranking(7)], [0.5])
    assert top["score"] == pytest.approx(0.5 / (RRF_K + 1), abs=ROUNDING)


def test_a_zero_weight_ignores_a_ranking_but_keeps_its_chunks():
    fused = reciprocal_rank_fusion([_ranking(1), _ranking(2)], [1.0, 0.0])
    assert _ids(fused) == [1, 2]
    assert fused[1]["score"] == 0.0


def test_mismatched_weight_count_is_rejected():
    with pytest.raises(ValueError, match="one entry per ranking"):
        reciprocal_rank_fusion([_ranking(1), _ranking(2)], [1.0])


def test_weighting_rescues_a_result_only_the_dense_search_found():
    """The exact man016 failure: dense rank 6, absent from BM25's top 20.

    Unweighted, the chunk scores 1/66 = 0.01515 while BM25's ranks 1-5 score
    1/61 to 1/65, so five keyword-only chunks plus dense's own top five push
    it to rank 11 -- out of the top 10. Weighting the keyword side at 0.5
    caps its best contribution at 0.5/61 = 0.00820, below the chunk's own
    0.01515, so relevance decides the order instead of arithmetic.
    """
    wanted = 3808
    dense = _ranking(101, 102, 103, 104, 105, wanted, 107, 108, 109, 110)
    keyword = _ranking(*range(201, 221))

    unweighted = reciprocal_rank_fusion([dense, keyword])
    weighted = reciprocal_rank_fusion([dense, keyword], [1.0, 0.5])

    # Dense'in ilk beşi ve BM25'in ilk altısı önüne geçiyor: 12. sıra.
    # Gerçek man016 çalışmasında da birleşik sıra tam olarak 12 çıkmıştı.
    assert _ids(unweighted).index(wanted) + 1 == 12
    assert _ids(weighted).index(wanted) + 1 == 6

    # Sayısal olarak: 1.0/66 > 0.5/61
    dense_only = 1.0 / (RRF_K + 6)
    best_keyword_only = 0.5 / (RRF_K + 1)
    assert dense_only > best_keyword_only
    assert dense_only == pytest.approx(0.015152, abs=ROUNDING)
    assert best_keyword_only == pytest.approx(0.008197, abs=ROUNDING)


def test_the_unweighted_case_is_genuinely_broken():
    # Ağırlıksız hâlde BM25'in ilk beşi dense'in 6. sırasını gerçekten geçiyor
    assert 1.0 / (RRF_K + 5) > 1.0 / (RRF_K + 6)


def test_a_chunk_both_searches_find_still_outranks_a_dense_only_one():
    # Ağırlıklandırma, uzlaşmayı ödüllendirmeyi bozmamalı
    dense = _ranking(1, 2, 3, 4, 5, 6)
    keyword = _ranking(6, 7, 8)
    fused = reciprocal_rank_fusion([dense, keyword], [1.0, 0.5])
    # 6: dense 6. + bm25 1. -> 1/66 + 0.5/61 = 0.02335, 1'in 1/61=0.01639'undan büyük
    assert _ids(fused)[0] == 6


def test_weights_do_not_change_a_single_ranking_order():
    assert _ids(reciprocal_rank_fusion([_ranking(5, 6, 7)], [0.3])) == [5, 6, 7]


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
    # k'ya kırpma retrieve içinde de yapılıyor; store'un k'yı dinlemesine
    # güvenilmiyor
    assert _ids(results) == [1, 2]


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


def test_retrieve_passes_the_measured_default_weights(fake_store, monkeypatch):
    """The weights are a measured choice, not an arbitrary one.

    0.5, 0.87, 0.90, 0.95 and 1.0 were run against the golden set; 0.95 won.
    Changing these should mean re-running that sweep, so the values are
    pinned here and explained in eval/README.md.
    """
    captured = {}

    def _fusion(rankings, weights=None, k=RRF_K):
        captured["weights"] = weights
        return rankings[0]

    monkeypatch.setattr("rag.service.reciprocal_rank_fusion", _fusion)
    retrieve(fake_store, "soru", k=3)

    assert captured["weights"] == [DENSE_WEIGHT, BM25_WEIGHT]
    assert (DENSE_WEIGHT, BM25_WEIGHT) == (1.0, 0.95)


def test_the_weights_are_overridable_per_call(fake_store, monkeypatch):
    captured = {}

    def _fusion(rankings, weights=None, k=RRF_K):
        captured["weights"] = weights
        return rankings[0]

    monkeypatch.setattr("rag.service.reciprocal_rank_fusion", _fusion)
    retrieve(fake_store, "soru", k=3, dense_weight=1.0, bm25_weight=0.5)

    assert captured["weights"] == [1.0, 0.5]


def test_an_unknown_mode_is_rejected(fake_store):
    with pytest.raises(ValueError, match="unknown retrieval mode"):
        retrieve(fake_store, "soru", k=3, mode="sparse")
