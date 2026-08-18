import pytest

from eval.metrics import (
    first_relevant_rank,
    mean,
    percentile,
    recall_at_k,
    reciprocal_rank_at_k,
)

# Tek ilgili chunk (7) 3. sırada: recall@1/@2 = 0, recall@3+ = 1, RR = 1/3
RANKED = [5, 9, 7, 2, 4]


def test_recall_misses_when_relevant_is_below_cutoff():
    assert recall_at_k(RANKED, [7], 1) == 0.0
    assert recall_at_k(RANKED, [7], 2) == 0.0


def test_recall_hits_once_cutoff_reaches_relevant():
    assert recall_at_k(RANKED, [7], 3) == 1.0
    assert recall_at_k(RANKED, [7], 5) == 1.0


def test_recall_is_fraction_of_relevant_found():
    # 3 ilgili chunk'tan (5, 7, 42) ilk 3'te 2 tanesi var
    assert recall_at_k(RANKED, [5, 7, 42], 3) == pytest.approx(2 / 3)


def test_recall_of_one_when_all_relevant_are_retrieved():
    assert recall_at_k(RANKED, [5, 9], 2) == 1.0


def test_recall_ignores_duplicate_retrievals():
    assert recall_at_k([5, 5, 5], [5, 7], 3) == 0.5


def test_recall_k_larger_than_result_list_is_safe():
    assert recall_at_k([5, 7], [7], 10) == 1.0


def test_recall_is_zero_for_empty_results():
    assert recall_at_k([], [7], 5) == 0.0


def test_first_relevant_rank_is_one_based():
    assert first_relevant_rank(RANKED, [7], 5) == 3
    assert first_relevant_rank(RANKED, [5], 5) == 1


def test_first_relevant_rank_reports_earliest_hit():
    assert first_relevant_rank(RANKED, [7, 9], 5) == 2


def test_first_relevant_rank_is_none_when_not_found():
    assert first_relevant_rank(RANKED, [99], 5) is None
    assert first_relevant_rank(RANKED, [7], 2) is None


def test_reciprocal_rank_is_inverse_of_rank():
    assert reciprocal_rank_at_k(RANKED, [7], 5) == pytest.approx(1 / 3)
    assert reciprocal_rank_at_k(RANKED, [5], 5) == 1.0
    assert reciprocal_rank_at_k(RANKED, [9], 5) == 0.5


def test_reciprocal_rank_is_zero_when_relevant_is_beyond_cutoff():
    assert reciprocal_rank_at_k(RANKED, [7], 2) == 0.0
    assert reciprocal_rank_at_k(RANKED, [99], 5) == 0.0


def test_invalid_k_raises():
    for k in (0, -1):
        with pytest.raises(ValueError):
            recall_at_k(RANKED, [7], k)
        with pytest.raises(ValueError):
            reciprocal_rank_at_k(RANKED, [7], k)


def test_empty_relevant_ids_raise():
    # Altın sette ilgili chunk'ı olmayan satır veri hatasıdır, sessizce 0 sayılmamalı
    with pytest.raises(ValueError):
        recall_at_k(RANKED, [], 5)
    with pytest.raises(ValueError):
        reciprocal_rank_at_k(RANKED, [], 5)


def test_mean_of_known_values():
    assert mean([1.0, 0.0, 0.5]) == pytest.approx(0.5)
    assert mean([]) == 0.0


def test_percentile_of_known_values():
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 0) == 10.0
    assert percentile(values, 100) == 40.0
    assert percentile(values, 50) == pytest.approx(25.0)


def test_percentile_edge_cases():
    assert percentile([], 95) == 0.0
    assert percentile([7.0], 95) == 7.0
    with pytest.raises(ValueError):
        percentile([1.0], 101)
