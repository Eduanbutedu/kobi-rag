import json

import pytest

from eval.compare import build_markdown, comparison_warnings, load_result, run_name
from eval.run import aggregate, save_result


def _record(retrieved, relevant, rr, latency=10.0):
    return {
        "id": "q",
        "question": "S?",
        "relevant_chunk_ids": relevant,
        "retrieved_ids": retrieved,
        "rank": None,
        "reciprocal_rank": rr,
        "latency_ms": latency,
    }


def test_aggregate_averages_recall_across_questions():
    # Biri 1. sırada bulunuyor, diğeri hiç bulunmuyor -> recall@3 = 0.5
    records = [
        _record([7, 1, 2], [7], 1.0),
        _record([1, 2, 3], [9], 0.0),
    ]
    metrics = aggregate(records, k=3)
    assert metrics["recall@1"] == 0.5
    assert metrics["recall@3"] == 0.5
    assert metrics["mrr@3"] == 0.5


def test_aggregate_skips_cutoffs_above_k():
    metrics = aggregate([_record([7], [7], 1.0)], k=3)
    assert "recall@3" in metrics
    assert "recall@5" not in metrics
    assert "recall@10" not in metrics
    assert "hit@3" in metrics
    assert "hit@5" not in metrics


def test_aggregate_reports_hit_rate_at_every_cutoff():
    metrics = aggregate([_record([7, 1, 2], [7], 1.0)], k=10)
    assert [name for name in metrics if name.startswith("hit@")] == [
        "hit@1",
        "hit@3",
        "hit@5",
        "hit@10",
    ]


def test_hit_rate_is_not_dragged_down_by_multi_chunk_questions():
    # İki ilgili chunk'tan yalnızca biri 1. sırada: recall 0.5, hit rate 1.0
    records = [_record([21, 1, 2], [21, 22], 1.0)]
    metrics = aggregate(records, k=3)
    assert metrics["recall@1"] == 0.5
    assert metrics["hit@1"] == 1.0


def test_hit_rate_averages_across_questions():
    records = [
        _record([7, 1, 2], [7], 1.0),  # isabet
        _record([1, 2, 3], [9], 0.0),  # isabet yok
    ]
    metrics = aggregate(records, k=3)
    assert metrics["hit@1"] == 0.5
    assert metrics["hit@3"] == 0.5


def test_hit_rate_never_falls_below_recall_in_aggregate():
    records = [
        _record([21, 1, 2], [21, 22], 1.0),
        _record([1, 2, 3], [9], 0.0),
        _record([5, 6, 7], [7], 1 / 3),
    ]
    metrics = aggregate(records, k=10)
    for cutoff in (1, 3, 5, 10):
        assert metrics[f"hit@{cutoff}"] >= metrics[f"recall@{cutoff}"]


def test_aggregate_reports_partial_recall_for_multi_chunk_answers():
    # İki ilgili chunk'tan yalnızca biri ilk 3'te
    metrics = aggregate([_record([21, 1, 2], [21, 22], 1.0)], k=3)
    assert metrics["recall@3"] == 0.5


def test_aggregate_latency_summary():
    records = [_record([1], [1], 1.0, latency=10.0), _record([1], [1], 1.0, latency=20.0)]
    metrics = aggregate(records, k=1)
    assert metrics["latency_ms_avg"] == 15.0
    assert metrics["latency_ms_p95"] == pytest.approx(19.5)


def test_save_result_writes_readable_json(tmp_path):
    result = {
        "timestamp": "2026-08-18T11:39:15+00:00",
        "label": "baseline",
        "metrics": {"recall@1": 0.5},
        "question": "Yıllık izin kaç gün?",
    }
    path = save_result(result, tmp_path)
    assert path.name == "20260818T113915Z-baseline.json"
    assert json.loads(path.read_text(encoding="utf-8"))["question"] == "Yıllık izin kaç gün?"


BASELINE = {
    "label": "baseline",
    "dataset": "eval/dataset.jsonl",
    "k": 10,
    "question_count": 5,
    "chunk_count": 45,
    "metrics": {
        "recall@1": 0.3,
        "recall@5": 1.0,
        "mrr@10": 0.547,
        "latency_ms_avg": 21.21,
    },
}
CANDIDATE = {
    **BASELINE,
    "label": "hybrid",
    "metrics": {
        "recall@1": 0.6,
        "recall@5": 1.0,
        "mrr@10": 0.720,
        "latency_ms_avg": 30.00,
    },
}


def test_markdown_has_a_row_per_metric():
    md = build_markdown(BASELINE, CANDIDATE, "baseline", "hybrid")
    assert "| Metric | baseline | hybrid | Δ |" in md
    assert "| Recall@1 | 0.300 | 0.600 |" in md
    assert "| MRR@10 | 0.547 | 0.720 |" in md
    assert "| Latency avg (ms) | 21.21 | 30.00 |" in md


def test_markdown_marks_quality_gains_and_latency_costs():
    md = build_markdown(BASELINE, CANDIDATE, "baseline", "hybrid")
    assert "+0.300 ✅" in md  # recall arttı: iyi
    assert "+8.79 ❌" in md  # gecikme arttı: kötü


def test_markdown_reports_unchanged_metric_as_zero():
    md = build_markdown(BASELINE, CANDIDATE, "baseline", "hybrid")
    assert "| Recall@5 | 1.000 | 1.000 | 0 |" in md


def test_markdown_handles_metric_missing_from_one_run():
    candidate = {**CANDIDATE, "metrics": {**CANDIDATE["metrics"], "recall@3": 0.8}}
    md = build_markdown(BASELINE, candidate, "a", "b")
    assert "| Recall@3 | n/a | 0.800 | n/a |" in md


def test_markdown_orders_recall_then_mrr_then_latency():
    md = build_markdown(BASELINE, CANDIDATE, "a", "b")
    assert md.index("Recall@1") < md.index("MRR@10") < md.index("Latency avg")


def test_markdown_places_hit_rate_between_recall_and_mrr():
    with_hits = {
        **BASELINE,
        "metrics": {**BASELINE["metrics"], "hit@1": 0.5, "hit@5": 0.9},
    }
    md = build_markdown(with_hits, with_hits, "a", "b")

    assert "| Hit rate@1 | 0.500 | 0.500 |" in md
    assert md.index("Recall@1") < md.index("Hit rate@1") < md.index("MRR@10")
    assert md.index("Hit rate@5") < md.index("Latency avg")


def test_a_metric_with_no_label_still_appears():
    # Runner'a yeni bir ölçüm eklendiğinde tabloda kaybolmamalı
    extended = {**BASELINE, "metrics": {**BASELINE["metrics"], "ndcg@10": 0.42}}
    md = build_markdown(extended, extended, "a", "b")
    assert "ndcg@10" in md


def test_comparable_runs_produce_no_warning():
    assert comparison_warnings(BASELINE, CANDIDATE) == []
    assert "Warning" not in build_markdown(BASELINE, CANDIDATE, "a", "b")


def test_mismatched_runs_are_flagged():
    other = {**CANDIDATE, "question_count": 40, "k": 5}
    warnings = comparison_warnings(BASELINE, other)
    assert len(warnings) == 2
    assert "Warning" in build_markdown(BASELINE, other, "a", "b")


def test_run_name_prefers_label_then_filename():
    assert run_name({"label": "hybrid"}, "file") == "hybrid"
    assert run_name({"label": ""}, "file") == "file"
    assert run_name({}, "file") == "file"


def test_load_result_rejects_non_result_json(tmp_path):
    path = tmp_path / "x.json"
    path.write_text('{"hello": 1}', encoding="utf-8")
    with pytest.raises(SystemExit, match="metrics"):
        load_result(path)


def test_load_result_reports_missing_file(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        load_result(tmp_path / "yok.json")
