"""Compare two evaluation runs as a markdown table ready to paste into the README.

    python -m eval.compare eval/results/<baseline>.json eval/results/<candidate>.json
"""

import argparse
import json
from pathlib import Path

# Ölçüm adı -> (görünen ad, yüksek olan iyi mi?)
METRIC_LABELS: dict[str, tuple[str, bool]] = {
    "recall@1": ("Recall@1", True),
    "recall@3": ("Recall@3", True),
    "recall@5": ("Recall@5", True),
    "recall@10": ("Recall@10", True),
    "latency_ms_avg": ("Latency avg (ms)", False),
    "latency_ms_p95": ("Latency p95 (ms)", False),
}


def _label_for(metric: str) -> tuple[str, bool]:
    if metric.startswith("mrr@"):
        return f"MRR@{metric.split('@')[1]}", True
    return METRIC_LABELS.get(metric, (metric, True))


def _is_latency(metric: str) -> bool:
    return metric.startswith("latency")


def run_name(result: dict, fallback: str) -> str:
    """Prefer the run's --label, fall back to the file name."""
    return result.get("label") or fallback


def _format_value(metric: str, value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}" if _is_latency(metric) else f"{value:.3f}"


def _format_delta(metric: str, baseline: float | None, candidate: float | None) -> str:
    if baseline is None or candidate is None:
        return "n/a"
    delta = candidate - baseline
    text = f"{delta:+.2f}" if _is_latency(metric) else f"{delta:+.3f}"
    if abs(delta) < 1e-9:
        return "0"
    _, higher_is_better = _label_for(metric)
    improved = delta > 0 if higher_is_better else delta < 0
    return f"{text} {'✅' if improved else '❌'}"


def _ordered_metrics(baseline: dict, candidate: dict) -> list[str]:
    """Metric order: known order first, then any extras, without duplicates."""
    seen = list(baseline.get("metrics", {})) + list(candidate.get("metrics", {}))
    known = [m for m in METRIC_LABELS if m in seen]
    extra = [m for m in dict.fromkeys(seen) if m not in METRIC_LABELS]
    mrr = [m for m in extra if m.startswith("mrr@")]
    other = [m for m in extra if not m.startswith("mrr@")]
    # MRR, recall'lardan hemen sonra ve latency'den önce gelsin
    recalls = [m for m in known if m.startswith("recall")]
    latency = [m for m in known if _is_latency(m)]
    return recalls + mrr + latency + other


def comparison_warnings(baseline: dict, candidate: dict) -> list[str]:
    """Flag differences that make the two runs not directly comparable."""
    warnings = []
    for field, description in (
        ("dataset", "different golden sets"),
        ("question_count", "different question counts"),
        ("k", "different k"),
        ("chunk_count", "different corpus sizes"),
    ):
        base, cand = baseline.get(field), candidate.get(field)
        if base != cand:
            warnings.append(f"{description}: `{base}` vs `{cand}`")
    return warnings


def build_markdown(baseline: dict, candidate: dict, base_name: str, cand_name: str) -> str:
    """Render the side-by-side comparison table."""
    base_metrics = baseline.get("metrics", {})
    cand_metrics = candidate.get("metrics", {})

    lines = [
        f"| Metric | {base_name} | {cand_name} | Δ |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in _ordered_metrics(baseline, candidate):
        label, _ = _label_for(metric)
        base_value = base_metrics.get(metric)
        cand_value = cand_metrics.get(metric)
        lines.append(
            f"| {label} | {_format_value(metric, base_value)} "
            f"| {_format_value(metric, cand_value)} "
            f"| {_format_delta(metric, base_value, cand_value)} |"
        )

    dataset = baseline.get("dataset", "?")
    questions = baseline.get("question_count", "?")
    lines += [
        "",
        f"_{questions} questions from `{dataset}`, k={baseline.get('k', '?')}. "
        "Higher is better except latency._",
    ]

    warnings = comparison_warnings(baseline, candidate)
    if warnings:
        lines += ["", "> **Warning:** these runs are not directly comparable —"]
        lines += [f"> - {w}" for w in warnings]
    return "\n".join(lines) + "\n"


def load_result(path: Path) -> dict:
    """Read one result JSON produced by eval.run."""
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"result file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc.msg}") from exc
    if "metrics" not in result:
        raise SystemExit(f"{path} has no 'metrics' key -- is it an eval.run result?")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two evaluation runs as markdown.")
    parser.add_argument("baseline", type=Path, help="baseline result JSON")
    parser.add_argument("candidate", type=Path, help="candidate result JSON")
    parser.add_argument("--names", nargs=2, metavar=("BASE", "CAND"), help="column headers")
    parser.add_argument("--out", type=Path, help="also write the markdown to this file")
    args = parser.parse_args()

    baseline = load_result(args.baseline)
    candidate = load_result(args.candidate)

    if args.names:
        base_name, cand_name = args.names
    else:
        base_name = run_name(baseline, args.baseline.stem)
        cand_name = run_name(candidate, args.candidate.stem)

    markdown = build_markdown(baseline, candidate, base_name, cand_name)
    print(markdown)
    if args.out:
        args.out.write_text(markdown, encoding="utf-8")
        print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
