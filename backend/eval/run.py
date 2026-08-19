"""Run the retrieval golden set and report Recall@k, hit rate, MRR and latency.

Calls rag.service.retrieve() directly, so the measured path is exactly the
one the API serves -- no HTTP layer in between.

    python -m eval.run --k 10 --dataset eval/dataset.jsonl
"""

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

from eval.dataset import EvalCase, load_dataset
from eval.metrics import (
    first_relevant_rank,
    hit_rate_at_k,
    mean,
    percentile,
    recall_at_k,
    reciprocal_rank_at_k,
)
from rag.service import HYBRID, RETRIEVAL_MODES, retrieve
from rag.store import DocumentStore

DEFAULT_DATASET = Path("eval/dataset.jsonl")
DEFAULT_DB = Path("data/kobi_rag.db")
DEFAULT_RESULTS_DIR = Path("eval/results")
RECALL_CUTOFFS = (1, 3, 5, 10)


def evaluate_case(store: DocumentStore, case: EvalCase, k: int, mode: str = HYBRID) -> dict:
    """Retrieve for one question and score it. Returns a per-question record."""
    started = time.perf_counter()
    hits = retrieve(store, case.question, k=k, mode=mode)
    latency_ms = (time.perf_counter() - started) * 1000

    retrieved_ids = [hit["id"] for hit in hits]
    reciprocal = reciprocal_rank_at_k(retrieved_ids, case.relevant_chunk_ids, k)
    return {
        "id": case.id,
        "question": case.question,
        "relevant_chunk_ids": case.relevant_chunk_ids,
        "retrieved_ids": retrieved_ids,
        "rank": first_relevant_rank(retrieved_ids, case.relevant_chunk_ids, k),
        "reciprocal_rank": round(reciprocal, 4),
        "latency_ms": round(latency_ms, 2),
        "top_source": hits[0]["source"] if hits else None,
    }


def aggregate(records: list[dict], k: int) -> dict:
    """Average the per-question records into the reported metric set."""
    metrics: dict[str, float] = {}
    cutoffs = [c for c in RECALL_CUTOFFS if c <= k]
    for cutoff in cutoffs:
        recalls = [
            recall_at_k(r["retrieved_ids"], r["relevant_chunk_ids"], cutoff) for r in records
        ]
        metrics[f"recall@{cutoff}"] = round(mean(recalls), 4)

    # Hit rate, recall'ın aksine kaç chunk işaretlendiğine bağlı değil
    for cutoff in cutoffs:
        hits = [
            hit_rate_at_k(r["retrieved_ids"], r["relevant_chunk_ids"], cutoff) for r in records
        ]
        metrics[f"hit@{cutoff}"] = round(mean(hits), 4)

    metrics[f"mrr@{k}"] = round(mean([r["reciprocal_rank"] for r in records]), 4)

    latencies = [r["latency_ms"] for r in records]
    metrics["latency_ms_avg"] = round(mean(latencies), 2)
    metrics["latency_ms_p95"] = round(percentile(latencies, 95), 2)
    return metrics


def _warm_up(store: DocumentStore, mode: str) -> None:
    """Pay the embedding-model load cost before timing anything."""
    retrieve(store, "isinma sorgusu", k=1, mode=mode)


def _missing_chunk_ids(store: DocumentStore, cases: list[EvalCase]) -> list[int]:
    """Return golden-set chunk ids that no longer exist in the store."""
    known = {chunk["id"] for chunk in store.all_chunks()}
    referenced = {cid for case in cases for cid in case.relevant_chunk_ids}
    return sorted(referenced - known)


def format_report(result: dict) -> str:
    """Render the run as an aligned plain-text report."""
    lines = ["", "Retrieval Evaluation", "=" * 46]
    for label, value in (
        ("dataset", f"{result['dataset']} ({result['question_count']} questions)"),
        ("db", f"{result['db']} ({result['chunk_count']} chunks)"),
        ("k", result["k"]),
        ("mode", result.get("mode", "dense")),
        ("run at", result["timestamp"]),
    ):
        lines.append(f"{label:<10}{value}")

    lines += ["", f"{'Metric':<16}{'Value':>10}", "-" * 26]
    for name, value in result["metrics"].items():
        rendered = f"{value:.2f} ms" if name.startswith("latency") else f"{value:.3f}"
        lines.append(f"{name:<16}{rendered:>10}")

    misses = [r for r in result["per_question"] if r["rank"] is None]
    if misses:
        lines += [
            "",
            f"Missed ({len(misses)}/{result['question_count']}) "
            f"-- no relevant chunk in top {result['k']}:",
        ]
        lines += [f"  {r['id']}  {r['question'][:60]}" for r in misses]
    return "\n".join(lines) + "\n"


def format_per_question(records: list[dict]) -> str:
    """Render one row per question with its rank and latency."""
    header = f"{'id':<8}{'rank':>6}{'RR':>8}{'ms':>9}  question"
    lines = ["", header, "-" * (len(header) + 12)]
    for r in records:
        rank = "-" if r["rank"] is None else str(r["rank"])
        lines.append(
            f"{r['id']:<8}{rank:>6}{r['reciprocal_rank']:>8.3f}"
            f"{r['latency_ms']:>9.1f}  {r['question'][:52]}"
        )
    return "\n".join(lines) + "\n"


def run(
    dataset_path: Path, db_path: Path, k: int, label: str = "", mode: str = HYBRID
) -> dict:
    """Evaluate the whole golden set and return the result document."""
    cases = load_dataset(dataset_path)
    store = DocumentStore(db_path)
    try:
        chunk_count = len(store.all_chunks())
        if chunk_count == 0:
            raise SystemExit(
                f"No chunks in {db_path}. Upload documents before running the evaluation."
            )

        missing = _missing_chunk_ids(store, cases)
        if missing:
            shown = f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
            print(
                f"WARNING: {len(missing)} golden-set chunk id(s) are not in the store: {shown}\n"
                "         Chunk ids change when a document is re-ingested; regenerate the\n"
                "         golden set if this is unexpected.\n"
            )

        _warm_up(store, mode)
        records = [evaluate_case(store, case, k, mode) for case in cases]
    finally:
        store.close()

    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "label": label,
        "dataset": str(dataset_path),
        "db": str(db_path),
        "k": k,
        "mode": mode,
        "question_count": len(records),
        "chunk_count": chunk_count,
        "missing_chunk_ids": missing,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "metrics": aggregate(records, k),
        "per_question": records,
    }


def save_result(result: dict, results_dir: Path) -> Path:
    """Write the result document to <results_dir>/<timestamp>[-label].json."""
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = result["timestamp"].replace(":", "").replace("-", "").replace("+0000", "Z")
    suffix = f"-{result['label']}" if result["label"] else ""
    path = results_dir / f"{stamp}{suffix}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval against a golden set.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="golden set .jsonl")
    parser.add_argument("--k", type=int, default=10, help="number of chunks to retrieve")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="vector store path")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="results folder")
    parser.add_argument("--label", default="", help="short name for this run, e.g. 'baseline'")
    parser.add_argument(
        "--mode",
        choices=RETRIEVAL_MODES,
        default=HYBRID,
        help="retrieval strategy to measure",
    )
    parser.add_argument("--per-question", action="store_true", help="print each question's rank")
    parser.add_argument("--no-save", action="store_true", help="print only, write no JSON")
    args = parser.parse_args()

    if args.k <= 0:
        parser.error("--k must be positive")
    if args.label and not args.label.replace("-", "").replace("_", "").isalnum():
        parser.error("--label must be alphanumeric (dashes and underscores allowed)")

    result = run(args.dataset, args.db, args.k, args.label, args.mode)
    print(format_report(result))
    if args.per_question:
        print(format_per_question(result["per_question"]))
    if not args.no_save:
        print(f"Saved to {save_result(result, args.out_dir)}")


if __name__ == "__main__":
    main()
