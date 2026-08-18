"""Assign chunk ids to hand-written questions, one question at a time.

Questions written in the user's own words -- not the wording of the
legislation -- are the ones worth measuring retrieval against, but nothing
can guess which chunks answer them. This walks through eval/questions_manual.txt,
shows the top hits for each question and records the ones you pick.

    python -m eval.annotate
    python -m eval.annotate --resume

Answers go to eval/dataset_manual.jsonl. Questions whose answer is not in the
top hits at all go to eval/unanswered_manual.jsonl instead: that is a
retrieval finding worth keeping, not a mistake to delete.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from eval.dataset import PROTECTED_OUTPUT, EvalCase
from eval.textio import read_text_utf8
from rag.service import retrieve
from rag.store import DocumentStore

DEFAULT_QUESTIONS = Path("eval/questions_manual.txt")
DEFAULT_OUTPUT = Path("eval/dataset_manual.jsonl")
DEFAULT_UNANSWERED = Path("eval/unanswered_manual.jsonl")
DEFAULT_DB = Path("data/kobi_rag.db")
TOP_K = 10
PREVIEW_CHARS = 250
ID_PREFIX = "man"
NO_HIT_NOTE = "NO-HIT-IN-TOP10"

SELECT = "select"
NONE = "none"
SKIP = "skip"
QUIT = "quit"

_CASE_ID = re.compile(rf"^{ID_PREFIX}(\d+)$")
# BOM ve sıfır genişlikli karakterler; konsoldan gelen ilk satıra karışabiliyor
_ZERO_WIDTH = str.maketrans({"﻿": None, "​": None, "‎": None, "‏": None})


@dataclass(frozen=True)
class Selection:
    """What the reviewer asked for at the prompt."""

    action: str
    indices: tuple[int, ...] = ()


def load_questions(path: str | Path) -> list[str]:
    """Read the question list, dropping comments and blank lines."""
    file_path = Path(path)
    if not file_path.exists():
        raise SystemExit(f"question file not found: {file_path}")

    questions = []
    for line in read_text_utf8(file_path).splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            questions.append(stripped)
    if not questions:
        raise SystemExit(f"no questions in {file_path}")
    return questions


def parse_selection(raw: str, result_count: int) -> Selection:
    """Interpret one prompt answer. Raises ValueError with a usable message.

    A byte order mark can arrive on the first typed line under Windows, which
    would turn "1" into an unrecognised token and silently record the question
    as unanswered. Zero-width characters are stripped before anything else.
    """
    text = raw.translate(_ZERO_WIDTH).strip().lower()
    if text in ("s", "skip"):
        return Selection(SKIP)
    if text in ("q", "quit"):
        return Selection(QUIT)
    if not text:
        return Selection(NONE)

    indices: list[int] = []
    for piece in (p for p in re.split(r"[,\s]+", text) if p):
        if not piece.isdigit():
            raise ValueError(f"'{piece}' is not a number, 's', 'q' or blank")
        number = int(piece)
        if not 1 <= number <= result_count:
            raise ValueError(f"{number} is out of range (1-{result_count})")
        if number not in indices:
            indices.append(number)
    return Selection(SELECT, tuple(indices))


def format_hits(question: str, hits: list[dict], number: int, total: int) -> str:
    """Render one question and its retrieval results for reading in a terminal."""
    lines = [
        "",
        "=" * 78,
        f"[{number}/{total}]  {question}",
        "=" * 78,
    ]
    if not hits:
        lines.append("  (no results)")
        return "\n".join(lines)

    for position, hit in enumerate(hits, start=1):
        preview = " ".join(hit["text"].split())[:PREVIEW_CHARS]
        lines.append(
            f"\n{position:>2}. id={hit['id']:<6} {hit['score']:.4f}  {hit['source']}"
        )
        lines.append(f"    {preview}")
    return "\n".join(lines)


def build_case(case_number: int, question: str, chosen: list[dict]) -> EvalCase:
    """Turn the picked results into a golden-set row."""
    if chosen:
        sources = sorted({hit["source"] for hit in chosen})
        note = f"MANUAL | source={', '.join(sources)}"
    else:
        note = f"MANUAL | {NO_HIT_NOTE}"
    return EvalCase(
        id=f"{ID_PREFIX}{case_number:03d}",
        question=question,
        relevant_chunk_ids=[hit["id"] for hit in chosen],
        note=note,
    )


def read_rows(path: Path) -> list[dict]:
    """Read a .jsonl file written by this script; empty list if absent."""
    if not path.exists():
        return []
    rows = []
    for line in read_text_utf8(path).splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def previous_work(paths: list[Path]) -> tuple[set[str], int]:
    """Questions already handled, and the highest case number used so far."""
    done: set[str] = set()
    highest = 0
    for path in paths:
        for row in read_rows(path):
            done.add(row.get("question", ""))
            match = _CASE_ID.match(str(row.get("id", "")))
            if match:
                highest = max(highest, int(match.group(1)))
    return done, highest


def append_case(path: Path, case: EvalCase) -> None:
    """Append one row, so an interrupted session keeps everything already done."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")


def ask(question_number: int, result_count: int) -> Selection:
    """Prompt until the answer parses. EOF is treated as quit."""
    prompt = (
        f"  Which results answer question {question_number}? "
        "[numbers / blank = none / s = skip / q = quit]: "
    )
    while True:
        try:
            raw = input(prompt)
        except EOFError:
            print()
            return Selection(QUIT)
        try:
            return parse_selection(raw, result_count)
        except ValueError as exc:
            print(f"  {exc}. Try again.")


def _use_utf8_console() -> None:
    """Keep Turkish characters readable in the Windows console."""
    for stream in (sys.stdout, sys.stdin):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign chunk ids to hand-written questions.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS, help="question list")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="vector store path")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="answered questions")
    parser.add_argument(
        "--unanswered", type=Path, default=DEFAULT_UNANSWERED, help="questions with no hit"
    )
    parser.add_argument("--k", type=int, default=TOP_K, help="how many results to show")
    parser.add_argument(
        "--resume", action="store_true", help="skip questions already recorded and carry on"
    )
    args = parser.parse_args()

    _use_utf8_console()

    if args.k <= 0:
        parser.error("--k must be positive")
    for path in (args.out, args.unanswered):
        if path.name == PROTECTED_OUTPUT:
            parser.error(
                f"refusing to write {PROTECTED_OUTPUT}: it is the reviewed golden set. "
                "Write a separate file and move rows over by hand."
            )

    questions = load_questions(args.questions)
    done, highest = previous_work([args.out, args.unanswered])
    if done and not args.resume:
        raise SystemExit(
            f"{args.out} already holds {len(done)} answered question(s).\n"
            "Pass --resume to carry on from there, or move the file aside to start over."
        )

    store = DocumentStore(args.db)
    try:
        if not store.all_chunks():
            raise SystemExit(f"No chunks in {args.db}. Upload documents first.")

        print("Loading the embedding model...")
        retrieve(store, "ısınma sorgusu", k=1)
        print(f"Ready. {len(questions)} question(s) in {args.questions}.")
        if args.resume and done:
            print(f"Resuming: {len(done)} already recorded.")

        case_number = highest
        answered = unanswered = skipped = 0
        for position, question in enumerate(questions, start=1):
            if args.resume and question in done:
                continue

            hits = retrieve(store, question, k=args.k)
            print(format_hits(question, hits, position, len(questions)))

            selection = ask(position, len(hits))
            if selection.action == QUIT:
                print("\nStopping here; everything answered so far is saved.")
                break
            if selection.action == SKIP:
                skipped += 1
                print("  skipped")
                continue

            chosen = [hits[index - 1] for index in selection.indices]
            case_number += 1
            case = build_case(case_number, question, chosen)
            if chosen:
                append_case(args.out, case)
                answered += 1
                print(f"  recorded {case.id} -> chunks {case.relevant_chunk_ids}")
            else:
                append_case(args.unanswered, case)
                unanswered += 1
                print(f"  recorded {case.id} as unanswered ({NO_HIT_NOTE})")
    finally:
        store.close()

    print(
        f"\n{answered} answered -> {args.out}"
        f"\n{unanswered} with no hit in the top {args.k} -> {args.unanswered}"
        f"\n{skipped} skipped"
    )
    if answered:
        print(
            "\nReview eval/dataset_manual.jsonl, then move the rows you trust into "
            "eval/dataset.jsonl.\nNothing is added to the golden set for you."
        )


if __name__ == "__main__":
    main()
