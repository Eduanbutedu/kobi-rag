"""Loading and validating the golden set of evaluation questions."""

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from eval.textio import read_text_utf8

PROTECTED_OUTPUT = "dataset.jsonl"


class DatasetError(ValueError):
    """Raised when a golden-set file is malformed."""


@dataclass(frozen=True)
class EvalCase:
    """One golden-set row: a question and the chunks that should answer it.

    `extras` carries review-only annotations written into drafts, such as the
    source document and a text preview. They are written out but never read
    back: the loader ignores any field it does not know, so a draft row can be
    pasted into the golden set as-is.
    """

    id: str
    question: str
    relevant_chunk_ids: list[int]
    note: str = ""
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        row = {
            "id": self.id,
            "question": self.question,
            "relevant_chunk_ids": list(self.relevant_chunk_ids),
            "note": self.note,
        }
        # Ek alanlar zorunlu alanların üzerine yazamaz
        row.update({k: v for k, v in self.extras.items() if k not in row})
        return row


def _fail(line_number: int, message: str) -> None:
    raise DatasetError(f"line {line_number}: {message}")


def parse_case(row: object, line_number: int) -> EvalCase:
    """Turn one decoded JSON row into an EvalCase, or raise DatasetError."""
    if not isinstance(row, dict):
        _fail(line_number, "each line must be a JSON object")

    case_id = row.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        _fail(line_number, "'id' must be a non-empty string")

    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        _fail(line_number, "'question' must be a non-empty string")

    chunk_ids = row.get("relevant_chunk_ids")
    if not isinstance(chunk_ids, list) or not chunk_ids:
        _fail(line_number, "'relevant_chunk_ids' must be a non-empty list")
    # bool int'in alt sınıfı olduğu için ayrıca eleniyor
    if any(isinstance(i, bool) or not isinstance(i, int) for i in chunk_ids):
        _fail(line_number, "'relevant_chunk_ids' must contain integers only")

    note = row.get("note", "")
    if not isinstance(note, str):
        _fail(line_number, "'note' must be a string")

    return EvalCase(
        id=case_id.strip(),
        question=question.strip(),
        relevant_chunk_ids=list(dict.fromkeys(chunk_ids)),
        note=note,
    )


def load_dataset(path: str | Path) -> list[EvalCase]:
    """Read a .jsonl golden set. Blank lines are ignored; ids must be unique."""
    file_path = Path(path)
    if not file_path.exists():
        raise DatasetError(f"dataset not found: {file_path}")

    cases: list[EvalCase] = []
    seen: dict[str, int] = {}
    for number, raw in enumerate(read_text_utf8(file_path).splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"line {number}: invalid JSON ({exc.msg})") from exc
        case = parse_case(row, number)
        if case.id in seen:
            _fail(number, f"duplicate id '{case.id}', first seen on line {seen[case.id]}")
        seen[case.id] = number
        cases.append(case)

    if not cases:
        raise DatasetError(f"dataset is empty: {file_path}")
    return cases


def write_dataset(path: str | Path, cases: Iterable[EvalCase]) -> int:
    """Write cases as .jsonl. Returns the number of rows written."""
    rows = [json.dumps(case.to_dict(), ensure_ascii=False) for case in cases]
    Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)
