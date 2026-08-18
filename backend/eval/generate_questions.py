"""Draft golden-set questions from stored chunks using the local LLM.

Samples N chunks, asks the model for one Turkish question each, and writes
eval/dataset_draft.jsonl with the source chunk already marked as relevant.

The draft is a starting point, not a golden set: review every line by hand
and move the good ones into eval/dataset.jsonl yourself. This script never
writes to dataset.jsonl.

    python -m eval.generate_questions -n 20 --seed 42
"""

import argparse
import random
import re
import sys
from pathlib import Path

from eval.dataset import EvalCase, write_dataset
from rag.llm import complete
from rag.store import DocumentStore

DEFAULT_DB = Path("data/kobi_rag.db")
DEFAULT_OUTPUT = Path("eval/dataset_draft.jsonl")
PROTECTED_OUTPUT = "dataset.jsonl"
MIN_CHUNK_CHARS = 200

SYSTEM_PROMPT = """Sen bir arama sistemi için değerlendirme verisi hazırlayan asistansın.
Sana bir doküman parçası verilecek. Görevin, bu parçanın cevapladığı TEK bir Türkçe soru yazmak.

Kurallar:
- Soru, yalnızca bu parçadaki bilgiyle cevaplanabilmeli.
- Soru, dokümanı hiç görmemiş birinin sorabileceği gibi kendi başına anlamlı olmalı.
- "Bu metinde", "bu parçada", "yukarıdaki metne göre" gibi ifadeleri ASLA kullanma.
- Parçadaki ayırt edici terimleri (isim, sayı, tarih, teknik terim) soruda mutlaka kullan.
- Evet/hayır sorusu yazma; "hangi", "kaç", "ne", "nasıl", "neden" ile başlayan sorular yaz.
- Soruyu Türkçe yaz; İngilizce kelime kullanma (teknik terimlerin özel adları hariç).
- SADECE soruyu yaz. Açıklama, numara, tırnak veya başka hiçbir şey ekleme.

Örnek parça: "FD002 alt kümesi 6 farklı çalışma koşulu içerir ve eğitim için 260 motor barındırır."
Örnek soru: FD002 alt kümesinde eğitim için kaç motor bulunmaktadır?"""

# Modelin sık ürettiği "Bu metinde, ..." kalıbı; soruyu bozmadan sökülebiliyor
# Uzun varyantlar önce gelmeli, yoksa "parçada" içindeki "parça" eşleşip "da" artıyor
_META_PREFIX = re.compile(
    r"^(?:"
    r"bu (?:metinde|metnin|metne|metin|parçaya|parçanın|parçada|parça"
    r"|belgede|belgeye|dokümanda|dokümana)(?:\s+göre)?"
    r"|söz konusu (?:metinde|metin|parçada|parça)"
    r"|yukarıdaki[^,]{0,40}(?=,)"
    r")\b[\s,:]*",
    re.IGNORECASE,
)
_META_MENTION = re.compile(
    r"\b(bu metin|bu parça|yukarıdaki|söz konusu metin|verilen metin|dokümanda)", re.IGNORECASE
)
# Türkçe evet/hayır soru eki: "... söylenebilir mi?"
_YES_NO = re.compile(r"\b(mi|mı|mu|mü)(dir|dır|dur|dür)?\s*\?$", re.IGNORECASE)
_LATIN_ONLY = re.compile(r"^[\x00-\x7f]+$")


def build_prompt(chunk_text: str) -> str:
    return (
        f"Doküman parçası:\n\n{chunk_text}\n\n"
        "Bu parçanın cevapladığı tek bir soru yaz. /no_think"
    )


def clean_question(raw: str) -> str:
    """Reduce a model reply to a single usable question line, or '' if there is none.

    Takes the last question-shaped line, because the model often narrates a
    little before landing on the actual question.
    """
    best = ""
    for line in (line.strip() for line in raw.splitlines()):
        if not line:
            continue
        # Model bazen "1." ya da "Soru:" ön eki veya tırnak ekliyor
        line = line.lstrip("0123456789.-) ").strip()
        for prefix in ("Soru:", "SORU:", "Question:"):
            if line.startswith(prefix):
                line = line[len(prefix) :].strip()
        line = line.strip('"').strip("'").strip()
        # "... şu sorudur: X?" kalıbında yalnızca son cümle soruyu taşır
        if ": " in line and line.endswith("?"):
            line = line.rsplit(": ", 1)[-1].strip().strip('"').strip()
        line = _META_PREFIX.sub("", line).strip()
        if line and line.endswith("?") and len(line) > 10:
            best = line[0].upper() + line[1:]
    return best


def quality_flags(question: str) -> list[str]:
    """Name the things a human reviewer should look at before accepting a question."""
    flags = []
    if _META_MENTION.search(question):
        flags.append("meta-reference")
    if _YES_NO.search(question):
        flags.append("yes-no")
    if len(question) < 25:
        flags.append("very-short")
    if _LATIN_ONLY.match(question):
        flags.append("maybe-not-turkish")
    return flags


def select_chunks(chunks: list[dict], count: int, seed: int, min_chars: int) -> list[dict]:
    """Randomly pick up to `count` chunks that are long enough to hold a fact."""
    usable = [c for c in chunks if len(c["text"].strip()) >= min_chars]
    if not usable:
        raise SystemExit(
            f"No chunk is at least {min_chars} characters long. "
            "Lower --min-chars or ingest longer documents."
        )
    return random.Random(seed).sample(usable, min(count, len(usable)))


def generate_case(chunk: dict, index: int) -> EvalCase | None:
    """Ask the model for one question about this chunk. None if it gave nothing usable."""
    reply = complete(SYSTEM_PROMPT, build_prompt(chunk["text"]), temperature=0.4, max_tokens=120)
    question = clean_question(reply)
    if not question:
        return None
    flags = quality_flags(question)
    note = f"AUTO-DRAFT, review before use | source={chunk['source']} | chunk_id={chunk['id']}"
    if flags:
        note += f" | CHECK: {', '.join(flags)}"
    return EvalCase(
        id=f"gen{index:03d}",
        question=question,
        relevant_chunk_ids=[chunk["id"]],
        note=note,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft eval questions from stored chunks.")
    parser.add_argument("-n", "--count", type=int, default=20, help="how many chunks to sample")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="vector store path")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="draft output .jsonl")
    parser.add_argument("--seed", type=int, default=42, help="sampling seed for reproducibility")
    parser.add_argument(
        "--min-chars",
        type=int,
        default=MIN_CHUNK_CHARS,
        help="skip chunks shorter than this",
    )
    args = parser.parse_args()

    if args.count <= 0:
        parser.error("--count must be positive")
    if args.out.name == PROTECTED_OUTPUT:
        parser.error(
            f"refusing to write {PROTECTED_OUTPUT}: it is the reviewed golden set. "
            "Write a draft instead and move rows over by hand."
        )

    store = DocumentStore(args.db)
    try:
        chunks = store.all_chunks()
    finally:
        store.close()
    if not chunks:
        raise SystemExit(f"No chunks in {args.db}. Upload documents first.")

    selected = select_chunks(chunks, args.count, args.seed, args.min_chars)
    print(f"Sampled {len(selected)} of {len(chunks)} chunks; generating questions...\n")

    cases: list[EvalCase] = []
    for index, chunk in enumerate(selected, start=1):
        case = generate_case(chunk, index)
        if case is None:
            print(
                f"  [{index:>3}/{len(selected)}] chunk {chunk['id']}: "
                "no usable question, skipped"
            )
            continue
        cases.append(case)
        flags = quality_flags(case.question)
        marker = f"  [CHECK: {', '.join(flags)}]" if flags else ""
        print(f"  [{index:>3}/{len(selected)}] chunk {chunk['id']}: {case.question[:70]}{marker}")

    if not cases:
        raise SystemExit("\nThe model produced no usable questions; nothing was written.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = write_dataset(args.out, cases)
    flagged = sum(1 for case in cases if quality_flags(case.question))
    print(f"\nWrote {written} draft question(s) to {args.out}; {flagged} need a closer look.")
    print(
        "Review each line, fix the questions and relevant_chunk_ids, then move the\n"
        "good ones into eval/dataset.jsonl. Nothing is added to the golden set for you."
    )
    if written < len(selected):
        print(f"Note: {len(selected) - written} chunk(s) yielded no question.", file=sys.stderr)


if __name__ == "__main__":
    main()
