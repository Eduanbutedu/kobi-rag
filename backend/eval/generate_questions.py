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
from collections import Counter, defaultdict
from pathlib import Path

from eval.chunk_filter import boilerplate_reasons
from eval.dataset import EvalCase, write_dataset
from rag.llm import complete
from rag.service import retrieve
from rag.store import DocumentStore

DEFAULT_DB = Path("data/kobi_rag.db")
DEFAULT_OUTPUT = Path("eval/dataset_draft.jsonl")
PROTECTED_OUTPUT = "dataset.jsonl"
MIN_CHUNK_CHARS = 200
# Taslakta gösterilecek aday chunk sayısı ve önizleme uzunluğu
CANDIDATE_K = 5
PREVIEW_CHARS = 200

# Mevzuat metinlerinde model, soru yazmak yerine parçayı düzyazı olarak tekrar
# etmeye eğilimli. "?" şartı ve hukuk metninden örnekler bunu belirgin biçimde
# azaltıyor: 16 chunk'lık ölçümde kullanılabilir çıktı 9'dan 13'e çıktı.
SYSTEM_PROMPT = """Sen bir arama sistemi için değerlendirme verisi hazırlayan asistansın.
Sana bir doküman parçası verilecek. Görevin, bu parçanın cevapladığı TEK bir Türkçe soru yazmak.

EN ÖNEMLİ KURAL: Çıktın bir SORU olmalı ve mutlaka "?" ile bitmeli.
Parçayı özetleme, tekrarlama veya cevabı yazma. Yalnızca soruyu yaz.

Kurallar:
- Soru, yalnızca bu parçadaki bilgiyle cevaplanabilmeli.
- Soru, dokümanı hiç görmemiş birinin sorabileceği gibi kendi başına anlamlı olmalı.
- "Bu metinde", "bu parçada", "yukarıdaki metne göre" gibi ifadeleri ASLA kullanma.
- Parçadaki ayırt edici terimleri (isim, sayı, tarih, teknik terim) soruda mutlaka kullan.
- Evet/hayır sorusu yazma; "hangi", "kaç", "ne", "nasıl", "neden" ile başlayan sorular yaz.
- Soruyu Türkçe yaz; İngilizce kelime kullanma (teknik terimlerin özel adları hariç).
- SADECE soruyu yaz. Açıklama, numara, tırnak veya başka hiçbir şey ekleme.

Örnek parça 1: "FD002 alt kümesi 6 farklı çalışma koşulu içerir ve eğitim için 260 motor \
barındırır."
Örnek soru 1: FD002 alt kümesinde eğitim için kaç motor bulunmaktadır?

Örnek parça 2: "İşveren, işçinin yıllık ücretli izin hakkını iş sözleşmesinin devamı \
süresince kullandırmak zorundadır. Bu haktan vazgeçilemez."
Örnek soru 2: İşveren yıllık ücretli izin hakkını ne zaman kullandırmak zorundadır?

Örnek parça 3: "Mülki idare amiri, 5442 sayılı İl İdaresi Kanununun verdiği yetkiler \
çerçevesinde talebi uygun bulursa yeteri kadar kolluk kuvveti görevlendirir."
Örnek soru 3: Mülki idare amiri talebi uygun bulduğunda hangi yetkiye dayanarak kolluk \
kuvveti görevlendirir?"""

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

# Soru kelimeleri. Ekli biçimler açıkça sayılıyor ve iki yanı da \b ile
# sınırlanıyor: aksi hâlde "ne" -> "nedeniyle", "kim" -> "kimyasal",
# "kaç" -> "kaçınılmaz" içinde eşleşir.
_QUESTION_WORD = re.compile(
    r"\b(?:"
    # Bu köklerle başlayan başka yaygın kelime yok, ek serbest bırakılabilir
    r"hangi\w*"
    r"|nasıl\w*"
    r"|nere\w*"
    r"|niçin"
    # "niye" serbest bırakılamaz: "niyet" onunla başlıyor
    r"|niye"
    # Bu kökler gerçek kelimelerle çakışıyor (kaçınılmaz, kimyasal, nedeniyle),
    # bu yüzden ekleri tek tek sayılıyor
    r"|kaç(?:ı|a|ta|tan|tır|ar|ıncı|ında|ının)?"
    r"|kim(?:e|i|in|den|dir|ler|lere|lerin|lerde|lerdir|lerdedir)?"
    r"|ne(?:yi|ye|yin|den|dir|ler|lere|lerin|lerdir)?"
    r")\b",
    re.IGNORECASE,
)

# Ayrı yazılan soru eki. Türkçede cümlenin sonunda durur, bu yüzden yalnızca
# son birkaç kelimede aranıyor: "müdür" hem soru eki hem de unvan olduğu için
# cümle ortasındaki bir "genel müdür" soru sayılmamalı.
_QUESTION_PARTICLE = re.compile(
    r"(?:mi|mı|mu|mü)"
    r"(?:dir|dır|dur|dür|sin|sın|sun|sün|siniz|sınız|sunuz|sünüz"
    r"|yim|yım|yum|yüm|ydi|ydı|ydu|ydü|ymiş|ymış|ymuş|ymüş)?",
    re.IGNORECASE,
)
PARTICLE_WINDOW_WORDS = 3

# Modelin düz cümlenin sonuna soru işareti iliştirdiğinin işareti: ".?", "!?"
_PUNCTUATION_RESIDUE = re.compile(r"[.!;:,]\s*\?")

# Dokümanın konusunu değil, kendi yapısını soran kalıplar: "hangi maddede ...?"
_STRUCTURAL_TARGET = re.compile(
    # Yalnızca dokümanın kendi parçalarını adlandıran sözcükler. "belge" ve "ek"
    # bilerek yok: "hangi belgeler başvuruda zorunludur" gerçek dünyadaki
    # evrakı sorar, "ek\w*" ise "ekonomik" gibi kelimelere de eşleşirdi.
    # "şekil" de yok: "hangi şekilde" olağan bir soru kalıbı.
    r"\b(?:hangi|bu|söz konusu|ilgili)\s+"
    r"(?:tanım|madde|fıkra|bent|bölüm|kısım|başlık|paragraf|tablo|sayfa|"
    r"cümle|metin|doküman)"
    r"\w*\b",
    re.IGNORECASE,
)
# "Hangi kurum ...?" gibi, özne yerinde adı konmamış genel bir adla açılan soru
_GENERIC_OPENING = re.compile(r"^\s*hangi\w*\s+\w+", re.IGNORECASE)


def has_question_particle(text: str) -> bool:
    """Whether the sentence ends with a separately written Turkish question particle."""
    words = text.rstrip("?").split()
    return any(
        _QUESTION_PARTICLE.fullmatch(word.strip(".,;:!"))
        for word in words[-PARTICLE_WINDOW_WORDS:]
    )


def looks_like_question(text: str) -> bool:
    """Whether the sentence is really interrogative, not a statement plus '?'.

    A question mark alone means nothing: the model reliably ends declarative
    sentences with one. Turkish marks a question either with a question word
    or with the separately written mi/mı/mu/mü particle, so one of the two
    has to be present.
    """
    return bool(_QUESTION_WORD.search(text)) or has_question_particle(text)


def build_prompt(chunk_text: str) -> str:
    return (
        f"Doküman parçası:\n\n{chunk_text}\n\n"
        "Bu parçanın cevapladığı tek bir soru yaz. Çıktın yalnızca soru cümlesi olsun "
        've "?" ile bitsin. /no_think'
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

        # Soru işaretinden önceki boşluk ve yinelenen işaretler zararsız, düzelt
        line = re.sub(r"\s+\?", "?", line)
        line = re.sub(r"\?{2,}", "?", line)
        # Ama ".?" düz cümleye iliştirilmiş soru işaretidir, satırı ele
        if _PUNCTUATION_RESIDUE.search(line):
            continue

        if not (line.endswith("?") and len(line) > 10):
            continue
        if not looks_like_question(line):
            continue
        best = line[0].upper() + line[1:]
    return best


def is_structural_reference(question: str) -> bool:
    """Whether the question asks about the document's own structure.

    Two shapes both make a question useless for retrieval. One points at a
    part of the document rather than its subject ("hangi tanımda", "hangi
    maddede"). The other names no concrete thing at all -- no number, no
    proper noun -- so nothing anchors it to a particular chunk, as in
    "Hangi model değerlendirme yöntemlerini kullanmaktadır?".
    """
    if _STRUCTURAL_TARGET.search(question):
        return True
    # Yalnızca "Hangi <ad> ...?" ile açılan sorular inceleniyor: burada özne
    # yerinde adı konmamış genel bir ad var. "İşveren kaç gün içinde ...?"
    # gibi gerçek bir özneyle başlayan soru bu kurala girmez.
    if not _GENERIC_OPENING.match(question):
        return False
    body = question.rstrip("?")
    # İlk kelime cümle başı olduğu için her hâlükârda büyük harfli, sayılmıyor
    rest = body.split()[1:]
    has_anchor = any(char.isdigit() for char in body) or any(
        word.strip("\"'“”(),.").istitle() or word.isupper() for word in rest
    )
    return not has_anchor


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
    if is_structural_reference(question):
        flags.append("structural-reference")
    return flags


def partition_chunks(chunks: list[dict], min_chars: int) -> tuple[list[dict], Counter]:
    """Split chunks into usable ones and a tally of why the rest were dropped."""
    usable: list[dict] = []
    dropped: Counter = Counter()
    for chunk in chunks:
        if len(chunk["text"].strip()) < min_chars:
            dropped["too-short"] += 1
            continue
        reasons = boilerplate_reasons(chunk["text"])
        if reasons:
            dropped.update(reasons)
            dropped["_total"] += 1
            continue
        usable.append(chunk)
    return usable, dropped


def select_chunks(
    chunks: list[dict],
    count: int,
    seed: int,
    min_chars: int,
    per_document: int | None = None,
) -> list[dict]:
    """Pick chunks evenly across documents, skipping boilerplate.

    Sampling at random would follow document size, and the two largest laws
    are half the corpus. Instead each document is drawn from in turn, so a
    small law is represented as well as a large one. A document that runs out
    of usable chunks simply drops out and the rest take up its share.

    With `per_document` set, that number is a hard quota per document and no
    redistribution happens.
    """
    usable, _ = partition_chunks(chunks, min_chars)
    if not usable:
        raise SystemExit(
            f"No chunk is at least {min_chars} characters long and free of boilerplate. "
            "Lower --min-chars or ingest more documents."
        )

    rng = random.Random(seed)
    pools: dict[str, list[dict]] = defaultdict(list)
    for chunk in usable:
        pools[chunk["source"]].append(chunk)
    # Kaynak sırası ve her havuzun karışımı seed'e bağlı, yani tekrarlanabilir
    sources = sorted(pools)
    for source in sources:
        rng.shuffle(pools[source])

    if per_document is not None:
        return [chunk for source in sources for chunk in pools[source][:per_document]]

    selected: list[dict] = []
    positions = dict.fromkeys(sources, 0)
    while len(selected) < count:
        progressed = False
        for source in sources:
            if len(selected) >= count:
                break
            index = positions[source]
            if index < len(pools[source]):
                selected.append(pools[source][index])
                positions[source] += 1
                progressed = True
        if not progressed:
            break
    return selected


def format_filter_report(total: int, usable: list[dict], dropped: Counter) -> str:
    """Show how many chunks the filter removed and why."""
    boilerplate = dropped.get("_total", 0)
    too_short = dropped.get("too-short", 0)
    lines = [
        "",
        f"{total} chunk(s) in the store:",
        f"  {too_short:>6} too short (< min-chars)",
        f"  {boilerplate:>6} boilerplate",
    ]
    reasons = [(name, n) for name, n in dropped.most_common() if not name.startswith("_")]
    lines += [
        f"         {n:>5}  {name}" for name, n in reasons if name != "too-short"
    ]
    lines.append(f"  {len(usable):>6} usable")
    return "\n".join(lines)


def format_selection(selected: list[dict]) -> str:
    """Show how the sample spread across documents."""
    per_source = Counter(chunk["source"] for chunk in selected)
    lines = ["", f"Sampled {len(selected)} chunk(s) across {len(per_source)} document(s):"]
    lines += [f"  {n:>4}  {source}" for source, n in sorted(per_source.items())]
    return "\n".join(lines)


def preview_of(text: str, limit: int = PREVIEW_CHARS) -> str:
    """Collapse a chunk to a single readable line for the reviewer."""
    collapsed = " ".join(text.split())
    return collapsed[:limit]


def candidate_annotations(store: DocumentStore, question: str, k: int = CANDIDATE_K) -> dict:
    """Run the question through retrieval so the reviewer can widen the answer set.

    A question about legislation is often answered by several articles, and
    marking only the source chunk as relevant understates Recall@1. These are
    suggestions to check by hand -- they are never added to
    relevant_chunk_ids automatically.
    """
    hits = retrieve(store, question, k=k)
    return {
        "candidate_chunk_ids": [hit["id"] for hit in hits],
        "candidates": [
            {"id": hit["id"], "source": hit["source"], "preview": preview_of(hit["text"], 120)}
            for hit in hits
        ],
    }


def generate_case(chunk: dict, index: int, store: DocumentStore | None = None) -> EvalCase | None:
    """Ask the model for one question about this chunk. None if it gave nothing usable."""
    reply = complete(SYSTEM_PROMPT, build_prompt(chunk["text"]), temperature=0.4, max_tokens=120)
    question = clean_question(reply)
    if not question:
        return None
    flags = quality_flags(question)
    note = f"AUTO-DRAFT, review before use | source={chunk['source']} | chunk_id={chunk['id']}"
    if flags:
        note += f" | CHECK: {', '.join(flags)}"

    extras = {"source": chunk["source"], "chunk_preview": preview_of(chunk["text"])}
    if store is not None:
        extras.update(candidate_annotations(store, question))

    return EvalCase(
        id=f"gen{index:03d}",
        question=question,
        relevant_chunk_ids=[chunk["id"]],
        note=note,
        extras=extras,
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
    parser.add_argument(
        "--per-document",
        type=int,
        help="exact quota per document; overrides --count and disables redistribution",
    )
    parser.add_argument(
        "--no-candidates",
        action="store_true",
        help="skip the retrieval pass that suggests further relevant chunks",
    )
    args = parser.parse_args()

    if args.count <= 0:
        parser.error("--count must be positive")
    if args.per_document is not None and args.per_document <= 0:
        parser.error("--per-document must be positive")
    if args.out.name == PROTECTED_OUTPUT:
        parser.error(
            f"refusing to write {PROTECTED_OUTPUT}: it is the reviewed golden set. "
            "Write a draft instead and move rows over by hand."
        )

    store = DocumentStore(args.db)
    try:
        chunks = store.all_chunks()
        if not chunks:
            raise SystemExit(f"No chunks in {args.db}. Upload documents first.")

        usable, dropped = partition_chunks(chunks, args.min_chars)
        print(format_filter_report(len(chunks), usable, dropped))

        selected = select_chunks(
            chunks, args.count, args.seed, args.min_chars, args.per_document
        )
        print(format_selection(selected))
        print("Generating questions...\n")

        cases: list[EvalCase] = []
        for index, chunk in enumerate(selected, start=1):
            case = generate_case(chunk, index, None if args.no_candidates else store)
            if case is None:
                print(
                    f"  [{index:>3}/{len(selected)}] chunk {chunk['id']}: "
                    "no usable question, skipped"
                )
                continue
            cases.append(case)
            flags = quality_flags(case.question)
            marker = f"  [CHECK: {', '.join(flags)}]" if flags else ""
            print(
                f"  [{index:>3}/{len(selected)}] chunk {chunk['id']}: "
                f"{case.question[:70]}{marker}"
            )
    finally:
        store.close()

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
