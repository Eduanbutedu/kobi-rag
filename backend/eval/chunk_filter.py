"""Tell apart chunks that carry a fact from PDF boilerplate.

Legislation PDFs are full of material that reads like text but answers no
question: tables of contents, article-heading lists, amendment and footnote
tables, and blocks of repealed articles. Questions generated from those are
unusable, so they are filtered out before sampling.

Every threshold below was calibrated against the ingested corpus rather than
guessed. The measured split on 8,851 chunks:

    genuine prose      filler 0.00  density 0.03-0.08  max words 11-26
    dotted TOC         filler 0.70  density 0.74       max words 4
    whitespace TOC     blank lines 0.71-0.75           max words 8-10
    amendment tables   density 0.68-0.77               max words 2-6
    repealed articles  repeal markers on half the lines

Deliberately not used: "share of lines ending in a number". It looks like a
table signal but Turkish legal prose cites article numbers at line ends, so
real text scored 0.40 on it -- as high as the boilerplate.
"""

import re
import string

# Nokta/tire "leader" dizileri: içindekiler tablolarının imzası
MAX_FILLER_RATIO = 0.15
# Rakam ve noktalama yoğunluğu: değişiklik ve tarih tabloları
MAX_DIGIT_PUNCT_DENSITY = 0.30
# Satır başına ortalama kelime: tümüyle liste olan bloklar
MIN_WORDS_PER_LINE = 2.5
# En az bir satır gerçek bir cümle taşımalı
MIN_WORDS_IN_LONGEST_LINE = 8
# "(Mülga: ...)" kayıtlarından oluşan bloklar
MAX_REPEAL_MARKER_RATIO = 0.4
# Boşluk "leader"lı içindekiler tabloları
MAX_BLANK_LINE_RATIO = 0.5
# Kısa satırlardan oluşan bloklar
SHORT_LINE_CHARS = 25
MAX_SHORT_LINE_RATIO = 0.8
SHORT_LINE_ESCAPE_WORDS = 12

_FILLER_RUN = re.compile(r"[.\-_·]{3,}")
_REPEAL_MARKER = re.compile(r"\(\s*(?:Mülga|Değişik|Ek|İptal)\s*:", re.IGNORECASE)


def _measure(text: str) -> dict[str, float]:
    """Compute every signal in one pass over the text."""
    stripped = text.strip()
    chars = max(len(stripped), 1)
    raw_lines = stripped.splitlines()
    lines = [line.strip() for line in raw_lines if line.strip()]
    line_count = max(len(lines), 1)

    filler_chars = sum(len(match.group()) for match in _FILLER_RUN.finditer(stripped))
    noisy_chars = sum(1 for c in stripped if c.isdigit() or c in string.punctuation)
    words_per_line = [len(line.split()) for line in lines]

    return {
        "filler_ratio": filler_chars / chars,
        "digit_punct_density": noisy_chars / chars,
        "words_per_line": sum(words_per_line) / line_count,
        "longest_line_words": max(words_per_line, default=0),
        "repeal_ratio": len(_REPEAL_MARKER.findall(stripped)) / line_count,
        "blank_line_ratio": (len(raw_lines) - len(lines)) / max(len(raw_lines), 1),
        "short_line_ratio": sum(1 for line in lines if len(line) < SHORT_LINE_CHARS) / line_count,
    }


def boilerplate_reasons(text: str) -> list[str]:
    """Name every reason this chunk looks like boilerplate. Empty means usable."""
    if not text.strip():
        return ["empty"]

    m = _measure(text)
    # PDF çıkarımı gerçek bir cümleyi tek tek kelimelere bölebiliyor. Bu yüzden
    # satır biçimine bakan ortalamalar, metinde hakiki bir cümle varsa elemez.
    has_sentence_line = m["longest_line_words"] >= SHORT_LINE_ESCAPE_WORDS

    reasons = []
    if m["filler_ratio"] > MAX_FILLER_RATIO:
        reasons.append("filler-runs")
    if m["digit_punct_density"] > MAX_DIGIT_PUNCT_DENSITY:
        reasons.append("digit-heavy")
    if m["words_per_line"] < MIN_WORDS_PER_LINE and not has_sentence_line:
        reasons.append("few-words-per-line")
    if m["longest_line_words"] < MIN_WORDS_IN_LONGEST_LINE:
        reasons.append("no-sentence-like-line")
    if m["repeal_ratio"] > MAX_REPEAL_MARKER_RATIO:
        reasons.append("repealed-articles")
    if m["blank_line_ratio"] > MAX_BLANK_LINE_RATIO:
        reasons.append("blank-line-leaders")
    if m["short_line_ratio"] > MAX_SHORT_LINE_RATIO and not has_sentence_line:
        reasons.append("short-lines")
    return reasons


def is_informative(text: str) -> bool:
    """Whether a chunk carries enough real prose to build a question from."""
    return not boilerplate_reasons(text)
