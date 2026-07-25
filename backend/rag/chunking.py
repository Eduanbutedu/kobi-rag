"""Split raw document text into overlapping chunks for embedding."""

import re

# Cümle sonu noktalaması + boşluk = cümle sınırı
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text) if s.strip()]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into chunks of at most chunk_size characters.

    Chunks are built from whole sentences so that no sentence is cut in
    half; only a single sentence longer than chunk_size is hard-split.
    Trailing sentences of a chunk (up to ~overlap characters) are carried
    into the next chunk, so facts near a boundary stay findable.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    # 1) Cümlelere böl; chunk_size'ı aşan tek cümleyi kayan pencereyle kır
    pieces: list[str] = []
    for sentence in _split_sentences(text):
        if len(sentence) <= chunk_size:
            pieces.append(sentence)
            continue
        start = 0
        while start < len(sentence):
            end = start + chunk_size
            piece = sentence[start:end].strip()
            if piece:
                pieces.append(piece)
            if end >= len(sentence):
                break
            start = end - overlap

    # 2) Cümleleri chunk_size'ı aşmayacak şekilde chunk'lara topla
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for piece in pieces:
        extra = len(piece) + (1 if current else 0)
        if current and current_len + extra > chunk_size:
            chunks.append(" ".join(current))
            # Örtüşme: son cümlelerden ~overlap karakteri sonraki chunk'a taşı
            kept: list[str] = []
            kept_len = 0
            for s in reversed(current):
                if kept_len + len(s) + 1 > overlap:
                    break
                kept.insert(0, s)
                kept_len += len(s) + 1
            if kept_len + len(piece) + 1 > chunk_size:
                kept, kept_len = [], 0
            current = kept
            current_len = kept_len
        current.append(piece)
        current_len += len(piece) + (1 if len(current) > 1 else 0)

    if current:
        chunks.append(" ".join(current))

    return [c for c in (chunk.strip() for chunk in chunks) if c]