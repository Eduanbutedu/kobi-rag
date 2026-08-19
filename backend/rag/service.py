"""High-level RAG operations: ingest documents, retrieve relevant chunks."""

from collections.abc import Sequence
from pathlib import Path

from rag.chunking import chunk_text
from rag.embedding import embed_texts
from rag.extraction import extract_text
from rag.store import DocumentStore

DENSE = "dense"
HYBRID = "hybrid"
RETRIEVAL_MODES = (DENSE, HYBRID)

# Her iki aramadan da bu kadar aday çekilip birleştirilir
FUSION_CANDIDATES = 20
# Reciprocal Rank Fusion sabiti. 60, yöntemi tanıtan çalışmadaki değer;
# tek bir listenin ilk sıralarının sonucu tek başına belirlemesini engeller.
RRF_K = 60


def ingest_file(store: DocumentStore, file_path: str | Path) -> int:
    """Extract, chunk, embed and store a document. Returns chunk count."""
    path = Path(file_path)
    text = extract_text(path)
    chunks = chunk_text(text)
    if not chunks:
        return 0
    vectors = embed_texts(chunks)
    return store.add_document(path.name, chunks, vectors)


def reciprocal_rank_fusion(rankings: Sequence[Sequence[dict]], k: int = RRF_K) -> list[dict]:
    """Merge ranked result lists by Reciprocal Rank Fusion.

    Each list contributes 1/(k + rank) to a chunk's score, with rank counted
    from 1. Only positions matter, never the underlying scores, which is what
    makes it safe to combine a cosine similarity with a BM25 value: the two
    are on scales that cannot be compared directly.

    Returns the chunks ordered by fused score, each carrying that score.
    """
    scores: dict[int, float] = {}
    chunks: dict[int, dict] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            chunk_id = chunk["id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            chunks.setdefault(chunk_id, chunk)

    # Eşit skorlarda sıralama kararlı olsun diye ikincil ölçüt chunk id
    order = sorted(scores, key=lambda i: (-scores[i], i))
    return [{**chunks[i], "score": round(scores[i], 6)} for i in order]


def retrieve(
    store: DocumentStore, query: str, k: int = 3, mode: str = HYBRID
) -> list[dict]:
    """Return the k chunks most relevant to the query: [{id, text, source, score}, ...].

    This is the single entry point for retrieval: the HTTP layer and the
    evaluation harness both call it, so they always measure the same code.

    In "hybrid" mode embedding search and keyword search are run separately
    and merged with Reciprocal Rank Fusion; `score` is then the fused score,
    not a similarity. "dense" keeps the embedding-only behaviour so the two
    can be measured against each other.
    """
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"unknown retrieval mode '{mode}' (expected one of {RETRIEVAL_MODES})")

    [query_vector] = embed_texts([query])
    if mode == DENSE:
        return store.search(query_vector, k=k)

    dense_hits = store.search(query_vector, k=FUSION_CANDIDATES)
    keyword_hits = store.bm25_search(query, k=FUSION_CANDIDATES)
    return reciprocal_rank_fusion([dense_hits, keyword_hits])[:k]