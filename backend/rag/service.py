"""High-level RAG operations: ingest documents, retrieve relevant chunks."""

from collections.abc import Sequence
from pathlib import Path

from rag.chunking import chunk_text
from rag.embedding import embed_texts
from rag.extraction import extract_text
from rag.rerank import rerank as rerank_candidates
from rag.store import DocumentStore

DENSE = "dense"
HYBRID = "hybrid"
RETRIEVAL_MODES = (DENSE, HYBRID)

# Her iki aramadan da bu kadar aday çekilip birleştirilir
FUSION_CANDIDATES = 20
# Cross-encoder'a verilecek kısa liste; puanlama pahalı olduğu için dar tutulur.
# k=10 ile bu değer de 10 olduğunda yeniden sıralama yalnızca ilk onu kendi
# içinde sıralayabilir: 11-20 arasındaki bir chunk artık ilk ona giremez.
RERANK_CANDIDATES = 10
# Reciprocal Rank Fusion sabiti. 60, yöntemi tanıtan çalışmadaki değer;
# tek bir listenin ilk sıralarının sonucu tek başına belirlemesini engeller.
RRF_K = 60

# Altın sette 0,5 / 0,87 / 0,90 / 0,95 / 1,0 ölçüldü; 0,95 seçildi. Gerekçe
# ve kabul edilen ödünler eval/README.md'deki "Choosing the keyword weight"
# bölümünde. Eşit ağırlıkta (1,0) yalnızca dense'in bulduğu 6. sıradaki bir
# sonuç 1/66 alıp BM25'in ilk beşinin altında kalıyor; 0,95 bu baskıyı
# yumuşatırken anahtar kelimenin tek başına bulduğu sonuçların ilk ona
# girmesine hâlâ izin veriyor.
DENSE_WEIGHT = 1.0
BM25_WEIGHT = 0.95


def ingest_file(store: DocumentStore, file_path: str | Path) -> int:
    """Extract, chunk, embed and store a document. Returns chunk count."""
    path = Path(file_path)
    text = extract_text(path)
    chunks = chunk_text(text)
    if not chunks:
        return 0
    vectors = embed_texts(chunks)
    return store.add_document(path.name, chunks, vectors)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[dict]],
    weights: Sequence[float] | None = None,
    k: int = RRF_K,
) -> list[dict]:
    """Merge ranked result lists by weighted Reciprocal Rank Fusion.

    Each list contributes weight/(k + rank) to a chunk's score, with rank
    counted from 1. Only positions matter, never the underlying scores, which
    is what makes it safe to combine a cosine similarity with a BM25 value:
    the two are on scales that cannot be compared directly.

    `weights` lines up with `rankings` and defaults to 1.0 for each, so an
    unweighted call behaves exactly as before. Weighting matters because with
    equal weights a chunk only one search finds is decided by arithmetic
    rather than by relevance: at rank 6 it scores 1/66, which every one of
    the other search's top five beats.

    Returns the chunks ordered by fused score, each carrying that score.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must have one entry per ranking")

    scores: dict[int, float] = {}
    chunks: dict[int, dict] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, chunk in enumerate(ranking, start=1):
            chunk_id = chunk["id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
            chunks.setdefault(chunk_id, chunk)

    # Eşit skorlarda sıralama kararlı olsun diye ikincil ölçüt chunk id
    order = sorted(scores, key=lambda i: (-scores[i], i))
    return [{**chunks[i], "score": round(scores[i], 6)} for i in order]


def retrieve(
    store: DocumentStore,
    query: str,
    k: int = 3,
    mode: str = HYBRID,
    dense_weight: float = DENSE_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
    rerank: bool = False,
) -> list[dict]:
    """Return the k chunks most relevant to the query: [{id, text, source, score}, ...].

    This is the single entry point for retrieval: the HTTP layer and the
    evaluation harness both call it, so they always measure the same code.

    In "hybrid" mode embedding search and keyword search are run separately
    and merged with weighted Reciprocal Rank Fusion; `score` is then the fused
    score, not a similarity. "dense" keeps the embedding-only behaviour so the
    two can be measured against each other.

    Keyword results carry less weight than embedding results so that they add
    to the ranking instead of overruling it. The two weights are arguments
    rather than constants so they can be tuned against the golden set.

    With `rerank` set, a wider shortlist is retrieved and then re-scored by a
    cross-encoder that reads question and chunk together; `score` is then the
    cross-encoder score. It costs a model call per query, so it is off by
    default and switched on per run.
    """
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"unknown retrieval mode '{mode}' (expected one of {RETRIEVAL_MODES})")

    # Yeniden sıralanacaksa geniş bir kısa liste çekilir, sonra k'ya inilir
    candidate_k = RERANK_CANDIDATES if rerank else k

    [query_vector] = embed_texts([query])
    if mode == DENSE:
        candidates = store.search(query_vector, k=candidate_k)
    else:
        dense_hits = store.search(query_vector, k=FUSION_CANDIDATES)
        keyword_hits = store.bm25_search(query, k=FUSION_CANDIDATES)
        # Ağırlıklar sıralamalarla aynı yerde ve aynı sırada kuruluyor
        candidates = reciprocal_rank_fusion(
            [dense_hits, keyword_hits], [dense_weight, bm25_weight]
        )[:candidate_k]

    if rerank:
        return rerank_candidates(query, candidates, k)
    return candidates[:k]