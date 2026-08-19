"""Re-score retrieval candidates with a multilingual cross-encoder.

Embedding search compares a question and a chunk through two vectors built
independently, so it never sees the pair together. A cross-encoder reads both
at once and scores how well the chunk answers that question, which is far
more accurate and far too slow to run over a whole corpus. It is used here
the usual way: cheap retrieval proposes a shortlist, this re-orders it.

The model is multilingual and covers Turkish. It is loaded once per process,
because loading costs seconds while scoring twenty pairs costs milliseconds.
"""

from collections.abc import Sequence
from functools import lru_cache

from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
# Soru ve chunk birlikte modele giriyor; 512 token ikisine birden yetiyor
MAX_LENGTH = 512


@lru_cache(maxsize=1)
def _get_model() -> CrossEncoder:
    """Load the cross-encoder once and reuse it (loading takes seconds)."""
    return CrossEncoder(MODEL_NAME, max_length=MAX_LENGTH)


def order_by_score(
    candidates: Sequence[dict], scores: Sequence[float], top_k: int
) -> list[dict]:
    """Sort candidates by score, highest first, and keep the best top_k.

    Ties break on chunk id so a run is reproducible. The cross-encoder score
    replaces `score`, the same way the fused score does after fusion.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if len(candidates) != len(scores):
        raise ValueError("candidates and scores must have the same length")

    ordered = sorted(
        zip(candidates, scores, strict=True),
        key=lambda pair: (-pair[1], pair[0]["id"]),
    )
    return [{**chunk, "score": round(float(score), 6)} for chunk, score in ordered[:top_k]]


def rerank(query: str, candidates: Sequence[dict], top_k: int) -> list[dict]:
    """Re-score candidates against the query and return the best top_k."""
    if not candidates:
        return []
    pairs = [(query, chunk["text"]) for chunk in candidates]
    scores = _get_model().predict(pairs)
    return order_by_score(candidates, list(scores), top_k)
