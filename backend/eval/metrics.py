"""Pure ranking metrics for retrieval evaluation.

Every function here takes plain ids and returns a number, with no database,
model or file access. That keeps them cheap to unit test and makes the
reported numbers auditable by hand.
"""

from collections.abc import Iterable, Sequence


def _top_k(retrieved_ids: Sequence[int], k: int) -> list[int]:
    if k <= 0:
        raise ValueError("k must be positive")
    return list(retrieved_ids[:k])


def _as_set(relevant_ids: Iterable[int]) -> set[int]:
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("relevant_ids must not be empty")
    return relevant


def recall_at_k(retrieved_ids: Sequence[int], relevant_ids: Iterable[int], k: int) -> float:
    """Fraction of the relevant chunks that appear in the top k results.

    With a single relevant chunk this is 1.0 when it was found and 0.0
    otherwise, so the dataset average equals the familiar hit rate.
    """
    relevant = _as_set(relevant_ids)
    found = relevant & set(_top_k(retrieved_ids, k))
    return len(found) / len(relevant)


def first_relevant_rank(
    retrieved_ids: Sequence[int], relevant_ids: Iterable[int], k: int
) -> int | None:
    """1-based rank of the first relevant chunk in the top k, or None if absent."""
    relevant = _as_set(relevant_ids)
    for rank, chunk_id in enumerate(_top_k(retrieved_ids, k), start=1):
        if chunk_id in relevant:
            return rank
    return None


def hit_rate_at_k(retrieved_ids: Sequence[int], relevant_ids: Iterable[int], k: int) -> float:
    """1.0 if any relevant chunk appears in the top k, 0.0 otherwise.

    Unlike recall, this does not care how many relevant chunks were marked.
    Recall for a question with three relevant chunks cannot exceed 0.333 at
    k=1, so a set with many multi-chunk questions reads low for reasons that
    have nothing to do with retrieval. This answers the plainer question of
    whether the user was shown something useful at all.
    """
    return 1.0 if first_relevant_rank(retrieved_ids, relevant_ids, k) is not None else 0.0


def reciprocal_rank_at_k(
    retrieved_ids: Sequence[int], relevant_ids: Iterable[int], k: int
) -> float:
    """1 / rank of the first relevant chunk in the top k; 0.0 if none is found."""
    rank = first_relevant_rank(retrieved_ids, relevant_ids, k)
    return 0.0 if rank is None else 1.0 / rank


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean; 0.0 for an empty sequence."""
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (p between 0 and 100); 0.0 if empty."""
    if not 0 <= p <= 100:
        raise ValueError("p must be between 0 and 100")
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight
