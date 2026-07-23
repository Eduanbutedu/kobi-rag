"""Turn text into semantic vectors using a local multilingual model."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the model once and reuse it (loading takes seconds)."""
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts into 384-dim vectors."""
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]