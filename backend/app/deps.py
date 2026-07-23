"""Shared application dependencies."""

from functools import lru_cache
from pathlib import Path

from rag.store import DocumentStore

DATA_DIR = Path("data")


@lru_cache(maxsize=1)
def get_store() -> DocumentStore:
    DATA_DIR.mkdir(exist_ok=True)
    return DocumentStore(DATA_DIR / "kobi_rag.db")