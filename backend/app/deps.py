"""Shared application dependencies."""

from functools import lru_cache
from pathlib import Path

from rag.chat import ChatStore
from rag.store import DocumentStore

DATA_DIR = Path("data")


@lru_cache(maxsize=1)
def get_store() -> DocumentStore:
    DATA_DIR.mkdir(exist_ok=True)
    return DocumentStore(DATA_DIR / "kobi_rag.db")


@lru_cache(maxsize=1)
def get_chat() -> ChatStore:
    """Chat history, in the same SQLite file as the vector store."""
    DATA_DIR.mkdir(exist_ok=True)
    return ChatStore(DATA_DIR / "kobi_rag.db")
