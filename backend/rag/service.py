"""High-level RAG operations: ingest documents, retrieve relevant chunks."""

from pathlib import Path

from rag.chunking import chunk_text
from rag.embedding import embed_texts
from rag.extraction import extract_text
from rag.store import DocumentStore


def ingest_file(store: DocumentStore, file_path: str | Path) -> int:
    """Extract, chunk, embed and store a document. Returns chunk count."""
    path = Path(file_path)
    text = extract_text(path)
    chunks = chunk_text(text)
    if not chunks:
        return 0
    vectors = embed_texts(chunks)
    return store.add_document(path.name, chunks, vectors)


def retrieve(store: DocumentStore, query: str, k: int = 3) -> list[dict]:
    """Return the k chunks most relevant to the query."""
    [query_vector] = embed_texts([query])
    return store.search(query_vector, k=k)