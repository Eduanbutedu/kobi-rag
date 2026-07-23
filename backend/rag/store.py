"""SQLite-based vector store using the sqlite-vec extension."""

import sqlite3
from pathlib import Path

import sqlite_vec
from sqlite_vec import serialize_float32

from rag.embedding import EMBEDDING_DIM


class DocumentStore:
    """Stores document chunks and their embeddings; supports semantic search."""

    def __init__(self, db_path: str | Path = "kobi_rag.db") -> None:
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self._create_tables()

    def _create_tables(self) -> None:
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                source TEXT NOT NULL,
                text TEXT NOT NULL
            )
            """
        )
        self.db.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                embedding float[{EMBEDDING_DIM}] distance_metric=cosine
            )
            """
        )
        self.db.commit()

    def add_document(self, source: str, chunks: list[str], vectors: list[list[float]]) -> int:
        """Store all chunks of a document with their embeddings. Returns chunk count."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        for text, vector in zip(chunks, vectors, strict=True):
            cur = self.db.execute(
                "INSERT INTO chunks (source, text) VALUES (?, ?)", (source, text)
            )
            self.db.execute(
                "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
                (cur.lastrowid, serialize_float32(vector)),
            )
        self.db.commit()
        return len(chunks)

    def search(self, query_vector: list[float], k: int = 3) -> list[dict]:
        """Return the k most similar chunks: [{text, source, score}, ...]."""
        rows = self.db.execute(
            """
            SELECT c.text, c.source, v.distance
            FROM vec_chunks AS v
            JOIN chunks AS c ON c.id = v.rowid
            WHERE v.embedding MATCH ? AND v.k = ?
            ORDER BY v.distance
            """,
            (serialize_float32(query_vector), k),
        ).fetchall()
        return [
            {"text": text, "source": source, "score": round(1 - distance, 4)}
            for text, source, distance in rows
        ]

    def close(self) -> None:
        self.db.close()