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
        """Return the k most similar chunks: [{id, text, source, score}, ...]."""
        rows = self.db.execute(
            """
            SELECT c.id, c.text, c.source, v.distance
            FROM vec_chunks AS v
            JOIN chunks AS c ON c.id = v.rowid
            WHERE v.embedding MATCH ? AND v.k = ?
            ORDER BY v.distance
            """,
            (serialize_float32(query_vector), k),
        ).fetchall()
        return [
            {"id": chunk_id, "text": text, "source": source, "score": round(1 - distance, 4)}
            for chunk_id, text, source, distance in rows
        ]

    def list_documents(self) -> list[dict]:
        """Return uploaded documents with their chunk counts."""
        rows = self.db.execute(
            "SELECT source, COUNT(*) FROM chunks GROUP BY source ORDER BY source"
        ).fetchall()
        return [{"source": source, "chunks": count} for source, count in rows]

    def all_chunks(self) -> list[dict]:
        """Return every stored chunk: [{id, source, text}, ...] ordered by id."""
        rows = self.db.execute("SELECT id, source, text FROM chunks ORDER BY id").fetchall()
        return [
            {"id": chunk_id, "source": source, "text": text}
            for chunk_id, source, text in rows
        ]

    def delete_document(self, source: str) -> int:
        """Delete all chunks of a document. Returns deleted chunk count."""
        ids = [
            row[0]
            for row in self.db.execute(
                "SELECT id FROM chunks WHERE source = ?", (source,)
            ).fetchall()
        ]
        for chunk_id in ids:
            self.db.execute("DELETE FROM vec_chunks WHERE rowid = ?", (chunk_id,))
        self.db.execute("DELETE FROM chunks WHERE source = ?", (source,))
        self.db.commit()
        return len(ids)

    def close(self) -> None:
        self.db.close()