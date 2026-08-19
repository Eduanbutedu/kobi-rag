"""SQLite-based vector store using the sqlite-vec extension.

Chunks are searchable two ways: by embedding similarity through sqlite-vec,
and by keyword through an FTS5 index. The FTS5 index is an external-content
table over `chunks`, so the text is stored once and the two views cannot
drift apart.
"""

import re
import sqlite3
from pathlib import Path

import sqlite_vec
from sqlite_vec import serialize_float32

from rag.embedding import EMBEDDING_DIM
from rag.stopwords_tr import is_stopword

# FTS5 şeması eklendiğinde artırılır; eski veritabanları açılışta yükseltilir
SCHEMA_VERSION = 1

# Türkçe metin için remove_diacritics=0: ş/ğ/ö/ü/ç harflerinin işaretleri
# korunur, aksi hâlde "sağlık" ile "saglik" aynı kabul edilirdi.
#
# Bilinen sınır: unicode61, İ (U+0130) harfini küçültmüyor ve I harfini ı
# yerine i'ye indiriyor. Yani "İZİN" başlığı "izin" sorgusuyla eşleşmez.
# Gerçek korpusta ölçüldü: 31.038 terimin yalnızca 379'u (%1,2) bu yüzden
# ikiye ayrılıyor ve altın sette BM25 isabetini 25 soruda 1 soru kadar
# değiştiriyor. Metnin katlanmış ikinci bir kopyasını tutmayı hak etmiyor.
FTS_TOKENIZER = "unicode61 remove_diacritics 0"

# FTS5 sorgu dili tırnak, yıldız, AND/OR gibi işaretlere anlam yüklüyor;
# kullanıcı sorusundan yalnızca kelimeler alınıp tırnaklanarak veriliyor
_WORD = re.compile(r"\w+", re.UNICODE)
MIN_TERM_CHARS = 2


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
        self._create_fts()
        self.db.commit()

    def _create_fts(self) -> None:
        """Create the keyword index and keep it in step with `chunks`.

        The index holds no text of its own: `content='chunks'` points it at
        the existing table, and the triggers below forward every insert and
        delete. A database written before this index existed is filled in
        once, tracked with PRAGMA user_version.
        """
        self.db.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                content='chunks',
                content_rowid='id',
                tokenize='{FTS_TOKENIZER}'
            )
            """
        )
        for statement in (
            """
            CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END
            """,
            """
            CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text)
                VALUES ('delete', old.id, old.text);
            END
            """,
            """
            CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text)
                VALUES ('delete', old.id, old.text);
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END
            """,
        ):
            self.db.execute(statement)

        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version < SCHEMA_VERSION:
            # Trigger'lardan önce yazılmış chunk'lar için indeksi bir kez kur
            self.db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
            self.db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

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

    def bm25_search(self, query: str, k: int = 3) -> list[dict]:
        """Return the k best keyword matches: [{id, text, source, score}, ...].

        Terms are OR'ed rather than AND'ed: a whole question rarely appears in
        one chunk, and BM25 already rewards the chunks that match more of it.
        Score is the negated FTS5 bm25() value, so larger is better and it
        reads the same way round as the embedding score.
        """
        expression = self._match_expression(query)
        if not expression:
            return []

        rows = self.db.execute(
            """
            SELECT c.id, c.text, c.source, bm25(chunks_fts) AS rank
            FROM chunks_fts
            JOIN chunks AS c ON c.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (expression, k),
        ).fetchall()
        return [
            {"id": chunk_id, "text": text, "source": source, "score": round(-rank, 4)}
            for chunk_id, text, source, rank in rows
        ]

    @staticmethod
    def _match_expression(query: str) -> str:
        """Turn a free-text question into a safe FTS5 MATCH expression.

        Function words are dropped, because OR'ed terms let a chunk qualify
        on "içinde" alone. If the question is nothing but function words the
        unfiltered terms are used instead -- a worse query still beats no
        query at all.
        """
        terms = [t for t in _WORD.findall(query) if len(t) >= MIN_TERM_CHARS]
        content_terms = [t for t in terms if not is_stopword(t)]
        # Her terim tırnak içinde: sorgudaki tırnak, yıldız, tire gibi
        # işaretler FTS5 söz dizimi olarak yorumlanmasın
        return " OR ".join(f'"{term}"' for term in content_terms or terms)

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