"""Persistent chat history: sessions and the messages inside them.

Kept apart from DocumentStore because it answers a different question. The
vector store is about the corpus; this is about what was asked of it. They
share the SQLite file so a deployment is still one database, but neither
touches the other's tables.
"""

import json
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

USER = "user"
ASSISTANT = "assistant"
ROLES = (USER, ASSISTANT)


def _now() -> str:
    # Mikrosaniye çözünürlüğü şart: saniye çözünürlüğünde, aynı saniye içinde
    # kullanılan iki oturum eşit görünüp sıralama id'ye düşüyor ve "en son
    # kullanılan üstte" bozuluyor. Sabit genişlikli ISO metni olduğu gibi
    # sıralanabiliyor.
    return datetime.now(UTC).isoformat(timespec="microseconds")


class ChatStore:
    """Stores question-and-answer sessions alongside the vector store."""

    def __init__(self, db_path: str | Path = "kobi_rag.db") -> None:
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        # ON DELETE CASCADE bağlantı başına açılmak zorunda
        self.db.execute("PRAGMA foreign_keys = ON")
        self._create_tables()

    def _create_tables(self) -> None:
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                session_id INTEGER NOT NULL
                    REFERENCES sessions(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                sources_json TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        # Bir oturumun mesajlarını sırayla okumak en sık yapılan sorgu
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS messages_by_session ON messages (session_id, id)"
        )
        self.db.commit()

    def create_session(self, title: str = "") -> int:
        """Start a session. The title is filled in later, once there is one."""
        now = _now()
        cur = self.db.execute(
            "INSERT INTO sessions (title, created_at, updated_at) VALUES (?, ?, ?)",
            (title, now, now),
        )
        self.db.commit()
        return cur.lastrowid

    def session_exists(self, session_id: int) -> bool:
        row = self.db.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return row is not None

    def list_sessions(self) -> list[dict]:
        """Sessions, most recently active first."""
        rows = self.db.execute(
            """
            SELECT s.id, s.title, s.created_at, s.updated_at,
                   COUNT(m.id) AS message_count
            FROM sessions AS s
            LEFT JOIN messages AS m ON m.session_id = s.id
            GROUP BY s.id
            ORDER BY s.updated_at DESC, s.id DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def add_message(
        self,
        session_id: int,
        role: str,
        content: str,
        sources: Sequence[dict] | None = None,
    ) -> int:
        """Append a message and mark the session as just used."""
        if role not in ROLES:
            raise ValueError(f"unknown role '{role}' (expected one of {ROLES})")
        if not self.session_exists(session_id):
            raise ValueError(f"no such session: {session_id}")

        now = _now()
        cur = self.db.execute(
            """
            INSERT INTO messages (session_id, role, content, sources_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                session_id,
                role,
                content,
                json.dumps(list(sources), ensure_ascii=False) if sources else None,
                now,
            ),
        )
        self.db.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
        self.db.commit()
        return cur.lastrowid

    def get_messages(self, session_id: int) -> list[dict]:
        """Every message in a session, oldest first, with sources restored."""
        rows = self.db.execute(
            """
            SELECT id, role, content, sources_json, created_at
            FROM messages WHERE session_id = ? ORDER BY id
            """,
            (session_id,),
        ).fetchall()
        return [
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "sources": json.loads(row["sources_json"]) if row["sources_json"] else [],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def message_count(self, session_id: int) -> int:
        return self.db.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()[0]

    def set_title(self, session_id: int, title: str) -> None:
        """Name a session. Does not count as activity, so ordering is unchanged."""
        self.db.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))
        self.db.commit()

    def delete_session(self, session_id: int) -> bool:
        """Delete a session and its messages. True if there was one to delete."""
        cur = self.db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self.db.commit()
        return cur.rowcount > 0

    def close(self) -> None:
        self.db.close()
