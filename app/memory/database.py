"""
Local SQLite database management for Fixi.
Provides thread-safe access, strict user isolation, and automatic schema initialization.
"""

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional, List, Dict, Any
from app.config.settings import settings


class Database:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = str(db_path or settings.DATABASE_PATH)
        self._local = threading.local()
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                self.db_path,
                timeout=20.0,
                check_same_thread=False
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def get_cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def init_db(self) -> None:
        """Initializes tables and indexes for users, memories, and facts."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self.get_cursor() as cur:
            # Users table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                face_embedding BLOB,
                consent_status TEXT NOT NULL DEFAULT 'pending'
            );
            """)

            # Short term conversation memory
            cur.execute("""
            CREATE TABLE IF NOT EXISTS short_term_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """)

            # Episodic memory (key past interactions, projects, preferences)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS episodic_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                details TEXT,
                category TEXT DEFAULT 'general',
                importance REAL DEFAULT 1.0,
                embedding BLOB,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """)

            # Stable profile facts (key-value facts like name, interests, languages)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS user_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, key),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                message TEXT NOT NULL,
                due_at TEXT NOT NULL,
                delivered INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """)

            # Indexes for strict user isolation and fast lookup
            cur.execute("CREATE INDEX IF NOT EXISTS idx_short_term_user ON short_term_memory(user_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_episodic_user ON episodic_memory(user_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_facts_user ON user_facts(user_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_reminders_user_due ON reminders(user_id, due_at, delivered);")

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None


db = Database()
