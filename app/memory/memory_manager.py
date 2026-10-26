"""
Memory Manager for Fixi.
Implements the 3-level memory architecture:
Level A: Short-Term Memory (recent conversation turns)
Level B: Episodic Memory (semantic retrieval of past interactions)
Level C: User Profile Memory (structured facts: name, interests, preferences)

Guarantees STRICT USER ISOLATION: No user's memories can ever be accessed without their user_id.
"""

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from app.memory.database import db
from app.memory.retrieval import retriever, serialize_vector, deserialize_vector
from app.config.settings import settings


class MemoryManager:
    def __init__(self):
        self.db = db
        self.retriever = retriever

    # ==========================================
    # LEVEL A: Short-Term Memory
    # ==========================================

    def add_short_term_turn(self, user_id: str, role: str, content: str) -> None:
        """Appends a single dialogue turn to user's short-term history."""
        if not user_id or not content.strip():
            return
        now = datetime.now(timezone.utc).isoformat()
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO short_term_memory (user_id, role, content, timestamp)
                VALUES (?, ?, ?, ?);
                """,
                (user_id, role, content.strip(), now)
            )

    def get_recent_conversation(self, user_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieves recent dialogue turns strictly for the given user_id."""
        if not user_id:
            return []
        max_turns = limit or settings.MAX_SHORT_TERM_TURNS
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                SELECT id, role, content, timestamp
                FROM short_term_memory
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?;
                """,
                (user_id, max_turns)
            )
            rows = cur.fetchall()
            # Return in chronological order
            history = [dict(r) for r in reversed(rows)]
            return history

    def clear_short_term_memory(self, user_id: str) -> int:
        """Clears short-term conversation history for a given user."""
        if not user_id:
            return 0
        with self.db.get_cursor() as cur:
            cur.execute(
                "DELETE FROM short_term_memory WHERE user_id = ?;",
                (user_id,)
            )
            return cur.rowcount

    # ==========================================
    # LEVEL B: Episodic Memory
    # ==========================================

    def add_episodic_memory(
        self,
        user_id: str,
        summary: str,
        details: str = "",
        category: str = "general",
        importance: float = 1.0
    ) -> int:
        """Stores a new episodic memory with its vector embedding."""
        if not user_id or not summary.strip():
            return -1

        text_to_embed = f"{summary.strip()} {details.strip()}"
        embedding_vec = self.retriever.embed(text_to_embed)
        embedding_blob = serialize_vector(embedding_vec)
        now = datetime.now(timezone.utc).isoformat()

        with self.db.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO episodic_memory (user_id, summary, details, category, importance, embedding, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (user_id, summary.strip(), details.strip(), category, importance, embedding_blob, now)
            )
            return cur.lastrowid

    def search_episodic_memories(
        self,
        user_id: str,
        query: str,
        top_k: Optional[int] = None,
        min_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top relevant episodic memories strictly for user_id
        using semantic embedding similarity ranking.
        """
        if not user_id or not query.strip():
            return []

        k = top_k or settings.TOP_K_EPISODIC_MEMORIES
        thresh = min_threshold if min_threshold is not None else settings.MEMORY_SIMILARITY_THRESHOLD
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                SELECT id, user_id, summary, details, category, importance, embedding, timestamp
                FROM episodic_memory
                WHERE user_id = ?
                ORDER BY id DESC;
                """,
                (user_id,)
            )
            rows = [dict(r) for r in cur.fetchall()]

        if not rows:
            return []

        ranked = self.retriever.rank_memories(
            query=query,
            memories=rows,
            top_k=k,
            min_threshold=thresh
        )
        return [mem for mem, score in ranked]

    def list_episodic_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """Lists all episodic memories for a user (excluding raw binary embedding)."""
        if not user_id:
            return []
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                SELECT id, user_id, summary, details, category, importance, timestamp
                FROM episodic_memory
                WHERE user_id = ?
                ORDER BY timestamp DESC;
                """,
                (user_id,)
            )
            return [dict(r) for r in cur.fetchall()]

    def delete_episodic_memory(self, user_id: str, memory_id: int) -> bool:
        """Deletes a specific episodic memory belonging to user_id."""
        with self.db.get_cursor() as cur:
            cur.execute(
                "DELETE FROM episodic_memory WHERE id = ? AND user_id = ?;",
                (memory_id, user_id)
            )
            return cur.rowcount > 0

    def edit_episodic_memory(self, user_id: str, memory_id: int, summary: str, details: str = "") -> bool:
        """Edits an existing episodic memory and recalculates its embedding."""
        text_to_embed = f"{summary.strip()} {details.strip()}"
        new_vec = self.retriever.embed(text_to_embed)
        blob = serialize_vector(new_vec)
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                UPDATE episodic_memory
                SET summary = ?, details = ?, embedding = ?
                WHERE id = ? AND user_id = ?;
                """,
                (summary.strip(), details.strip(), blob, memory_id, user_id)
            )
            return cur.rowcount > 0

    # ==========================================
    # LEVEL C: User Profile Memory (Stable Facts)
    # ==========================================

    def set_user_fact(self, user_id: str, key: str, value: str, category: str = "general") -> None:
        """Stores or updates a stable fact for the given user."""
        if not user_id or not key.strip() or not value.strip():
            return
        now = datetime.now(timezone.utc).isoformat()
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_facts (user_id, key, value, category, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, key) DO UPDATE SET
                    value = excluded.value,
                    category = excluded.category,
                    updated_at = excluded.updated_at;
                """,
                (user_id, key.strip().lower(), value.strip(), category, now)
            )

    def get_user_facts(self, user_id: str) -> Dict[str, str]:
        """Returns all stable facts for a user as a key-value dictionary."""
        if not user_id:
            return {}
        with self.db.get_cursor() as cur:
            cur.execute(
                "SELECT key, value FROM user_facts WHERE user_id = ? ORDER BY key ASC;",
                (user_id,)
            )
            rows = cur.fetchall()
            return {r["key"]: r["value"] for r in rows}

    def delete_user_fact(self, user_id: str, key: str) -> bool:
        """Deletes a specific fact for a user."""
        with self.db.get_cursor() as cur:
            cur.execute(
                "DELETE FROM user_facts WHERE user_id = ? AND key = ?;",
                (user_id, key.strip().lower())
            )
            return cur.rowcount > 0

    # ==========================================
    # Combined Context & Isolation Verification
    # ==========================================

    def get_full_user_context(self, user_id: str, current_query: str) -> Dict[str, Any]:
        """
        Retrieves the complete isolated context for generating a personalized response:
        - Profile Facts (Level C)
        - Relevant Episodic Memories (Level B)
        - Recent Conversation Turns (Level A)
        """
        facts = self.get_user_facts(user_id)
        episodic = self.search_episodic_memories(user_id, current_query)
        conversation = self.get_recent_conversation(user_id)

        return {
            "user_id": user_id,
            "facts": facts,
            "episodic_memories": episodic,
            "recent_turns": conversation
        }

    def clear_all_user_memories(self, user_id: str) -> None:
        """Deletes ALL memories (short-term, episodic, facts) for a user."""
        if not user_id:
            return
        with self.db.get_cursor() as cur:
            cur.execute("DELETE FROM short_term_memory WHERE user_id = ?;", (user_id,))
            cur.execute("DELETE FROM episodic_memory WHERE user_id = ?;", (user_id,))
            cur.execute("DELETE FROM user_facts WHERE user_id = ?;", (user_id,))

    def get_memory_counts(self, user_id: str) -> Dict[str, int]:
        """Returns counts of memories across all 3 levels for dashboard."""
        with self.db.get_cursor() as cur:
            cur.execute("SELECT COUNT(*) as c FROM short_term_memory WHERE user_id = ?;", (user_id,))
            st_count = cur.fetchone()["c"]
            cur.execute("SELECT COUNT(*) as c FROM episodic_memory WHERE user_id = ?;", (user_id,))
            ep_count = cur.fetchone()["c"]
            cur.execute("SELECT COUNT(*) as c FROM user_facts WHERE user_id = ?;", (user_id,))
            fact_count = cur.fetchone()["c"]
            return {
                "short_term": st_count,
                "episodic": ep_count,
                "facts": fact_count,
                "total": st_count + ep_count + fact_count
            }


memory_manager = MemoryManager()
