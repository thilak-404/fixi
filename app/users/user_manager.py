"""
User Manager for Fixi.
Handles local user identity, sequential user ID generation (user_001, user_002, etc.),
isolated face embedding registration, last seen tracking, and complete user deletion.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from app.memory.database import db
from app.memory.retrieval import serialize_vector, deserialize_vector
from app.users.consent import ConsentStatus
from app.config.settings import settings


class UserManager:
    def __init__(self):
        self.db = db

    def generate_next_user_id(self) -> str:
        """Generates the next sequential user ID (e.g. user_001, user_002)."""
        with self.db.get_cursor() as cur:
            cur.execute("SELECT id FROM users ORDER BY id DESC LIMIT 1;")
            row = cur.fetchone()
            if not row or not row["id"]:
                return "user_001"
            last_id = row["id"]
            if last_id.startswith("user_"):
                try:
                    num = int(last_id.split("_")[1]) + 1
                    return f"user_{num:03d}"
                except ValueError:
                    pass
            return f"user_{int(datetime.now().timestamp())}"

    def create_user(
        self,
        name: str,
        face_embedding: Optional[np.ndarray] = None,
        consent_status: ConsentStatus = ConsentStatus.GRANTED,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Creates a new user profile with isolated storage."""
        uid = user_id or self.generate_next_user_id()
        now = datetime.now(timezone.utc).isoformat()
        emb_blob = serialize_vector(face_embedding) if face_embedding is not None else None

        with self.db.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (id, name, created_at, last_seen, face_embedding, consent_status)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (uid, name.strip(), now, now, emb_blob, consent_status.value)
            )

        return {
            "id": uid,
            "name": name.strip(),
            "created_at": now,
            "last_seen": now,
            "consent_status": consent_status.value
        }

    def get_user(self, user_id: str, include_embedding: bool = False) -> Optional[Dict[str, Any]]:
        """Fetches a single user profile by user_id."""
        if not user_id:
            return None
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                SELECT id, name, created_at, last_seen, face_embedding, consent_status
                FROM users
                WHERE id = ?;
                """,
                (user_id,)
            )
            row = cur.fetchone()
            if not row:
                return None
            data = dict(row)
            has_face = data.get("face_embedding") is not None
            data["has_face"] = has_face
            if include_embedding and has_face:
                data["face_embedding"] = deserialize_vector(data["face_embedding"])
            else:
                data.pop("face_embedding", None)
            return data

    def get_user_by_name(self, name: str, include_embedding: bool = False) -> Optional[Dict[str, Any]]:
        """Finds user profile by name (case-insensitive)."""
        with self.db.get_cursor() as cur:
            cur.execute(
                "SELECT id, name, created_at, last_seen, face_embedding, consent_status FROM users WHERE LOWER(name) = LOWER(?) LIMIT 1;",
                (name.strip(),)
            )
            row = cur.fetchone()
            if not row:
                return None
            data = dict(row)
            has_face = data.get("face_embedding") is not None
            data["has_face"] = has_face
            if include_embedding and has_face:
                data["face_embedding"] = deserialize_vector(data["face_embedding"])
            else:
                data.pop("face_embedding", None)
            return data

    def update_last_seen(self, user_id: str) -> None:
        """Updates the last_seen timestamp for a user."""
        now = datetime.now(timezone.utc).isoformat()
        with self.db.get_cursor() as cur:
            cur.execute(
                "UPDATE users SET last_seen = ? WHERE id = ?;",
                (now, user_id)
            )

    def update_face_embedding(self, user_id: str, face_embedding: np.ndarray) -> None:
        """Updates the stored face embedding for a user."""
        emb_blob = serialize_vector(face_embedding)
        with self.db.get_cursor() as cur:
            cur.execute(
                "UPDATE users SET face_embedding = ? WHERE id = ?;",
                (emb_blob, user_id)
            )

    def find_matching_user(
        self,
        query_embedding: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Compares query face embedding against all enrolled users using cosine similarity.
        Returns (matched_user_dict, similarity_score).
        If highest similarity < threshold, returns (None, score).
        """
        match_thresh = threshold or settings.FACE_MATCH_THRESHOLD
        best_user: Optional[Dict[str, Any]] = None
        best_score = 0.0

        q_norm = np.linalg.norm(query_embedding)
        if q_norm < 1e-6:
            return None, 0.0
        q_unit = query_embedding / q_norm

        with self.db.get_cursor() as cur:
            cur.execute("SELECT id, name, created_at, last_seen, face_embedding, consent_status FROM users WHERE face_embedding IS NOT NULL;")
            rows = cur.fetchall()

        for row in rows:
            emb_blob = row["face_embedding"]
            emb = deserialize_vector(emb_blob)
            if emb is None:
                continue

            emb_norm = np.linalg.norm(emb)
            if emb_norm < 1e-6:
                continue
            emb_unit = emb / emb_norm

            sim = float(np.dot(q_unit, emb_unit))
            if sim > best_score:
                best_score = sim
                if sim >= match_thresh:
                    best_user = {
                        "id": row["id"],
                        "name": row["name"],
                        "created_at": row["created_at"],
                        "last_seen": row["last_seen"],
                        "consent_status": row["consent_status"]
                    }

        return best_user, best_score

    def list_users(self) -> List[Dict[str, Any]]:
        """Lists all enrolled users with statistics for the dashboard."""
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                SELECT u.id, u.name, u.created_at, u.last_seen, u.consent_status,
                       (SELECT COUNT(*) FROM short_term_memory WHERE user_id = u.id) as short_term_count,
                       (SELECT COUNT(*) FROM episodic_memory WHERE user_id = u.id) as episodic_count,
                       (SELECT COUNT(*) FROM user_facts WHERE user_id = u.id) as fact_count
                FROM users u
                ORDER BY u.created_at DESC;
                """
            )
            rows = cur.fetchall()
            result = []
            for r in rows:
                item = dict(r)
                item["total_memories"] = item["short_term_count"] + item["episodic_count"] + item["fact_count"]
                item["has_face"] = Path(settings.FACES_PATH / f"{item['id']}.jpg").exists()
                result.append(item)
            return result

    def delete_user(self, user_id: str) -> bool:
        """
        Permanently deletes a user and ALL associated data:
        - SQLite profile row (cascades short_term_memory, episodic_memory, user_facts)
        - Saved face images on disk
        - Saved conversation export files
        """
        if not user_id:
            return False

        with self.db.get_cursor() as cur:
            cur.execute("DELETE FROM users WHERE id = ?;", (user_id,))
            deleted = cur.rowcount > 0

        # Remove face image on disk if present
        face_img = settings.FACES_PATH / f"{user_id}.jpg"
        if face_img.exists():
            try:
                face_img.unlink()
            except OSError:
                pass

        # Remove conversation history file if present
        conv_file = settings.CONVERSATIONS_PATH / f"{user_id}.json"
        if conv_file.exists():
            try:
                conv_file.unlink()
            except OSError:
                pass

        return deleted


user_manager = UserManager()
