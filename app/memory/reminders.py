"""Per-user local reminders delivered when the user's face is recognized."""

from datetime import datetime, timezone
from typing import Any, Dict, List

from app.memory.database import db


class ReminderManager:
    def __init__(self):
        self.db = db

    def add(self, user_id: str, message: str, due_at: datetime) -> int:
        now = datetime.now(timezone.utc).isoformat()
        due = due_at.astimezone(timezone.utc).isoformat()
        with self.db.get_cursor() as cur:
            cur.execute(
                "INSERT INTO reminders (user_id, message, due_at, created_at) VALUES (?, ?, ?, ?)",
                (user_id, message.strip(), due, now),
            )
            return int(cur.lastrowid)

    def list_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        with self.db.get_cursor() as cur:
            cur.execute(
                "SELECT id, user_id, message, due_at, delivered, created_at FROM reminders WHERE user_id = ? ORDER BY due_at ASC",
                (user_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    def claim_due(self, user_id: str) -> List[Dict[str, Any]]:
        now = datetime.now(timezone.utc).isoformat()
        with self.db.get_cursor() as cur:
            cur.execute(
                "SELECT id, message, due_at FROM reminders WHERE user_id = ? AND delivered = 0 AND due_at <= ? ORDER BY due_at ASC",
                (user_id, now),
            )
            rows = [dict(row) for row in cur.fetchall()]
            for row in rows:
                cur.execute("UPDATE reminders SET delivered = 1 WHERE id = ? AND user_id = ?", (row["id"], user_id))
            return rows


reminder_manager = ReminderManager()
