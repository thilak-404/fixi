"""Allowlisted local system commands for authenticated Fixi users."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.memory.reminders import reminder_manager


@dataclass(frozen=True)
class SystemCommandResult:
    handled: bool
    message: str
    action: Optional[str] = None
    reminder_id: Optional[int] = None


def _open_program(candidates: list[str], args: list[str] | None = None) -> bool:
    for candidate in candidates:
        executable = shutil.which(candidate)
        if executable:
            subprocess.Popen([executable, *(args or [])], start_new_session=True)
            return True
    return False


def parse_reminder(text: str) -> Optional[tuple[str, datetime]]:
    match = re.search(r"\bremind me to\s+(.+?)\s+in\s+(\d+)\s+minutes?\b", text, re.IGNORECASE)
    if match:
        return match.group(1).strip(" ."), datetime.now(timezone.utc) + timedelta(minutes=int(match.group(2)))
    match = re.search(r"\bremind me to\s+(.+?)\s+at\s+(\d{1,2}):(\d{2})\b", text, re.IGNORECASE)
    if match:
        hour, minute = int(match.group(2)), int(match.group(3))
        now = datetime.now().astimezone()
        due = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if due <= now:
            due += timedelta(days=1)
        return match.group(1).strip(" ."), due
    return None


def execute_system_command(text: str, user_id: str) -> SystemCommandResult:
    lower = text.strip().lower()
    reminder = parse_reminder(text)
    if reminder:
        message, due_at = reminder
        reminder_id = reminder_manager.add(user_id, message, due_at)
        return SystemCommandResult(True, f"I will remind you to {message} at {due_at.astimezone().strftime('%H:%M')}.", "set_reminder", reminder_id)

    if re.search(r"\b(?:open|launch|start)\s+(?:the\s+)?(?:browser|web browser)\b", lower):
        ok = _open_program(["xdg-open"], ["http://localhost:8000/app/"])
        return SystemCommandResult(True, "Opening the local Fixi dashboard in your browser." if ok else "I couldn't find a local browser opener.", "open_browser")

    if re.search(r"\b(?:open|launch|start)\s+(?:the\s+)?terminal\b", lower):
        ok = _open_program(["x-terminal-emulator", "gnome-terminal", "konsole", "xfce4-terminal"])
        return SystemCommandResult(True, "Opening a terminal." if ok else "I couldn't find a supported terminal application.", "open_terminal")

    return SystemCommandResult(False, "I can open the browser, open a terminal, or set a reminder. Please use one of those commands.")
