from datetime import datetime, timedelta, timezone

from app.memory.database import Database
from app.memory.reminders import ReminderManager
from app.system_control import execute_system_command, parse_reminder
from app.users.consent import ConsentStatus
from app.users.user_manager import UserManager


def test_parse_relative_reminder():
    parsed = parse_reminder("Remind me to submit the report in 10 minutes")

    assert parsed is not None
    message, due_at = parsed
    assert message == "submit the report"
    assert due_at > datetime.now(timezone.utc)


def test_reminders_are_isolated_and_claimed_once(tmp_path):
    database = Database(tmp_path / "reminders.db")
    user_manager = UserManager()
    user_manager.db = database
    first = user_manager.create_user("Arun", consent_status=ConsentStatus.GRANTED)
    second = user_manager.create_user("Maya", consent_status=ConsentStatus.GRANTED)
    manager = ReminderManager()
    manager.db = database

    manager.add(first["id"], "drink water", datetime.now(timezone.utc) - timedelta(minutes=1))
    manager.add(second["id"], "check email", datetime.now(timezone.utc) - timedelta(minutes=1))

    first_due = manager.claim_due(first["id"])
    assert [item["message"] for item in first_due] == ["drink water"]
    assert manager.claim_due(first["id"]) == []
    assert [item["message"] for item in manager.list_for_user(second["id"])] == ["check email"]


def test_unknown_system_command_is_rejected_without_execution():
    result = execute_system_command("delete all my files", "user_001")

    assert result.handled is False
    assert result.action is None
