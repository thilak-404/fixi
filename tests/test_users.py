"""
Tests for Fixi User Management and Consent.
"""

from app.memory.database import Database
from app.users.user_manager import UserManager
from app.users.consent import ConsentStatus, parse_consent_response
from app.memory.memory_manager import MemoryManager


def test_consent_parsing():
    # Affirmatives
    assert parse_consent_response("Yes please") == (True, True)
    assert parse_consent_response("Sure, go ahead") == (True, True)
    assert parse_consent_response("Okay") == (True, True)

    # Negatives
    assert parse_consent_response("No thanks") == (True, False)
    assert parse_consent_response("Do not remember me") == (True, False)
    assert parse_consent_response("Never") == (True, False)

    # Unclear
    assert parse_consent_response("What time is it?")[0] is False


def test_user_creation_and_sequential_ids(tmp_path):
    db = Database(db_path=tmp_path / "users_test.db")
    mgr = UserManager()
    mgr.db = db

    u1 = mgr.create_user("Arun")
    assert u1["id"] == "user_001"
    assert u1["name"] == "Arun"

    u2 = mgr.create_user("Maya")
    assert u2["id"] == "user_002"
    assert u2["name"] == "Maya"


def test_user_deletion_cascades(tmp_path):
    db = Database(db_path=tmp_path / "delete_test.db")
    user_mgr = UserManager()
    user_mgr.db = db
    mem_mgr = MemoryManager()
    mem_mgr.db = db

    u = user_mgr.create_user("Arun")
    uid = u["id"]

    # Add memory and facts
    mem_mgr.set_user_fact(uid, "interest", "Football")
    mem_mgr.add_episodic_memory(uid, "Arun loves football")
    mem_mgr.add_short_term_turn(uid, "user", "Hi Fixi")

    assert len(mem_mgr.get_user_facts(uid)) == 1
    assert len(mem_mgr.list_episodic_memories(uid)) == 1
    assert len(mem_mgr.get_recent_conversation(uid)) == 1

    # Delete User
    deleted = user_mgr.delete_user(uid)
    assert deleted is True

    # Assert complete erasure
    assert user_mgr.get_user(uid) is None
    assert len(mem_mgr.get_user_facts(uid)) == 0
    assert len(mem_mgr.list_episodic_memories(uid)) == 0
    assert len(mem_mgr.get_recent_conversation(uid)) == 0
