"""
Tests for Fixi Memory System and Strict Memory Isolation (Requirement 6, 7).
"""

import pytest
import numpy as np
from pathlib import Path
from app.memory.database import Database
from app.memory.retrieval import VectorRetriever, LocalFastEmbedder
from app.memory.memory_manager import MemoryManager


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_fixi.db"
    db_instance = Database(db_path=db_file)
    return db_instance


@pytest.fixture
def memory_mgr(test_db):
    mgr = MemoryManager()
    mgr.db = test_db
    return mgr


def test_database_initialization(test_db):
    with test_db.get_cursor() as cur:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cur.fetchall()]
        assert "users" in tables
        assert "short_term_memory" in tables
        assert "episodic_memory" in tables
        assert "user_facts" in tables


def test_strict_memory_isolation_user_a_and_b(test_db, memory_mgr):
    """
    CRITICAL REQUIREMENT #7: Memory Isolation.
    User A must never receive User B's private memories.
    """
    # 1. Setup User A (Arun) and User B (Maya)
    with test_db.get_cursor() as cur:
        cur.execute("INSERT INTO users (id, name, created_at, last_seen, consent_status) VALUES ('user_001', 'Arun', '2026-10-01', '2026-10-01', 'granted');")
        cur.execute("INSERT INTO users (id, name, created_at, last_seen, consent_status) VALUES ('user_002', 'Maya', '2026-10-01', '2026-10-01', 'granted');")

    # 2. Store Level C Facts
    memory_mgr.set_user_fact("user_001", "interest", "Football")
    memory_mgr.set_user_fact("user_001", "learning", "Python")

    memory_mgr.set_user_fact("user_002", "interest", "Robotics Painting")
    memory_mgr.set_user_fact("user_002", "learning", "Rust")

    # 3. Store Level B Episodic Memories
    memory_mgr.add_episodic_memory(
        user_id="user_001",
        summary="Arun discussed his upcoming football championship match.",
        details="Position: Midfielder"
    )
    memory_mgr.add_episodic_memory(
        user_id="user_002",
        summary="Maya discussed designing a robotic arm for painting.",
        details="Servo motor calibration"
    )

    # 4. Store Level A Short-Term Turns
    memory_mgr.add_short_term_turn("user_001", "user", "I scored two goals in football!")
    memory_mgr.add_short_term_turn("user_001", "fixi", "Congratulations Arun!")

    memory_mgr.add_short_term_turn("user_002", "user", "My robot painted a sunset.")
    memory_mgr.add_short_term_turn("user_002", "fixi", "That sounds artistic Maya!")

    # ===============================================
    # VERIFY USER A ISOLATION
    # ===============================================
    arun_context = memory_mgr.get_full_user_context("user_001", "What are my hobbies and projects?")
    assert arun_context["facts"]["interest"] == "Football"
    assert arun_context["facts"]["learning"] == "Python"
    assert "Rust" not in str(arun_context["facts"])
    assert "painting" not in str(arun_context).lower()

    # ===============================================
    # VERIFY USER B ISOLATION
    # ===============================================
    maya_context = memory_mgr.get_full_user_context("user_002", "What are my hobbies?")
    assert maya_context["facts"]["interest"] == "Robotics Painting"
    assert maya_context["facts"]["learning"] == "Rust"
    # STRICT ASSERTION: Arun's information must NEVER appear in Maya's context
    assert "Football" not in str(maya_context)
    assert "Python" not in str(maya_context)
    assert "Arun" not in str(maya_context)


def test_semantic_episodic_retrieval(test_db, memory_mgr):
    with test_db.get_cursor() as cur:
        cur.execute("INSERT INTO users (id, name, created_at, last_seen, consent_status) VALUES ('user_001', 'Arun', 'now', 'now', 'granted');")

    memory_mgr.add_episodic_memory("user_001", "User is building an autonomous drone project.")
    memory_mgr.add_episodic_memory("user_001", "User loves listening to jazz music while working.")
    memory_mgr.add_episodic_memory("user_001", "User ordered replacement propellers.")

    # Search for drone hardware
    results = memory_mgr.search_episodic_memories("user_001", "Tell me about my drone and parts", top_k=2)
    assert len(results) > 0
    summaries = [r["summary"] for r in results]
    assert any("drone" in s.lower() or "propeller" in s.lower() for s in summaries)


def test_memory_deletion_and_privacy_controls(test_db, memory_mgr):
    with test_db.get_cursor() as cur:
        cur.execute("INSERT INTO users (id, name, created_at, last_seen, consent_status) VALUES ('user_001', 'Arun', 'now', 'now', 'granted');")

    mem_id = memory_mgr.add_episodic_memory("user_001", "Secret test interaction")
    assert len(memory_mgr.list_episodic_memories("user_001")) == 1

    # Delete specific memory
    deleted = memory_mgr.delete_episodic_memory("user_001", mem_id)
    assert deleted is True
    assert len(memory_mgr.list_episodic_memories("user_001")) == 0

    # Add facts and clear all
    memory_mgr.set_user_fact("user_001", "color", "blue")
    memory_mgr.clear_all_user_mgr = memory_mgr.clear_all_user_memories("user_001")
    assert len(memory_mgr.get_user_facts("user_001")) == 0


def test_personalized_memory_extraction_and_query_flow(test_db, memory_mgr):
    from app.memory.extraction import MemoryExtractor
    from app.ai.response import ResponseGenerator
    from app.ai.llm import MockLLMProvider

    with test_db.get_cursor() as cur:
        cur.execute("INSERT INTO users (id, name, created_at, last_seen, consent_status) VALUES ('user_001', 'Thilak', 'now', 'now', 'granted');")

    extractor = MemoryExtractor(memory_manager=memory_mgr)
    llm = MockLLMProvider()
    resp_gen = ResponseGenerator(llm=llm)
    resp_gen.memory_manager = memory_mgr
    resp_gen.extractor = extractor

    # 1. User says age: "iam 19 years old"
    r1 = resp_gen.generate_response("user_001", "Thilak", "iam 19 years old")
    assert "19" in r1["response"]
    facts = memory_mgr.get_user_facts("user_001")
    assert facts.get("age") == "19"

    # 2. User asks age: "do you know my age?"
    r2 = resp_gen.generate_response("user_001", "Thilak", "do you know my age?")
    assert "19" in r2["response"]

    # 3. User says game with typo: "i like to play valrount"
    r3 = resp_gen.generate_response("user_001", "Thilak", "i like to play valrount")
    assert "Valorant" in r3["response"]
    facts = memory_mgr.get_user_facts("user_001")
    assert facts.get("game") == "Valorant"

    # 4. User asks games: "what games do i play?"
    r4 = resp_gen.generate_response("user_001", "Thilak", "what games do i play?")
    assert "Valorant" in r4["response"]

    # 5. User asks full summary: "what do you remember about me?"
    r5 = resp_gen.generate_response("user_001", "Thilak", "what do you remember about me?")
    assert "19" in r5["response"]
    assert "Valorant" in r5["response"]

    # 6. Casual greetings and questions must NOT store junk
    initial_fact_count = len(memory_mgr.get_user_facts("user_001"))
    initial_ep_count = len(memory_mgr.list_episodic_memories("user_001"))

    resp_gen.generate_response("user_001", "Thilak", "hello fixi")
    resp_gen.generate_response("user_001", "Thilak", "how are you today?")
    resp_gen.generate_response("user_001", "Thilak", "Hey, Fixie. I'm not doing a mess.")

    assert len(memory_mgr.get_user_facts("user_001")) == initial_fact_count
    assert len(memory_mgr.list_episodic_memories("user_001")) == initial_ep_count

