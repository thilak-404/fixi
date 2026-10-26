"""
End-to-End Demonstration Scenario Test (Section 19).
Verifies the complete vertical slice:
Person Recognition + Individual Memory + Personalized AI Conversation + Strict Memory Isolation.
"""

import numpy as np
import pytest
from app.memory.database import Database
from app.memory.memory_manager import MemoryManager
from app.users.user_manager import UserManager
from app.users.consent import ConsentStatus, CONSENT_REQUEST_PHRASE
from app.vision.camera import SyntheticCamera
from app.vision.person_detector import SimulatedPersonDetector, PersonDetection
from app.vision.face_recognition import SimulatedFaceRecognizer
from app.speech.stt import MockSTT
from app.speech.tts import MockTTS
from app.ai.llm import MockLLMProvider
from app.robotics.controller import SimulatedRobotController
from app.main import FixiEngine, FixiState


@pytest.fixture
def fixi_demo_environment(tmp_path):
    # Setup clean isolated database
    test_db = Database(db_path=tmp_path / "fixi_demo.db")

    # Inject db into managers
    user_mgr = UserManager()
    user_mgr.db = test_db
    mem_mgr = MemoryManager()
    mem_mgr.db = test_db

    # Setup mocked/simulated components for deterministic execution
    cam = SyntheticCamera()
    detector = SimulatedPersonDetector()
    face_rec = SimulatedFaceRecognizer()
    stt = MockSTT()
    tts = MockTTS()
    llm = MockLLMProvider()
    robot = SimulatedRobotController()

    engine = FixiEngine(
        camera=cam,
        person_detector=detector,
        face_recognizer=face_rec,
        stt=stt,
        tts=tts,
        llm=llm,
        robot=robot,
        user_mgr=user_mgr,
        mem_mgr=mem_mgr
    )
    # Redirect engine's managers to isolated db
    engine.responder.memory_manager = mem_mgr
    engine.responder.extractor.memory_manager = mem_mgr

    # Monkeypatch singleton references for this test run
    import app.main, app.users.user_manager, app.memory.memory_manager
    app.main.user_manager = user_mgr
    app.main.memory_manager = mem_mgr
    app.users.user_manager.user_manager = user_mgr
    app.memory.memory_manager.memory_manager = mem_mgr

    return {
        "engine": engine,
        "face_rec": face_rec,
        "detector": detector,
        "tts": tts,
        "mem_mgr": mem_mgr,
        "user_mgr": user_mgr
    }


def test_section_19_complete_demonstration_scenario(fixi_demo_environment):
    env = fixi_demo_environment
    engine: FixiEngine = env["engine"]
    face_rec: SimulatedFaceRecognizer = env["face_rec"]
    tts: MockTTS = env["tts"]
    mem_mgr: MemoryManager = env["mem_mgr"]
    user_mgr: UserManager = env["user_mgr"]

    # ==============================================================
    # STEP 1: First Interaction - Unknown Person A approaches Fixi
    # ==============================================================
    face_rec.set_unknown_person()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Vision detects unknown person
    v_res = engine.process_vision_frame(frame)
    assert engine.state == FixiState.AWAITING_CONSENT
    assert engine.current_user_id is None

    # Person A: "Yes." (Grants consent)
    r1 = engine.handle_user_speech("Yes")
    assert engine.state == FixiState.AWAITING_NAME
    assert "name" in r1["response"].lower()

    # Person A: "My name is Arun."
    r2 = engine.handle_user_speech("My name is Arun")
    assert engine.state == FixiState.CONVERSING
    assert engine.current_user_name == "Arun"
    arun_id = engine.current_user_id
    assert arun_id is not None
    assert "Arun" in r2["response"]

    # Person A: "I like football and I'm learning Python."
    r3 = engine.handle_user_speech("I like football and I'm learning Python.")
    assert engine.state == FixiState.CONVERSING

    # Verify Fixi stores: Name: Arun, Interest: Football, Learning: Python
    arun_facts = mem_mgr.get_user_facts(arun_id)
    assert arun_facts["name"] == "Arun"
    assert "football" in arun_facts.get("interest", "").lower()
    assert "python" in arun_facts.get("learning", "").lower()

    # ==============================================================
    # STEP 2: Second Interaction - Arun returns later
    # ==============================================================
    # Arun leaves and Fixi returns to IDLE
    face_rec.clear_faces()
    env["detector"].enabled = False
    engine.process_vision_frame(frame)
    assert engine.state == FixiState.IDLE
    assert engine.current_user_id is None

    # Arun returns! Fixi recognizes him via face recognition
    env["detector"].enabled = True
    face_rec.set_recognized_person(user_id=arun_id, name="Arun", confidence=0.96)
    v_res2 = engine.process_vision_frame(frame)

    assert engine.current_user_id == arun_id
    assert engine.current_user_name == "Arun"
    assert engine.state == FixiState.CONVERSING

    # Arun asks: "What do you remember about me?"
    r4 = engine.handle_user_speech("What do you remember about me?")
    reply = r4["response"]

    # Fixi accurately recalls football and Python!
    assert "football" in reply.lower()
    assert "python" in reply.lower()

    # ==============================================================
    # STEP 3: Different Person - Person B (Maya) approaches
    # ==============================================================
    # Register Person B as Maya
    maya = user_mgr.create_user("Maya", consent_status=ConsentStatus.GRANTED)
    maya_id = maya["id"]
    mem_mgr.set_user_fact(maya_id, "interest", "Painting")
    mem_mgr.set_user_fact(maya_id, "learning", "Rust")

    # Person B (Maya) appears in camera
    face_rec.set_recognized_person(user_id=maya_id, name="Maya", confidence=0.94)
    engine.process_vision_frame(frame)

    assert engine.current_user_id == maya_id
    assert engine.current_user_name == "Maya"

    # Maya asks: "What do you remember about me?"
    r5 = engine.handle_user_speech("What do you remember about me?")
    reply_maya = r5["response"]

    # CRITICAL VERIFICATION: Fixi MUST NOT mention Arun's information!
    assert "painting" in reply_maya.lower()
    assert "rust" in reply_maya.lower()
    assert "football" not in reply_maya.lower()
    assert "python" not in reply_maya.lower()
    assert "arun" not in reply_maya.lower()
