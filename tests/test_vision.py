"""
Tests for Fixi Vision Layer (Person Detection and Face Recognition).
"""

import cv2
import numpy as np
import pytest
from app.vision.person_detector import PersonDetection, HOGPersonDetector, SimulatedPersonDetector
from app.vision.face_recognition import FaceInfo, SimulatedFaceRecognizer
from app.vision.camera import SyntheticCamera


def test_person_detection_dataclass():
    det = PersonDetection(bbox=(100, 50, 200, 400), confidence=0.92, detection_id=1)
    assert det.center == (200, 250)
    assert det.area == 80000
    assert det.confidence == 0.92


def test_synthetic_camera():
    cam = SyntheticCamera(width=320, height=240)
    assert cam.start() is True
    assert cam.is_active() is True
    ok, frame = cam.read_frame()
    assert ok is True
    assert frame is not None
    assert frame.shape == (240, 320, 3)
    cam.stop()
    assert cam.is_active() is False


def test_hog_person_detector_on_frame():
    detector = HOGPersonDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Blank frame should not crash and return empty or few false positives
    dets = detector.detect(frame)
    assert isinstance(dets, list)


def test_simulated_face_recognition():
    rec = SimulatedFaceRecognizer()
    rec.set_recognized_person("user_001", "Arun", confidence=0.95)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    faces = rec.detect_and_recognize(frame)
    assert len(faces) == 1
    assert faces[0].is_recognized is True
    assert faces[0].matched_user_id == "user_001"
    assert faces[0].matched_name == "Arun"
    assert faces[0].similarity_score == 0.95

    # Switch to unknown person
    rec.set_unknown_person()
    faces_unknown = rec.detect_and_recognize(frame)
    assert len(faces_unknown) == 1
    assert faces_unknown[0].is_recognized is False
    assert faces_unknown[0].matched_user_id is None
