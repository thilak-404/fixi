import numpy as np

from app.robotics.pointing import GestureClass, PointingGestureRecognizer, landmarks_from_sequence


def make_hand(index_extended=True, other_extended=0):
    points = np.zeros((21, 2), dtype=np.float32)
    points[0] = (0, 0)
    for tip, pip, extended in ((8, 6, index_extended), (12, 10, other_extended >= 1), (16, 14, other_extended >= 2), (20, 18, other_extended >= 3)):
        points[pip] = (0, 10)
        points[tip] = (0, 30 if extended else 8)
    return points


def test_pointing_requires_temporal_stability():
    recognizer = PointingGestureRecognizer(stable_frames=3)
    hand = landmarks_from_sequence(make_hand(), confidence=0.95)

    first = recognizer.update(hand)
    second = recognizer.update(hand)
    third = recognizer.update(hand)

    assert first.gesture == GestureClass.POINTING
    assert first.confidence < 0.55
    assert second.confidence < 0.55
    assert third.gesture == GestureClass.POINTING
    assert third.confidence >= 0.55
    assert third.stable_frames == 3


def test_open_and_closed_hands_are_not_pointing():
    recognizer = PointingGestureRecognizer(stable_frames=1)

    assert recognizer.update(landmarks_from_sequence(make_hand(other_extended=3))).gesture == GestureClass.OPEN
    assert recognizer.update(landmarks_from_sequence(make_hand(index_extended=False))).gesture == GestureClass.CLOSED


def test_missing_hand_clears_confirmation():
    recognizer = PointingGestureRecognizer(stable_frames=2)
    hand = landmarks_from_sequence(make_hand())
    recognizer.update(hand)
    assert recognizer.update(hand).confidence >= 0.55

    missing = recognizer.update(None)
    assert missing.gesture == GestureClass.NONE
    assert missing.stable_frames == 0


def test_optional_mediapipe_tracker_degrades_without_model_package():
    from app.robotics.pointing import MediaPipeHandTracker

    tracker = MediaPipeHandTracker()
    assert isinstance(tracker.detect(np.zeros((32, 32, 3), dtype=np.uint8)), list)
