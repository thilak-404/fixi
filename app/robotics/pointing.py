"""Local pointing gesture recognition contracts and temporal filtering."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


class GestureClass(str, Enum):
    POINTING = "pointing"
    OPEN = "open"
    CLOSED = "closed"
    AMBIGUOUS = "ambiguous"
    NONE = "none"


@dataclass(frozen=True)
class HandLandmarks:
    points: np.ndarray
    handedness: Optional[str] = None
    confidence: float = 0.0

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float32)
        if points.shape[0] < 21 or points.shape[1] < 2:
            raise ValueError("Hand landmarks must contain at least 21 x/y points")
        object.__setattr__(self, "points", points[:, :2])


@dataclass(frozen=True)
class GestureObservation:
    gesture: GestureClass
    confidence: float
    direction: Optional[Tuple[float, float]]
    fingertip: Optional[Tuple[float, float]]
    stable_frames: int
    hand_confidence: float


class PointingGestureRecognizer:
    """Classifies one hand and confirms pointing only after temporal stability."""

    def __init__(self, stable_frames: int = 4, history_size: int = 8, min_confidence: float = 0.55):
        if stable_frames < 1:
            raise ValueError("stable_frames must be positive")
        self.stable_frames = stable_frames
        self.min_confidence = min_confidence
        self._history: deque[GestureClass] = deque(maxlen=max(stable_frames, history_size))
        self._last_observation = GestureObservation(GestureClass.NONE, 0.0, None, None, 0, 0.0)

    @staticmethod
    def _distance(points: np.ndarray, first: int, second: int) -> float:
        return float(np.linalg.norm(points[first] - points[second]))

    @classmethod
    def _finger_extended(cls, points: np.ndarray, tip: int, pip: int, wrist: int = 0) -> bool:
        return cls._distance(points, wrist, tip) > cls._distance(points, wrist, pip) * 1.12

    @classmethod
    def classify_landmarks(cls, hand: HandLandmarks) -> Tuple[GestureClass, float, Tuple[float, float]]:
        points = hand.points
        index_extended = cls._finger_extended(points, 8, 6)
        middle_extended = cls._finger_extended(points, 12, 10)
        ring_extended = cls._finger_extended(points, 16, 14)
        pinky_extended = cls._finger_extended(points, 20, 18)
        extended = sum((index_extended, middle_extended, ring_extended, pinky_extended))

        direction_vector = points[8] - points[6]
        norm = float(np.linalg.norm(direction_vector))
        direction = (0.0, 0.0) if norm < 1e-6 else tuple((direction_vector / norm).tolist())

        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            confidence = min(1.0, 0.55 + float(np.linalg.norm(direction_vector)) / 200.0)
            return GestureClass.POINTING, confidence, direction
        if extended >= 3:
            return GestureClass.OPEN, 0.85, direction
        if extended == 0:
            return GestureClass.CLOSED, 0.85, direction
        return GestureClass.AMBIGUOUS, 0.45, direction

    def update(self, hand: Optional[HandLandmarks]) -> GestureObservation:
        if hand is None or hand.confidence < self.min_confidence:
            self._history.clear()
            self._last_observation = GestureObservation(GestureClass.NONE, 0.0, None, None, 0, 0.0)
            return self._last_observation

        gesture, confidence, direction = self.classify_landmarks(hand)
        self._history.append(gesture)
        stable = sum(item == gesture for item in self._history)
        confirmed = gesture if gesture == GestureClass.POINTING and stable >= self.stable_frames else gesture
        fingertip = tuple(hand.points[8].tolist())
        self._last_observation = GestureObservation(
            gesture=confirmed,
            confidence=confidence if confirmed == GestureClass.POINTING and stable >= self.stable_frames else min(confidence, 0.49) if gesture == GestureClass.POINTING else confidence,
            direction=direction,
            fingertip=fingertip,
            stable_frames=stable,
            hand_confidence=hand.confidence,
        )
        return self._last_observation

    @property
    def last_observation(self) -> GestureObservation:
        return self._last_observation


class MediaPipeHandTracker:
    """Optional local MediaPipe adapter returning normalized hand landmarks."""

    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.6, min_tracking_confidence: float = 0.6):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._hands = None
        try:
            import mediapipe as mp
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        except Exception:
            self._hands = None

    def detect(self, frame: np.ndarray) -> list[HandLandmarks]:
        if self._hands is None or frame is None or frame.size == 0:
            return []
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return []
        hands = []
        for index, landmarks in enumerate(result.multi_hand_landmarks):
            points = np.array([(point.x, point.y, point.z) for point in landmarks.landmark], dtype=np.float32)
            handedness = None
            if result.multi_handedness and index < len(result.multi_handedness):
                handedness = result.multi_handedness[index].classification[0].label
            hands.append(HandLandmarks(points, handedness=handedness, confidence=1.0))
        return hands


def landmarks_from_sequence(points: Sequence[Sequence[float]], confidence: float = 1.0) -> HandLandmarks:
    return HandLandmarks(np.asarray(points, dtype=np.float32), confidence=confidence)
