"""Image-space target selection for point-and-command interactions."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Dict, Iterable, List, Optional, Tuple

from app.robotics.pointing import GestureObservation, GestureClass


@dataclass(frozen=True)
class ObjectDetection:
    object_id: str
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    tracking_status: str = "tracked"

    @property
    def center(self) -> Tuple[float, float]:
        x, y, width, height = self.bbox
        return x + width / 2.0, y + height / 2.0


@dataclass(frozen=True)
class TargetCandidate:
    detection: ObjectDetection
    score: float
    line_distance: float
    temporal_score: float


@dataclass(frozen=True)
class TargetSelection:
    status: str
    selected: Optional[TargetCandidate]
    candidates: Tuple[TargetCandidate, ...]
    reason: str


class PointingTargetSelector:
    """Selects a target only when gesture and image-space evidence agree."""

    def __init__(self, max_line_distance: float = 90.0, ambiguity_margin: float = 0.2, min_score: float = 0.35):
        self.max_line_distance = max_line_distance
        self.ambiguity_margin = ambiguity_margin
        self.min_score = min_score
        self._history: Dict[str, int] = {}

    @staticmethod
    def _line_distance(origin: Tuple[float, float], direction: Tuple[float, float], point: Tuple[float, float]) -> Tuple[float, float]:
        ox, oy = origin
        dx, dy = direction
        px, py = point
        projection = (px - ox) * dx + (py - oy) * dy
        closest_x = ox + max(0.0, projection) * dx
        closest_y = oy + max(0.0, projection) * dy
        return hypot(px - closest_x, py - closest_y), projection

    def _score(self, gesture: GestureObservation, detection: ObjectDetection) -> Optional[TargetCandidate]:
        if gesture.gesture != GestureClass.POINTING or gesture.confidence < 0.55 or gesture.direction is None or gesture.fingertip is None:
            return None
        distance, projection = self._line_distance(gesture.fingertip, gesture.direction, detection.center)
        if projection <= 0 or distance > self.max_line_distance:
            return None
        line_score = max(0.0, 1.0 - distance / self.max_line_distance)
        confidence_score = max(0.0, min(1.0, detection.confidence))
        persistence = min(1.0, self._history.get(detection.object_id, 0) / 3.0)
        score = 0.55 * line_score + 0.25 * confidence_score + 0.10 * gesture.confidence + 0.10 * persistence
        return TargetCandidate(detection, score, distance, persistence)

    def select(self, gesture: GestureObservation, detections: Iterable[ObjectDetection]) -> TargetSelection:
        scored: List[TargetCandidate] = []
        seen_ids = set()
        for detection in detections:
            candidate = self._score(gesture, detection)
            if candidate is not None:
                scored.append(candidate)
                seen_ids.add(detection.object_id)
        for object_id in list(self._history):
            if object_id not in seen_ids:
                self._history[object_id] = max(0, self._history[object_id] - 1)
        scored.sort(key=lambda item: item.score, reverse=True)

        if not scored:
            self._history.clear()
            return TargetSelection("none", None, tuple(), "No object lies on the pointing ray")

        best = scored[0]
        self._history[best.detection.object_id] = self._history.get(best.detection.object_id, 0) + 1
        if best.score < self.min_score:
            return TargetSelection("none", None, tuple(scored), "Pointing evidence is below the target threshold")
        if len(scored) > 1 and best.score - scored[1].score < self.ambiguity_margin:
            return TargetSelection("ambiguous", None, tuple(scored[:2]), "Multiple objects are plausible targets")
        return TargetSelection("selected", best, tuple(scored), "Target selected from stable pointing evidence")
