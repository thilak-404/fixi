"""Optional local object detection adapter for point-and-command targets."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from app.config.settings import settings
from app.robotics.target_selection import ObjectDetection


class LocalYOLOObjectDetector:
    """Lazy Ultralytics adapter; no network access is performed by this class."""

    def __init__(self, model_path: str = "models/vision/yolov8n.pt", confidence_threshold: float = 0.35):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        except Exception:
            self.model = None

    def detect(self, frame: np.ndarray) -> List[ObjectDetection]:
        if self.model is None or frame is None or frame.size == 0:
            return []
        try:
            results = self.model(frame, conf=self.confidence_threshold, imgsz=settings.YOLO_IMAGE_SIZE, verbose=False)
        except Exception:
            return []
        detections: List[ObjectDetection] = []
        if not results or results[0].boxes is None:
            return detections
        names = results[0].names
        for index, box in enumerate(results[0].boxes):
            coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
            x1, y1, x2, y2 = coords
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            detections.append(ObjectDetection(
                object_id=f"object-{index + 1}",
                label=str(names.get(class_id, class_id)),
                confidence=confidence,
                bbox=(x1, y1, max(1, x2 - x1), max(1, y2 - y1)),
            ))
        return detections