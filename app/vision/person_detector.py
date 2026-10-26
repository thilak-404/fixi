"""
Computer Vision Person Detection and Tracking Module for Fixi.
Implements:
1. Local Ultralytics YOLOv8n Person Detection.
2. Multi-Person Tracking across video frames with stable tracking IDs.
3. Ground-plane spatial position estimation (pixel -> world x,y coordinates).
4. Real-time inference latency and FPS profiling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import cv2
import numpy as np
import time

from app.config.settings import settings
from app.events import event_bus, EventType


@dataclass
class PersonDetection:
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    detection_id: int
    tracking_id: int = 1
    spatial_x: float = 0.0  # Lateral position in meters (-left / +right)
    spatial_y: float = 2.0  # Forward distance in meters
    grid_x: int = 10        # Occupancy grid X coordinate
    grid_y: int = 15        # Occupancy grid Y coordinate
    label: str = "person"
    latency_ms: float = 0.0

    @property
    def center(self) -> Tuple[int, int]:
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)

    @property
    def bottom_center(self) -> Tuple[int, int]:
        x, y, w, h = self.bbox
        return (x + w // 2, y + h)

    @property
    def area(self) -> int:
        return self.bbox[2] * self.bbox[3]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "confidence": round(self.confidence, 3),
            "tracking_id": self.tracking_id,
            "spatial_x": round(self.spatial_x, 2),
            "spatial_y": round(self.spatial_y, 2),
            "grid_x": self.grid_x,
            "grid_y": self.grid_y,
            "latency_ms": round(self.latency_ms, 1)
        }


def estimate_spatial_position(
    bbox: Tuple[int, int, int, int],
    frame_width: int = 640,
    frame_height: int = 480,
    grid_cols: int = 20,
    grid_rows: int = 20
) -> Tuple[float, float, int, int]:
    """
    Estimates 2D ground-plane coordinates (lateral x, forward distance y in meters)
    and maps them to discrete occupancy grid cells (grid_x, grid_y).
    """
    x, y, w, h = bbox
    cx = x + w / 2.0

    # Pinhole projection approximation
    # Assumes average person height ~ 1.7m, calibrated focal length ~ 580px
    focal_length = max(frame_width, frame_height) * 0.9
    real_height = 1.7
    pixel_height = max(15, h)
    
    # Forward distance (y) in meters
    distance = (focal_length * real_height) / pixel_height
    distance = float(np.clip(distance, 0.6, 6.0))

    # Lateral displacement (x) in meters
    norm_x = (cx - frame_width / 2.0) / (frame_width / 2.0)
    lateral_x = norm_x * distance * 0.55

    # Map to 2D occupancy grid:
    # Robot is at bottom-center (col = grid_cols // 2, row = 0)
    # Forward is +row, lateral is col
    mid_col = grid_cols // 2
    grid_col = int(np.clip(mid_col + int(lateral_x * 3.0), 0, grid_cols - 1))
    grid_row = int(np.clip(int(distance * 3.0), 1, grid_rows - 1))

    return round(lateral_x, 2), round(distance, 2), grid_col, grid_row


class BasePersonDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        pass


class YOLOPersonDetector(BasePersonDetector):
    """
    Ultralytics YOLOv8n Person Detector with ByteTrack multi-object tracking.
    Runs 100% locally with real-time FPS and latency profiling.
    """
    def __init__(
        self,
        model_path: str = "models/vision/yolov8n.pt",
        confidence_threshold: float = 0.45,
        enable_tracking: bool = True
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.enable_tracking = enable_tracking
        self.model = None
        self._last_process_time = 0.0
        self._last_detections: List[PersonDetection] = []
        self._latency_ms = 0.0
        self._fps = 0.0
        self._frame_count = 0
        self._start_time = time.time()
        self._init_model()

    def _init_model(self):
        try:
            from ultralytics import YOLO
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = YOLO(self.model_path)
            self.device = device
            print(f"YOLO Person Detector initialized successfully on {device.upper()}.")
        except Exception as e:
            print(f"Notice: Failed to load YOLO model ({e}). Using HOG fallback.")
            self.model = None

    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        if frame is None or frame.size == 0:
            return []

        # Fallback if YOLO model is not loaded
        if self.model is None:
            return self._detect_hog_fallback(frame)

        t_start = time.perf_counter()
        h, w = frame.shape[:2]

        try:
            # Detect or Track with ByteTrack
            if self.enable_tracking:
                results = self.model.track(
                    source=frame,
                    classes=[0],  # Person class only
                    conf=self.confidence_threshold,
                    imgsz=settings.YOLO_IMAGE_SIZE,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False
                )
            else:
                results = self.model(
                    source=frame,
                    classes=[0],
                    conf=self.confidence_threshold,
                    imgsz=settings.YOLO_IMAGE_SIZE,
                    verbose=False
                )

            t_end = time.perf_counter()
            self._latency_ms = (t_end - t_start) * 1000.0

            detections: List[PersonDetection] = []
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for idx, box in enumerate(boxes):
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    bw = max(10, x2 - x1)
                    bh = max(10, y2 - y1)
                    conf = float(box.conf[0].cpu().numpy()) if box.conf is not None else 0.8
                    
                    # Extract persistent tracking ID from ByteTrack
                    track_id = int(box.id[0].cpu().numpy()) if (box.id is not None) else (idx + 1)

                    sp_x, sp_y, gx, gy = estimate_spatial_position((x1, y1, bw, bh), w, h)

                    det = PersonDetection(
                        bbox=(x1, y1, bw, bh),
                        confidence=round(conf, 2),
                        detection_id=idx + 1,
                        tracking_id=track_id,
                        spatial_x=sp_x,
                        spatial_y=sp_y,
                        grid_x=gx,
                        grid_y=gy,
                        latency_ms=self._latency_ms
                    )
                    detections.append(det)

            # Sort closest/largest person first
            detections.sort(key=lambda d: d.area, reverse=True)
            self._last_detections = detections

            # FPS profiling
            self._frame_count += 1
            now = time.time()
            elapsed = now - self._start_time
            if elapsed >= 1.0:
                self._fps = round(self._frame_count / elapsed, 1)
                self._frame_count = 0
                self._start_time = now

            # Broadcast typed events
            if detections:
                event_bus.emit(
                    EventType.PERSON_DETECTED,
                    {
                        "count": len(detections),
                        "primary_tracking_id": detections[0].tracking_id,
                        "spatial_coords": [detections[0].spatial_x, detections[0].spatial_y],
                        "latency_ms": round(self._latency_ms, 1),
                        "fps": self._fps
                    }
                )

            return detections

        except Exception as e:
            print(f"YOLO detection exception: {e}")
            return self._detect_hog_fallback(frame)

    def _detect_hog_fallback(self, frame: np.ndarray) -> List[PersonDetection]:
        # Fast fallback when YOLO is unavailable
        h, w = frame.shape[:2]
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        rects, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(4, 4), scale=1.05)
        detections = []
        for i, (rx, ry, rw, rh) in enumerate(rects):
            sp_x, sp_y, gx, gy = estimate_spatial_position((rx, ry, rw, rh), w, h)
            detections.append(PersonDetection(
                bbox=(int(rx), int(ry), int(rw), int(rh)),
                confidence=0.75,
                detection_id=i + 1,
                tracking_id=i + 1,
                spatial_x=sp_x,
                spatial_y=sp_y,
                grid_x=gx,
                grid_y=gy
            ))
        self._last_detections = detections
        return detections

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "model_name": "YOLOv8n (Local Ultralytics)",
            "device": getattr(self, "device", "cpu"),
            "confidence_threshold": self.confidence_threshold,
            "tracking_enabled": self.enable_tracking,
            "latency_ms": round(self._latency_ms, 1),
            "fps": self._fps,
            "active_detections": len(self._last_detections),
            "detections": [d.to_dict() for d in self._last_detections]
        }


# Backward compatibility alias
HOGPersonDetector = YOLOPersonDetector


class SimulatedPersonDetector(BasePersonDetector):
    """Deterministic simulated person detector for headless testing."""
    def __init__(self, default_bbox: Tuple[int, int, int, int] = (160, 60, 320, 360)):
        self.default_bbox = default_bbox
        self.enabled = True

    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        if not self.enabled:
            return []
        return [PersonDetection(
            bbox=self.default_bbox,
            confidence=0.96,
            detection_id=1,
            tracking_id=1,
            spatial_x=0.0,
            spatial_y=1.8,
            grid_x=10,
            grid_y=5,
            latency_ms=12.4
        )]

    def get_metrics(self) -> Dict[str, Any]:
        return {"model_name": "SimulatedDetector", "fps": 30.0, "latency_ms": 10.0}
