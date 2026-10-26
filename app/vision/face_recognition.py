"""
Face Recognition module for Fixi.
Implements 100% local face detection and embedding extraction using OpenCV YuNet + SFace.
Generates 128-dimensional L2-normalized embeddings for cosine similarity matching.
Strictly local: face images and embeddings never leave this device.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import cv2
import numpy as np
from app.config.settings import settings
from app.users.user_manager import user_manager


@dataclass
class FaceInfo:
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    landmarks: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    matched_user_id: Optional[str] = None
    matched_name: Optional[str] = None
    similarity_score: float = 0.0
    is_recognized: bool = False


class BaseFaceRecognizer(ABC):
    @abstractmethod
    def detect_and_recognize(self, frame: np.ndarray) -> List[FaceInfo]:
        """Detects faces in frame, computes embeddings, and matches against known users."""
        pass

    @abstractmethod
    def extract_embedding_from_crop(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """Extracts face embedding from an isolated face image."""
        pass


class OpenCVYuNetSFaceRecognizer(BaseFaceRecognizer):
    """
    State-of-the-art local face recognition using OpenCV Zoo YuNet (detector) + SFace (recognizer).
    Runs natively in OpenCV DNN on CPU or CUDA without external framework overhead.
    """
    def __init__(
        self,
        yunet_path: Optional[Path] = None,
        sface_path: Optional[Path] = None,
        match_threshold: float = settings.FACE_MATCH_THRESHOLD
    ):
        self.yunet_path = str(yunet_path or settings.YUNET_MODEL_PATH)
        self.sface_path = str(sface_path or settings.SFACE_MODEL_PATH)
        self.match_threshold = match_threshold

        self.detector: Optional[cv2.FaceDetectorYN] = None
        self.recognizer: Optional[cv2.FaceRecognizerSF] = None
        self._init_models()

    def _init_models(self) -> None:
        if Path(self.yunet_path).exists() and Path(self.sface_path).exists():
            try:
                self.detector = cv2.FaceDetectorYN.create(
                    model=self.yunet_path,
                    config="",
                    input_size=(settings.CAMERA_WIDTH, settings.CAMERA_HEIGHT),
                    score_threshold=0.45,
                    nms_threshold=0.3,
                    top_k=5000
                )
                self.recognizer = cv2.FaceRecognizerSF.create(
                    model=self.sface_path,
                    config=""
                )
            except Exception as e:
                print(f"Warning: Failed to load YuNet/SFace models: {e}")

    def is_ready(self) -> bool:
        return self.detector is not None and self.recognizer is not None

    def detect_and_recognize(self, frame: np.ndarray) -> List[FaceInfo]:
        if not self.is_ready() or frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))

        # Detect faces: returns (code, faces_array)
        _, faces = self.detector.detect(frame)
        if faces is None or len(faces) == 0:
            return []

        results: List[FaceInfo] = []
        for face in faces:
            # Face format in YuNet: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
            x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            score = float(face[-1])

            # Clamp coordinates to frame boundaries
            x = max(0, x)
            y = max(0, y)
            fw = min(w - x, fw)
            fh = min(h - y, fh)
            if fw <= 10 or fh <= 10:
                continue

            # Align face & extract 128D embedding using SFace
            aligned_face = self.recognizer.alignCrop(frame, face)
            raw_feature = self.recognizer.feature(aligned_face)
            embedding = raw_feature.flatten().astype(np.float32)

            # L2-normalize
            norm = np.linalg.norm(embedding)
            if norm > 1e-6:
                embedding = embedding / norm

            # Match against database
            matched_user, sim_score = user_manager.find_matching_user(embedding, threshold=self.match_threshold)

            face_info = FaceInfo(
                bbox=(x, y, fw, fh),
                confidence=round(score, 3),
                landmarks=face[4:14].copy(),
                embedding=embedding,
                matched_user_id=matched_user["id"] if matched_user else None,
                matched_name=matched_user["name"] if matched_user else None,
                similarity_score=round(sim_score, 3),
                is_recognized=matched_user is not None
            )
            results.append(face_info)

        return results

    def extract_embedding_from_crop(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """Extracts embedding from an already cropped face."""
        if not self.is_ready() or face_crop is None or face_crop.size == 0:
            return None
        h, w = face_crop.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(face_crop)
        if faces is None or len(faces) == 0:
            # Synthetic fallback: resize directly to 112x112
            resized = cv2.resize(face_crop, (112, 112))
            feature = self.recognizer.feature(resized).flatten().astype(np.float32)
            norm = np.linalg.norm(feature)
            return (feature / norm) if norm > 1e-6 else feature

        face = faces[0]
        aligned = self.recognizer.alignCrop(face_crop, face)
        feature = self.recognizer.feature(aligned).flatten().astype(np.float32)
        norm = np.linalg.norm(feature)
        return (feature / norm) if norm > 1e-6 else feature


class SimulatedFaceRecognizer(BaseFaceRecognizer):
    """
    Deterministic simulated face recognizer for unit tests, offline development,
    and verifying scenario flows without live camera feeds.
    """
    def __init__(self):
        self.simulated_faces: List[FaceInfo] = []
        self._user_embeddings: Dict[str, np.ndarray] = {}

    def set_recognized_person(self, user_id: str, name: str, confidence: float = 0.94) -> None:
        emb = self.get_or_create_simulated_embedding(user_id)
        self.simulated_faces = [
            FaceInfo(
                bbox=(200, 100, 160, 180),
                confidence=0.98,
                embedding=emb,
                matched_user_id=user_id,
                matched_name=name,
                similarity_score=confidence,
                is_recognized=True
            )
        ]

    def set_unknown_person(self) -> None:
        emb = np.random.RandomState(42).randn(128).astype(np.float32)
        emb /= np.linalg.norm(emb)
        self.simulated_faces = [
            FaceInfo(
                bbox=(200, 100, 160, 180),
                confidence=0.95,
                embedding=emb,
                matched_user_id=None,
                matched_name=None,
                similarity_score=0.15,
                is_recognized=False
            )
        ]

    def clear_faces(self) -> None:
        self.simulated_faces = []

    def get_or_create_simulated_embedding(self, user_id: str) -> np.ndarray:
        if user_id not in self._user_embeddings:
            # Deterministic pseudo-embedding based on user_id string
            rng = np.random.RandomState(abs(hash(user_id)) % (2**31))
            vec = rng.randn(128).astype(np.float32)
            vec /= np.linalg.norm(vec)
            self._user_embeddings[user_id] = vec
        return self._user_embeddings[user_id]

    def detect_and_recognize(self, frame: np.ndarray) -> List[FaceInfo]:
        return self.simulated_faces

    def extract_embedding_from_crop(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        rng = np.random.RandomState(1234)
        vec = rng.randn(128).astype(np.float32)
        return vec / np.linalg.norm(vec)


def save_face_crop(face_crop: np.ndarray, user_id: str) -> Path:
    """Saves a face thumbnail crop to disk for user profile and dashboard."""
    settings.FACES_PATH.mkdir(parents=True, exist_ok=True)
    target_path = settings.FACES_PATH / f"{user_id}.jpg"
    cv2.imwrite(str(target_path), face_crop)
    return target_path
