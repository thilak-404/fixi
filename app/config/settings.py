"""
Configuration settings for FIXI - Local Person-Aware Robotic Companion.
All settings are local, private, and customizable via environment variables or .env file.
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Project Information
    APP_NAME: str = "FIXI"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "production"

    # Paths
    BASE_PATH: Path = BASE_DIR
    DATA_PATH: Path = DATA_DIR
    FACES_PATH: Path = DATA_DIR / "faces"
    CONVERSATIONS_PATH: Path = DATA_DIR / "conversations"
    DATABASE_PATH: Path = DATA_DIR / "fixi.db"
    MODELS_PATH: Path = MODELS_DIR
    PIPER_MODEL_PATH: Path = MODELS_DIR / "piper" / "en_US-lessac-medium.onnx"
    PIPER_CONFIG_PATH: Path = MODELS_DIR / "piper" / "en_US-lessac-medium.onnx.json"
    YUNET_MODEL_PATH: Path = MODELS_DIR / "vision" / "face_detection_yunet.onnx"
    SFACE_MODEL_PATH: Path = MODELS_DIR / "vision" / "face_recognition_sface.onnx"

    # Vision Settings
    CAMERA_INDEX: int = 0
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480
    CAMERA_FPS: int = 30
    PERSON_DETECTION_CONFIDENCE: float = 0.5
    YOLO_IMAGE_SIZE: int = 320
    FACE_MATCH_THRESHOLD: float = 0.45  # Cosine distance/similarity threshold for SFace (optimal 0.45)
    FRAME_PROCESS_INTERVAL: float = 0.1  # Seconds between vision inference

    # Speech Settings
    WHISPER_MODEL_SIZE: str = "tiny.en"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHANNELS: int = 1
    VAD_ENERGY_THRESHOLD: float = 0.02
    SILENCE_DURATION_THRESHOLD: float = 1.2  # Seconds of silence to trigger end of speech
    MAX_RECORDING_DURATION: float = 15.0  # Seconds

    # LLM Settings (Ollama)
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "qwen2.5:0.5b"
    OLLAMA_TEMPERATURE: float = 0.7
    OLLAMA_TIMEOUT: float = 30.0

    # Memory Settings
    MAX_SHORT_TERM_TURNS: int = 6
    TOP_K_EPISODIC_MEMORIES: int = 5
    MEMORY_SIMILARITY_THRESHOLD: float = 0.25

    # Web Dashboard Settings
    DASHBOARD_HOST: str = "0.0.0.0"
    DASHBOARD_PORT: int = 8000


settings = Settings()

# Ensure directories exist
settings.DATA_PATH.mkdir(parents=True, exist_ok=True)
settings.FACES_PATH.mkdir(parents=True, exist_ok=True)
settings.CONVERSATIONS_PATH.mkdir(parents=True, exist_ok=True)
settings.MODELS_PATH.mkdir(parents=True, exist_ok=True)
(settings.MODELS_PATH / "piper").mkdir(parents=True, exist_ok=True)
(settings.MODELS_PATH / "vision").mkdir(parents=True, exist_ok=True)
