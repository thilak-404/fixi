"""
Local Speech-to-Text (STT) module for Fixi.
Uses faster-whisper (CTranslate2) for offline, high-speed speech recognition.
Zero cloud dependencies.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union
import io
import wave
import numpy as np
from app.config.settings import settings


class BaseSTT(ABC):
    @abstractmethod
    def transcribe(self, audio: Union[str, Path, np.ndarray], sample_rate: int = settings.AUDIO_SAMPLE_RATE) -> str:
        """Transcribes audio file or numpy array into text."""
        pass


class FasterWhisperSTT(BaseSTT):
    """
    Local Whisper transcription engine using faster-whisper.
    """
    def __init__(
        self,
        model_size: str = settings.WHISPER_MODEL_SIZE,
        device: str = settings.WHISPER_DEVICE,
        compute_type: str = settings.WHISPER_COMPUTE_TYPE
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _ensure_model_loaded(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=str(settings.MODELS_PATH / "whisper")
                )
            except Exception as e:
                print(f"Warning: Could not load faster-whisper model: {e}")

    def transcribe(self, audio: Union[str, Path, np.ndarray], sample_rate: int = settings.AUDIO_SAMPLE_RATE) -> str:
        self._ensure_model_loaded()
        if self._model is None:
            return ""

        try:
            if isinstance(audio, (str, Path)):
                segments, info = self._model.transcribe(str(audio), beam_size=2)
            elif isinstance(audio, np.ndarray):
                # faster-whisper accepts 1D float32 numpy array normalized to [-1.0, 1.0]
                if np.issubdtype(audio.dtype, np.integer):
                    info_type = np.iinfo(audio.dtype)
                    audio_float = audio.astype(np.float32) / max(abs(info_type.min), info_type.max)
                else:
                    audio_float = audio.astype(np.float32)

                # Ensure 1D
                if audio_float.ndim > 1:
                    audio_float = audio_float.mean(axis=1)

                segments, info = self._model.transcribe(audio_float, beam_size=2)
            else:
                return ""

            text = " ".join([seg.text.strip() for seg in segments]).strip()
            return text
        except Exception as e:
            print(f"STT Transcription error: {e}")
            return ""


class MockSTT(BaseSTT):
    """Mock STT engine for tests and text simulation."""
    def __init__(self, response_text: str = ""):
        self.response_text = response_text

    def set_next_text(self, text: str):
        self.response_text = text

    def transcribe(self, audio: Union[str, Path, np.ndarray], sample_rate: int = settings.AUDIO_SAMPLE_RATE) -> str:
        return self.response_text


stt_engine = FasterWhisperSTT()
