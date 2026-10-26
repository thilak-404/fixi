"""
Voice Activity Detection (VAD) module for Fixi.
Detects speech onset and offset using energy levels and silence duration tracking.
Runs 100% locally with zero external network access.
"""

import time
import numpy as np
from typing import Optional, Tuple
from app.config.settings import settings


class EnergyVAD:
    """
    Real-time energy-based voice activity detector with adaptive noise floor.
    """
    def __init__(
        self,
        energy_threshold: float = settings.VAD_ENERGY_THRESHOLD,
        silence_threshold_seconds: float = settings.SILENCE_DURATION_THRESHOLD,
        sample_rate: int = settings.AUDIO_SAMPLE_RATE
    ):
        self.energy_threshold = energy_threshold
        self.silence_threshold = silence_threshold_seconds
        self.sample_rate = sample_rate

        self.is_speech_active = False
        self._speech_start_time: Optional[float] = None
        self._last_speech_time: Optional[float] = None
        self._noise_floor: float = 0.005

    def reset(self) -> None:
        self.is_speech_active = False
        self._speech_start_time = None
        self._last_speech_time = None

    def calculate_rms(self, audio_chunk: np.ndarray) -> float:
        """Calculates Root-Mean-Square energy of an audio chunk."""
        if audio_chunk is None or len(audio_chunk) == 0:
            return 0.0
        # If integer PCM, convert to float32 range [-1.0, 1.0]
        if np.issubdtype(audio_chunk.dtype, np.integer):
            info = np.iinfo(audio_chunk.dtype)
            audio_float = audio_chunk.astype(np.float32) / max(abs(info.min), info.max)
        else:
            audio_float = audio_chunk.astype(np.float32)

        rms = float(np.sqrt(np.mean(np.square(audio_float))))
        return rms

    def process_chunk(self, audio_chunk: np.ndarray) -> Tuple[bool, bool]:
        """
        Processes an audio chunk.
        Returns (is_currently_speaking, speech_just_finished).
        """
        now = time.time()
        rms = self.calculate_rms(audio_chunk)

        effective_threshold = max(self.energy_threshold, self._noise_floor * 2.5)
        is_above_threshold = rms > effective_threshold

        if is_above_threshold:
            self._last_speech_time = now
            if not self.is_speech_active:
                self.is_speech_active = True
                self._speech_start_time = now
            return True, False
        else:
            # Update ambient noise floor gently when quiet
            self._noise_floor = (self._noise_floor * 0.95) + (rms * 0.05)

            if self.is_speech_active:
                silence_duration = now - (self._last_speech_time or now)
                if silence_duration >= self.silence_threshold:
                    # Speech ended
                    self.is_speech_active = False
                    self._speech_start_time = None
                    self._last_speech_time = None
                    return False, True
                return True, False
            return False, False
