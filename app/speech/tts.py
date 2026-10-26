"""
Local Text-to-Speech (TTS) module for Fixi.
Uses Piper TTS for fast, neural, 100% offline speech synthesis.
Zero cloud API calls.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import io
import threading
import subprocess
import wave
import numpy as np
from app.config.settings import settings


class BaseTTS(ABC):
    @abstractmethod
    def synthesize_wav_bytes(self, text: str) -> bytes:
        """Synthesizes text into WAV audio bytes."""
        pass

    @abstractmethod
    def speak(self, text: str) -> bool:
        """Synthesizes and plays audio through the speaker."""
        pass


class PiperTTS(BaseTTS):
    """
    Local Piper neural text-to-speech engine.
    """
    def __init__(
        self,
        model_path: Optional[Path] = None,
        config_path: Optional[Path] = None
    ):
        self.model_path = Path(model_path or settings.PIPER_MODEL_PATH)
        self.config_path = Path(config_path or settings.PIPER_CONFIG_PATH)
        self._piper_voice = None
        self._lock = threading.Lock()
        self._init_piper()

    def _init_piper(self) -> None:
        if self.model_path.exists():
            try:
                from piper.voice import PiperVoice
                self._piper_voice = PiperVoice.load(
                    model_path=str(self.model_path),
                    config_path=str(self.config_path) if self.config_path.exists() else None
                )
            except Exception as e:
                # Piper can also run as CLI subprocess
                pass

    def is_available(self) -> bool:
        return self._piper_voice is not None or self.model_path.exists()

    def synthesize_wav_bytes(self, text: str) -> bytes:
        if not text.strip():
            return b""

        with self._lock:
            # Try Python library first
            if self._piper_voice is not None:
                try:
                    chunks = list(self._piper_voice.synthesize(text))
                    if chunks:
                        sr = chunks[0].sample_rate
                        channels = chunks[0].sample_channels
                        width = chunks[0].sample_width
                        raw_pcm = b"".join([c.audio_int16_bytes for c in chunks])

                        buf = io.BytesIO()
                        with wave.open(buf, "wb") as wav_file:
                            wav_file.setnchannels(channels)
                            wav_file.setsampwidth(width)
                            wav_file.setframerate(sr)
                            wav_file.writeframes(raw_pcm)
                        return buf.getvalue()
                except Exception as e:
                    pass

        # Fallback to piper CLI binary if available
        if self.model_path.exists():
            try:
                cmd = [
                    "piper",
                    "--model", str(self.model_path),
                    "--output_file", "-"
                ]
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout, _ = proc.communicate(input=text.encode("utf-8"), timeout=10.0)
                if proc.returncode == 0 and len(stdout) > 0:
                    return stdout
            except Exception:
                pass

        # If piper is not initialized, generate a silent WAV container for safety
        return self._generate_silent_wav(duration_sec=0.5)

    def speak(self, text: str) -> bool:
        """Synthesizes text and attempts local audio output via sounddevice."""
        if not text.strip():
            return False
        wav_bytes = self.synthesize_wav_bytes(text)
        if not wav_bytes:
            return False

        try:
            import sounddevice as sd
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                sr = wf.getframerate()
                channels = wf.getnchannels()
                frames = wf.readframes(wf.getnframes())
                dtype = np.int16 if wf.getsampwidth() == 2 else np.uint8
                audio_data = np.frombuffer(frames, dtype=dtype)
                if channels > 1:
                    audio_data = audio_data.reshape(-1, channels)
                sd.play(audio_data, samplerate=sr)
                sd.wait()
                return True
        except Exception as e:
            # Speakers might be absent or inaccessible in headless/container environments
            return False

    def _generate_silent_wav(self, duration_sec: float = 0.5, sample_rate: int = 16000) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            silence = np.zeros(int(sample_rate * duration_sec), dtype=np.int16)
            wf.writeframes(silence.tobytes())
        return buf.getvalue()


class MockTTS(BaseTTS):
    """Mock TTS for unit tests and headless automated runs."""
    def __init__(self):
        self.last_spoken_text = ""

    def synthesize_wav_bytes(self, text: str) -> bytes:
        self.last_spoken_text = text
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(np.zeros(1600, dtype=np.int16).tobytes())
        return buf.getvalue()

    def speak(self, text: str) -> bool:
        self.last_spoken_text = text
        return True


tts_engine = PiperTTS()
