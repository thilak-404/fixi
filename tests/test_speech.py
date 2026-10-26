"""
Tests for Fixi Local Speech Layer (VAD, STT, TTS).
"""

import numpy as np
import wave
import io
from app.speech.vad import EnergyVAD
from app.speech.tts import PiperTTS, MockTTS
from app.speech.stt import MockSTT


def test_vad_energy_calculation():
    vad = EnergyVAD(energy_threshold=0.05, silence_threshold_seconds=0.5)

    # Silent chunk
    silence = np.zeros(1600, dtype=np.int16)
    speaking, ended = vad.process_chunk(silence)
    assert speaking is False
    assert ended is False

    # Loud chunk (simulate speech)
    speech = np.random.randint(-15000, 15000, size=1600, dtype=np.int16)
    speaking, ended = vad.process_chunk(speech)
    assert speaking is True


def test_tts_synthesis_wav_format():
    tts = PiperTTS()
    wav_bytes = tts.synthesize_wav_bytes("Hello from Fixi local assistant.")
    assert len(wav_bytes) > 0

    # Verify standard WAV header
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnchannels() in [1, 2]
        assert wf.getframerate() in [16000, 22050, 24000]
        assert wf.getnframes() > 0


def test_mock_stt_transcription():
    stt = MockSTT("My name is Arun")
    dummy_audio = np.zeros(1600, dtype=np.int16)
    result = stt.transcribe(dummy_audio)
    assert result == "My name is Arun"
