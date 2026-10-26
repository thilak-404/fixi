"""
Centralized Real-Time Event Bus for Fixi.
Streams typed, timestamped events across Computer Vision, Robotics, AI, and Memory modules
to WebSocket clients and internal listeners.
"""

import asyncio
import json
import threading
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Set, Optional
from collections import deque
from fastapi import WebSocket


class EventType(str, Enum):
    # Computer Vision & Identity Events
    PERSON_DETECTED = "person_detected"
    PERSON_LOST = "person_lost"
    TRACKING_UPDATED = "tracking_updated"
    RECOGNITION_STARTED = "recognition_started"
    USER_RECOGNIZED = "user_recognized"
    UNKNOWN_PERSON_DETECTED = "unknown_person_detected"
    
    # Memory Events
    MEMORY_LOADED = "memory_loaded"
    MEMORY_UPDATED = "memory_updated"
    
    # Voice & AI Events
    LISTENING_STARTED = "listening_started"
    TRANSCRIPTION_COMPLETED = "transcription_completed"
    INTENT_RESOLVED = "intent_resolved"
    LLM_STARTED = "llm_started"
    LLM_COMPLETED = "llm_completed"
    TTS_STARTED = "tts_started"
    TTS_COMPLETED = "tts_completed"
    
    # Robotics & Closed-Loop Planning Events
    PLANNING_STARTED = "planning_started"
    PATH_PLANNED = "path_planned"
    ACTION_STARTED = "action_started"
    ACTION_PROGRESS = "action_progress"
    ACTION_VERIFIED = "action_verified"
    ACTION_FAILED = "action_failed"
    REPLANNING_STARTED = "replanning_started"
    
    # System Status
    SYSTEM_STATUS_CHANGED = "system_status_changed"


class EventBus:
    """
    Thread-safe asynchronous Event Bus connecting backend modules
    with WebSocket clients and UI event timelines.
    """
    def __init__(self, history_size: int = 100):
        self._subscribers: Set[WebSocket] = set()
        self._history: deque = deque(maxlen=history_size)
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def get_history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._history)

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        with self._lock:
            self._subscribers.add(websocket)
            # Send historical events to newly connected client
            history_snapshot = list(self._history)
        
        for ev in history_snapshot:
            try:
                await websocket.send_text(json.dumps(ev))
            except Exception:
                break

    def disconnect(self, websocket: WebSocket):
        with self._lock:
            self._subscribers.discard(websocket)

    def emit(self, event_type: str, payload: Optional[Dict[str, Any]] = None, is_simulated: bool = False) -> Dict[str, Any]:
        """
        Thread-safe method to emit an event from any background thread or coroutine.
        """
        event = {
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload or {},
            "is_simulated": is_simulated
        }

        with self._lock:
            self._history.append(event)
            subs = list(self._subscribers)

        if subs:
            msg = json.dumps(event)
            # Schedule broadcast on asyncio event loop
            try:
                loop = self._loop
                if loop and loop.is_running():
                    asyncio.run_coroutine_threadsafe(self._broadcast(msg, subs), loop)
            except Exception:
                pass

        return event

    async def _broadcast(self, message: str, targets: List[WebSocket]):
        for ws in targets:
            try:
                await ws.send_text(message)
            except Exception:
                self.disconnect(ws)


event_bus = EventBus()
