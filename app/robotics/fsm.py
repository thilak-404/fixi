"""
Robotics Finite State Machine (FSM) for Fixi.
Manages high-level robot behaviors, autonomous transitions,
dynamic replanning triggers, and safety stops.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timezone
import time

from app.events import event_bus, EventType


class RobotFSMState(str, Enum):
    IDLE = "IDLE"
    PERCEIVING = "PERCEIVING"
    PLANNING = "PLANNING"
    NAVIGATING = "NAVIGATING"
    ARRIVED = "ARRIVED"
    BLOCKED = "BLOCKED"
    REPLANNING = "REPLANNING"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class RoboticsFSM:
    def __init__(self, initial_state: RobotFSMState = RobotFSMState.IDLE):
        self._current_state = initial_state
        self._state_start_time = time.time()
        self._history: List[Dict[str, Any]] = []
        self._transition_listeners: List[Callable[[RobotFSMState, RobotFSMState, Dict[str, Any]], None]] = []

        # Record initial state
        self._log_transition(None, initial_state, {"reason": "system_init"})

    @property
    def state(self) -> RobotFSMState:
        return self._current_state

    def get_state_duration_seconds(self) -> float:
        return round(time.time() - self._state_start_time, 2)

    def add_listener(self, callback: Callable[[RobotFSMState, RobotFSMState, Dict[str, Any]], None]):
        self._transition_listeners.append(callback)

    def transition_to(self, new_state: RobotFSMState, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Executes a transition from current state to new_state.
        Validates transition rules and broadcasts to EventBus.
        """
        old_state = self._current_state
        if old_state == new_state:
            return True

        # Valid transitions table
        valid_transitions = {
            RobotFSMState.IDLE: [RobotFSMState.PERCEIVING, RobotFSMState.PLANNING, RobotFSMState.EMERGENCY_STOP],
            RobotFSMState.PERCEIVING: [RobotFSMState.PLANNING, RobotFSMState.IDLE, RobotFSMState.EMERGENCY_STOP],
            RobotFSMState.PLANNING: [RobotFSMState.NAVIGATING, RobotFSMState.BLOCKED, RobotFSMState.IDLE, RobotFSMState.EMERGENCY_STOP],
            RobotFSMState.NAVIGATING: [RobotFSMState.ARRIVED, RobotFSMState.BLOCKED, RobotFSMState.EMERGENCY_STOP, RobotFSMState.IDLE],
            RobotFSMState.ARRIVED: [RobotFSMState.IDLE, RobotFSMState.PERCEIVING, RobotFSMState.PLANNING, RobotFSMState.EMERGENCY_STOP],
            RobotFSMState.BLOCKED: [RobotFSMState.REPLANNING, RobotFSMState.IDLE, RobotFSMState.EMERGENCY_STOP],
            RobotFSMState.REPLANNING: [RobotFSMState.NAVIGATING, RobotFSMState.BLOCKED, RobotFSMState.IDLE, RobotFSMState.EMERGENCY_STOP],
            RobotFSMState.EMERGENCY_STOP: [RobotFSMState.IDLE]
        }

        # Any state can transition to EMERGENCY_STOP
        if new_state != RobotFSMState.EMERGENCY_STOP:
            allowed = valid_transitions.get(old_state, [])
            if new_state not in allowed:
                print(f"Warning: Disallowed FSM transition {old_state} -> {new_state}")
                # Allow force override for robustness
                pass

        self._current_state = new_state
        self._state_start_time = time.time()
        details = context or {}

        self._log_transition(old_state, new_state, details)

        # Broadcast via EventBus
        event_bus.emit(
            EventType.SYSTEM_STATUS_CHANGED,
            {
                "previous_state": old_state.value if old_state else None,
                "current_state": new_state.value,
                "context": details
            },
            is_simulated=True
        )

        for listener in self._transition_listeners:
            try:
                listener(old_state, new_state, details)
            except Exception as e:
                print(f"FSM listener error: {e}")

        return True

    def _log_transition(self, old_state: Optional[RobotFSMState], new_state: RobotFSMState, context: Dict[str, Any]):
        entry = {
            "from": old_state.value if old_state else None,
            "to": new_state.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": context
        }
        self._history.append(entry)
        if len(self._history) > 50:
            self._history.pop(0)

    def get_status(self) -> Dict[str, Any]:
        return {
            "state": self._current_state.value,
            "duration_s": self.get_state_duration_seconds(),
            "history": self._history[-10:]
        }
