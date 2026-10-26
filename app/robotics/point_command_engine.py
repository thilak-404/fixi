"""Point-and-command decision engine built on the existing robotics simulator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Tuple

from app.robotics.commands import CommandIntent, ParsedCommand, authorize_point_command
from app.robotics.pointing import GestureObservation
from app.robotics.target_selection import ObjectDetection, PointingTargetSelector, TargetSelection


class PointCommandState(str, Enum):
    IDLE = "IDLE"
    OBSERVING = "OBSERVING"
    GESTURE_DETECTED = "GESTURE_DETECTED"
    TARGET_SELECTED = "TARGET_SELECTED"
    WAITING_FOR_CLARIFICATION = "WAITING_FOR_CLARIFICATION"
    COMMAND_VALIDATION = "COMMAND_VALIDATION"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    REPLANNING = "REPLANNING"
    COMPLETED = "COMPLETED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class PointCommandStatus:
    state: PointCommandState
    message: str
    target_id: Optional[str] = None
    target_label: Optional[str] = None
    target_grid: Optional[Tuple[int, int]] = None
    command: Optional[ParsedCommand] = None
    selection: Optional[TargetSelection] = None


class PointCommandEngine:
    """Coordinates perception and commands without granting the LLM movement authority."""

    def __init__(self, simulator, selector: Optional[PointingTargetSelector] = None):
        self.simulator = simulator
        self.selector = selector or PointingTargetSelector()
        self.state = PointCommandState.IDLE
        self.last_selection: Optional[TargetSelection] = None
        self.last_command: Optional[ParsedCommand] = None
        self.target_grid: Optional[Tuple[int, int]] = None
        self.status = PointCommandStatus(PointCommandState.IDLE, "Waiting for a stable pointing gesture")

    def observe(self, gesture: GestureObservation, detections: Iterable[ObjectDetection]) -> PointCommandStatus:
        self.state = PointCommandState.OBSERVING
        selection = self.selector.select(gesture, detections)
        self.last_selection = selection
        if selection.status == "ambiguous":
            self.state = PointCommandState.WAITING_FOR_CLARIFICATION
            return self._set_status("Two or more objects are plausible targets; ask the user to clarify", selection=selection)
        if selection.status != "selected":
            self.state = PointCommandState.GESTURE_DETECTED if gesture.confidence >= 0.55 else PointCommandState.OBSERVING
            return self._set_status(selection.reason, selection=selection)

        target = selection.selected.detection
        self.state = PointCommandState.TARGET_SELECTED
        self.target_grid = self.image_to_grid(target.center)
        return self._set_status("Target selected; say a command", target=target, selection=selection)

    def accept_command(
        self,
        command: ParsedCommand,
        active_user_id: Optional[str],
        recognition_confidence: float,
        people_count: int,
        face_is_recognized: bool,
    ) -> PointCommandStatus:
        self.last_command = command
        if command.intent in {CommandIntent.STOP, CommandIntent.CANCEL}:
            self.simulator.emergency_stop()
            self.state = PointCommandState.STOPPED
            return self._set_status("Navigation stopped", command=command)

        self.state = PointCommandState.COMMAND_VALIDATION
        allowed, reason = authorize_point_command(
            command, active_user_id, recognition_confidence, people_count, face_is_recognized
        )
        if not allowed:
            self.state = PointCommandState.ERROR
            return self._set_status(reason, command=command)
        if command.intent == CommandIntent.RETURN_HOME:
            self.target_grid = (10, 2)
        elif self.target_grid is None:
            self.state = PointCommandState.WAITING_FOR_CLARIFICATION
            return self._set_status("No confirmed target is available", command=command)

        self.state = PointCommandState.PLANNING
        if not self.simulator.set_goal(*self.target_grid, label=self._target_label()):
            self.state = PointCommandState.ERROR
            return self._set_status("No safe path is available to the target", command=command)
        self.state = PointCommandState.EXECUTING
        return self._set_status("Path planned; simulator execution started", command=command)

    def verify(self) -> PointCommandStatus:
        if self.state not in {PointCommandState.EXECUTING, PointCommandState.REPLANNING}:
            return self.status
        snapshot = self.simulator.get_full_state()
        if snapshot["fsm"]["state"] == "ARRIVED":
            self.state = PointCommandState.COMPLETED
            return self._set_status("Target reached and verified")
        if snapshot["fsm"]["state"] in {"BLOCKED", "REPLANNING"}:
            self.state = PointCommandState.REPLANNING
            return self._set_status("Path blocked; simulator is replanning")
        return self._set_status("Navigation progressing; awaiting feedback")

    @staticmethod
    def image_to_grid(center: Tuple[float, float], frame_size: Tuple[int, int] = (640, 480), grid_size: Tuple[int, int] = (20, 20)) -> Tuple[int, int]:
        """Maps image coordinates to simulation cells; it is not a real-world depth estimate."""
        x, y = center
        width, height = frame_size
        cols, rows = grid_size
        col = max(1, min(cols - 2, int((x / max(1, width)) * cols)))
        row = max(1, min(rows - 2, int((y / max(1, height)) * rows)))
        return col, row

    def _target_label(self) -> str:
        if self.last_selection and self.last_selection.selected:
            return self.last_selection.selected.detection.label
        return "target"

    def _set_status(self, message: str, target=None, selection=None, command=None) -> PointCommandStatus:
        if target is None and selection and selection.selected:
            target = selection.selected.detection
        self.status = PointCommandStatus(
            state=self.state,
            message=message,
            target_id=target.object_id if target else None,
            target_label=target.label if target else None,
            target_grid=self.target_grid,
            command=command or self.last_command,
            selection=selection or self.last_selection,
        )
        return self.status
