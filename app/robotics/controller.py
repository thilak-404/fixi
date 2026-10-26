"""
Robotics Integration abstraction for Fixi.
Separates the robotic hardware actuators from the AI brain.
Works seamlessly in simulation mode without physical hardware,
and provides serial/ROS2 extension points for physical robots.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple, Callable


@dataclass
class RobotState:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0  # degrees
    head_pan: float = 0.0  # degrees
    head_tilt: float = 0.0  # degrees
    is_moving: bool = False
    last_action: str = "idle"
    battery_level: float = 100.0


class BaseRobotController(ABC):
    @abstractmethod
    def move_forward(self, speed: float = 1.0, duration: float = 1.0) -> bool:
        pass

    @abstractmethod
    def move_backward(self, speed: float = 1.0, duration: float = 1.0) -> bool:
        pass

    @abstractmethod
    def turn_left(self, angle: float = 45.0) -> bool:
        pass

    @abstractmethod
    def turn_right(self, angle: float = 45.0) -> bool:
        pass

    @abstractmethod
    def stop(self) -> bool:
        pass

    @abstractmethod
    def look_at_person(self, bbox: Tuple[int, int, int, int], frame_size: Tuple[int, int] = (640, 480)) -> bool:
        pass

    @abstractmethod
    def wave(self) -> bool:
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        pass


class SimulatedRobotController(BaseRobotController):
    """
    Hardware-independent simulated controller.
    Records physical movements, virtual odometry, and action logs.
    """
    def __init__(self):
        self.state = RobotState()
        self.action_history: List[Dict[str, Any]] = []
        self._listeners: List[Callable[[Dict[str, Any]], None]] = []

    def add_listener(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self._listeners.append(callback)

    def _record_action(self, action: str, details: Dict[str, Any]) -> None:
        entry = {
            "action": action,
            "details": details,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state": {
                "x": round(self.state.x, 2),
                "y": round(self.state.y, 2),
                "heading": round(self.state.heading, 1),
                "head_pan": round(self.state.head_pan, 1),
                "head_tilt": round(self.state.head_tilt, 1)
            }
        }
        self.state.last_action = action
        self.action_history.append(entry)
        if len(self.action_history) > 100:
            self.action_history.pop(0)

        for cb in self._listeners:
            try:
                cb(entry)
            except Exception:
                pass

    def move_forward(self, speed: float = 1.0, duration: float = 1.0) -> bool:
        self.state.is_moving = True
        dist = speed * duration
        import math
        rad = math.radians(self.state.heading)
        self.state.x += dist * math.cos(rad)
        self.state.y += dist * math.sin(rad)
        self.state.is_moving = False
        self._record_action("move_forward", {"speed": speed, "duration": duration, "distance": dist})
        return True

    def move_backward(self, speed: float = 1.0, duration: float = 1.0) -> bool:
        self.state.is_moving = True
        dist = speed * duration
        import math
        rad = math.radians(self.state.heading)
        self.state.x -= dist * math.cos(rad)
        self.state.y -= dist * math.sin(rad)
        self.state.is_moving = False
        self._record_action("move_backward", {"speed": speed, "duration": duration, "distance": dist})
        return True

    def turn_left(self, angle: float = 45.0) -> bool:
        self.state.heading = (self.state.heading + angle) % 360
        self._record_action("turn_left", {"angle": angle})
        return True

    def turn_right(self, angle: float = 45.0) -> bool:
        self.state.heading = (self.state.heading - angle) % 360
        self._record_action("turn_right", {"angle": angle})
        return True

    def stop(self) -> bool:
        self.state.is_moving = False
        self._record_action("stop", {})
        return True

    def look_at_person(self, bbox: Tuple[int, int, int, int], frame_size: Tuple[int, int] = (640, 480)) -> bool:
        """
        Calculates pan/tilt angles to center the detected person in the camera frame.
        """
        fx, fy, fw, fh = bbox
        cx = fx + fw // 2
        cy = fy + fh // 2
        img_w, img_h = frame_size

        # Offset from center (-1.0 to 1.0)
        norm_dx = (cx - img_w / 2.0) / (img_w / 2.0)
        norm_dy = (cy - img_h / 2.0) / (img_h / 2.0)

        # Map to servo degrees (e.g. max 45 degrees pan, 30 degrees tilt)
        pan_deg = norm_dx * 45.0
        tilt_deg = -norm_dy * 30.0

        self.state.head_pan = pan_deg
        self.state.head_tilt = tilt_deg
        self._record_action("look_at_person", {
            "bbox": bbox,
            "pan": round(pan_deg, 1),
            "tilt": round(tilt_deg, 1)
        })
        return True

    def wave(self) -> bool:
        self._record_action("wave", {"gesture": "hand_wave"})
        return True

    def get_state(self) -> Dict[str, Any]:
        return {
            "x": round(self.state.x, 2),
            "y": round(self.state.y, 2),
            "heading": round(self.state.heading, 1),
            "head_pan": round(self.state.head_pan, 1),
            "head_tilt": round(self.state.head_tilt, 1),
            "is_moving": self.state.is_moving,
            "last_action": self.state.last_action,
            "battery_level": self.state.battery_level,
            "history_count": len(self.action_history)
        }


robot_controller = SimulatedRobotController()
