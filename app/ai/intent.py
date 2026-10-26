"""
Command and Intent Classifier for Fixi.
Dispatches user utterances into either:
1. Robotics Action Commands (Navigating toward person, stopping, waving, turning)
2. Personalized Conversational & Memory Queries (Level A/B/C memory, local LLM)
"""

import re
from enum import Enum
from typing import Dict, Any, Optional, Tuple

from app.events import event_bus, EventType
from app.robotics.simulation import robotics_sim
from app.robotics.controller import robot_controller
from app.robotics.commands import PointCommandParser, authorize_point_command
from app.ai.response import response_generator


class IntentCategory(str, Enum):
    ROBOTICS_COMMAND = "robotics_command"
    CONVERSATIONAL = "conversational"
    UNKNOWN = "unknown"


class RoboticsActionType(str, Enum):
    MOVE_TOWARD_PERSON = "move_toward_person"
    FOLLOW_PERSON = "follow_person"
    EMERGENCY_STOP = "emergency_stop"
    WAVE_GESTURE = "wave_gesture"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    RESET_POSITION = "reset_position"


class IntentClassifier:
    def __init__(self):
        self.command_parser = PointCommandParser()

    def classify(self, text: str) -> Tuple[IntentCategory, Optional[RoboticsActionType], Dict[str, Any]]:
        lower = text.strip().lower()

        # 1. Stop / Halt / Freeze commands
        if re.search(r"\b(?:stop|halt|freeze|emergency stop|cancel movement|don't move|dont move)\b", lower):
            return IntentCategory.ROBOTICS_COMMAND, RoboticsActionType.EMERGENCY_STOP, {}

        # 2. Follow or move toward the detected person
        if re.search(r"\b(?:follow me|come with me|keep following me)\b", lower):
            return IntentCategory.ROBOTICS_COMMAND, RoboticsActionType.FOLLOW_PERSON, {}

        # 3. Move toward person / Come closer
        if re.search(r"\b(?:move toward me|move towards me|come toward me|come towards me|come closer|come to me|come here|approach me|follow me|step closer)\b", lower):
            return IntentCategory.ROBOTICS_COMMAND, RoboticsActionType.MOVE_TOWARD_PERSON, {}

        # 3. Wave / Physical gesture
        if re.search(r"\b(?:wave|wave your hand|wave to me|wave at me|do a wave)\b", lower):
            return IntentCategory.ROBOTICS_COMMAND, RoboticsActionType.WAVE_GESTURE, {}

        # 4. Turn Left
        if re.search(r"\b(?:turn left|rotate left)\b", lower):
            return IntentCategory.ROBOTICS_COMMAND, RoboticsActionType.TURN_LEFT, {"angle": 45.0}

        # 5. Turn Right
        if re.search(r"\b(?:turn right|rotate right)\b", lower):
            return IntentCategory.ROBOTICS_COMMAND, RoboticsActionType.TURN_RIGHT, {"angle": 45.0}

        # 6. Reset simulation
        if re.search(r"\b(?:reset position|reset simulation|go to start|reset robot)\b", lower):
            return IntentCategory.ROBOTICS_COMMAND, RoboticsActionType.RESET_POSITION, {}

        # Default: Conversational / Memory
        return IntentCategory.CONVERSATIONAL, None, {}

    def process_utterance(
        self,
        text: str,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        person_spatial_coords: Optional[Tuple[int, int]] = None,
        responder: Optional[Any] = None,
        recognition_confidence: float = 0.0,
        people_count: int = 1,
        face_is_recognized: bool = False
    ) -> Dict[str, Any]:
        """
        Main entry point for processing user input from microphone or text chat.
        Resolves intent, triggers appropriate robotics or memory subroutines,
        and returns response payload for TTS and UI.
        """
        text_clean = text.strip()
        effective_name = user_name or "Friend"

        category, action_type, params = self.classify(text_clean)
        parsed_command = self.command_parser.parse(text_clean, user_id=user_id)

        event_bus.emit(
            EventType.INTENT_RESOLVED,
            {
                "input": text_clean,
                "category": category.value,
                "action": action_type.value if action_type else None,
                "user_id": user_id,
                "user_name": effective_name
            }
        )
        event_bus.emit(
            "command_parsed",
            {
                "intent": parsed_command.intent.value,
                "target_reference": parsed_command.target_reference,
                "confidence": parsed_command.confidence,
                "user_id": user_id,
            }
        )

        # Handle Robotics Commands
        if category == IntentCategory.ROBOTICS_COMMAND:
            authorized, authorization_reason = authorize_point_command(
                parsed_command,
                active_user_id=user_id,
                recognition_confidence=recognition_confidence,
                people_count=people_count,
                face_is_recognized=face_is_recognized,
            )
            if not authorized:
                event_bus.emit(
                    "command_rejected",
                    {"reason": authorization_reason, "user_id": user_id},
                )
                return {
                    "response": f"I can't execute that yet: {authorization_reason}.",
                    "action_executed": None,
                    "category": category.value,
                    "is_robotics": True,
                    "authorization_reason": authorization_reason,
                }

            if action_type == RoboticsActionType.EMERGENCY_STOP:
                robotics_sim.emergency_stop()
                resp = f"Safety stop activated, {effective_name}. All robot navigation is halted."
                return {
                    "response": resp,
                    "action_executed": "emergency_stop",
                    "category": category.value,
                    "is_robotics": True
                }

            elif action_type == RoboticsActionType.MOVE_TOWARD_PERSON:
                # Use provided spatial coords or check current person position in simulation
                target_col = None
                target_row = None
                if person_spatial_coords:
                    target_col, target_row = person_spatial_coords
                elif robotics_sim.person_col is not None and robotics_sim.person_row is not None:
                    target_col = robotics_sim.person_col
                    target_row = robotics_sim.person_row
                else:
                    # Default vision target cell in front of robot if camera sees someone at center
                    target_col = 10
                    target_row = 12

                success = robotics_sim.navigate_to_person(target_col, target_row)
                if success:
                    resp = f"I see you, {effective_name}! Planning safe A* path and moving toward your location now."
                else:
                    resp = f"I detected you, {effective_name}, but the path to your location is obstructed by obstacles."

                return {
                    "response": resp,
                    "action_executed": "navigate_to_person",
                    "target_coords": [target_col, target_row],
                    "success": success,
                    "category": category.value,
                    "is_robotics": True
                }

            elif action_type == RoboticsActionType.FOLLOW_PERSON:
                if person_spatial_coords:
                    target_col, target_row = person_spatial_coords
                elif robotics_sim.person_col is not None and robotics_sim.person_row is not None:
                    target_col, target_row = robotics_sim.person_col, robotics_sim.person_row
                else:
                    return {
                        "response": f"I can't see you clearly yet, {effective_name}. Please stay in view and try again.",
                        "action_executed": None,
                        "category": category.value,
                        "is_robotics": True,
                    }

                success = robotics_sim.start_following(target_col, target_row)
                resp = (
                    f"I can see you, {effective_name}. I will follow you in the simulation."
                    if success else f"I couldn't start following you, {effective_name}."
                )
                return {
                    "response": resp,
                    "action_executed": "follow_person" if success else None,
                    "target_coords": [target_col, target_row],
                    "success": success,
                    "category": category.value,
                    "is_robotics": True,
                }

            elif action_type == RoboticsActionType.WAVE_GESTURE:
                robot_controller.wave()
                resp = f"Hello {effective_name}! Waving to you."
                return {
                    "response": resp,
                    "action_executed": "wave",
                    "category": category.value,
                    "is_robotics": True
                }

            elif action_type == RoboticsActionType.TURN_LEFT:
                robot_controller.turn_left(params.get("angle", 45.0))
                resp = f"Turning left 45 degrees, {effective_name}."
                return {
                    "response": resp,
                    "action_executed": "turn_left",
                    "category": category.value,
                    "is_robotics": True
                }

            elif action_type == RoboticsActionType.TURN_RIGHT:
                robot_controller.turn_right(params.get("angle", 45.0))
                resp = f"Turning right 45 degrees, {effective_name}."
                return {
                    "response": resp,
                    "action_executed": "turn_right",
                    "category": category.value,
                    "is_robotics": True
                }

            elif action_type == RoboticsActionType.RESET_POSITION:
                robotics_sim.reset()
                resp = f"Simulation reset to initial coordinates, {effective_name}."
                return {
                    "response": resp,
                    "action_executed": "reset",
                    "category": category.value,
                    "is_robotics": True
                }

        # Otherwise: Conversational / Memory query
        active_responder = responder or response_generator
        chat_result = active_responder.generate_response(
            user_id=user_id,
            user_name=effective_name,
            user_input=text_clean
        )
        chat_result["is_robotics"] = False
        chat_result["category"] = category.value
        return chat_result


# Singleton intent classifier
intent_classifier = IntentClassifier()
