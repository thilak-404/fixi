"""Deterministic point-and-command parsing and authorization gates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class CommandIntent(str, Enum):
    GO = "go"
    MOVE = "move"
    LOOK = "look"
    IDENTIFY_TARGET = "identify_target"
    STOP = "stop"
    CANCEL = "cancel"
    RETURN_HOME = "return_home"
    FOLLOW = "follow"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ParsedCommand:
    user_id: Optional[str]
    intent: CommandIntent
    target_reference: Optional[str]
    requires_confirmation: bool
    confidence: float
    raw_text: str
    timestamp: str
    status: str = "parsed"


class PointCommandParser:
    def parse(self, text: str, user_id: Optional[str] = None) -> ParsedCommand:
        raw_text = text.strip()
        lower = re.sub(r"^fixi[,:]?\s*", "", raw_text.lower()).strip()
        intent = CommandIntent.UNKNOWN
        target: Optional[str] = None
        confidence = 0.0
        confirmation = False

        if re.search(r"\b(?:stop|halt|freeze|emergency stop)\b", lower):
            intent, confidence = CommandIntent.STOP, 0.99
        elif re.search(r"\b(?:cancel|never mind|nevermind)\b", lower):
            intent, confidence = CommandIntent.CANCEL, 0.98
        elif re.search(r"\b(?:come back|return|go back)\b.*\b(?:start(?:ing)?|home|beginning)\b", lower):
            intent, confidence = CommandIntent.RETURN_HOME, 0.95
        elif re.search(r"\bwhat am i pointing at\b", lower):
            intent, confidence = CommandIntent.IDENTIFY_TARGET, 0.98
        elif re.search(r"\bfollow me\b", lower):
            intent, confidence = CommandIntent.FOLLOW, 0.98
        elif re.search(r"\b(?:look at|look toward|look towards)\b", lower):
            intent, confidence = CommandIntent.LOOK, 0.92
            target = self._extract_target(lower)
        elif re.search(r"\b(?:go|move)\b", lower):
            intent, confidence = CommandIntent.GO if lower.startswith("go") else CommandIntent.MOVE, 0.9
            target = self._extract_target(lower)
            confirmation = target is None
        elif re.search(r"\b(?:there|that object|that)\b", lower):
            intent, confidence, target, confirmation = CommandIntent.GO, 0.65, "pointed_object", True

        if target in {"that", "there", "that object"}:
            target = "pointed_object"
            confirmation = True

        return ParsedCommand(
            user_id=user_id,
            intent=intent,
            target_reference=target,
            requires_confirmation=confirmation,
            confidence=confidence,
            raw_text=raw_text,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _extract_target(text: str) -> Optional[str]:
        match = re.search(r"\b(?:to|at|toward|towards)\s+(?:that\s+)?(?:a\s+|an\s+|the\s+)?([a-z][a-z0-9_-]*)", text)
        if match:
            return match.group(1)
        if re.search(r"\b(?:there|that object|that)\b", text):
            return "pointed_object"
        return None


def authorize_point_command(
    command: ParsedCommand,
    active_user_id: Optional[str],
    recognition_confidence: float,
    people_count: int,
    face_is_recognized: bool,
) -> tuple[bool, str]:
    """Returns whether a command may reach the simulator, never the LLM."""
    if command.intent in {CommandIntent.STOP, CommandIntent.CANCEL}:
        return True, "Safety command is always accepted"
    if command.intent == CommandIntent.UNKNOWN:
        return False, "Command is not supported"
    if not active_user_id or not face_is_recognized:
        return False, "A recognized user is required before movement"
    if recognition_confidence < 0.60:
        return False, "Recognition confidence is too low"
    if people_count != 1:
        return False, "Movement requires one clearly identified person"
    if command.confidence < 0.75:
        return False, "Command confidence is too low"
    return True, "Command authorized for the active user"
