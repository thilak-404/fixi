"""
Consent management for Fixi.
Enforces privacy-first user onboarding: no persistent profile or memory is created
without explicit user consent.
"""

from enum import Enum
from typing import Tuple


class ConsentStatus(str, Enum):
    PENDING = "pending"
    GRANTED = "granted"
    DENIED = "denied"
    ANONYMOUS = "anonymous"


CONSENT_REQUEST_PHRASE = (
    "Hello! I don't think I've met you before. "
    "Would you like me to remember you for future conversations?"
)

NAME_REQUEST_PHRASE = (
    "Great! What is your name?"
)

DENIED_ACKNOWLEDGEMENT_PHRASE = (
    "No problem at all. We can still chat, but I won't save any personal memory."
)


def parse_consent_response(user_input: str) -> Tuple[bool, bool]:
    """
    Parses user utterance for consent.
    Returns (answered, is_granted).
    """
    text = user_input.strip().lower()

    # Affirmative cues
    affirmatives = [
        "yes", "yeah", "sure", "yep", "okay", "ok", "please", "certainly",
        "of course", "i would", "go ahead", "sounds good", "why not", "definitely"
    ]
    # Negative cues
    negatives = [
        "no", "nope", "never", "don't", "do not", "nah", "no thanks",
        "negative", "i'd rather not", "stop"
    ]

    for neg in negatives:
        if re_match_word(neg, text):
            return True, False

    for aff in affirmatives:
        if re_match_word(aff, text):
            return True, True

    return False, False


def re_match_word(word: str, text: str) -> bool:
    import re
    return bool(re.search(rf"\b{re.escape(word)}\b", text))
