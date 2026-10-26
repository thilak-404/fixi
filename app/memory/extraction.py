"""
Memory Extraction System for Fixi.
Extracts meaningful user facts (Name, Age, Games, Interests, Goals, Preferences)
and episodic summaries from dialogue without saving trivial chit-chat or questions.
Runs completely locally with deterministic heuristics and ground-truth verified LLM extraction.
"""

import json
import re
from typing import Dict, Any, List, Optional
from app.memory.memory_manager import memory_manager


NAME_STOPWORDS = {
    "not", "here", "fine", "good", "happy", "sad", "ready", "just", "doing",
    "working", "learning", "playing", "sorry", "asking", "telling", "talking",
    "eating", "going", "coming", "sure", "ok", "okay", "alright", "back", "tired",
    "busy", "feeling", "fixi", "bixi", "robot", "human", "friend", "user", "someone", "nobody"
}

GAME_SPELLINGS = {
    "valrount": "Valorant",
    "valorent": "Valorant",
    "vallorant": "Valorant",
    "pubg": "PUBG",
    "cod": "Call of Duty",
    "csgo": "CS:GO",
    "cs2": "Counter-Strike 2",
    "chess": "Chess",
    "football": "Football",
    "cricket": "Cricket",
    "minecraft": "Minecraft",
    "fortnite": "Fortnite",
    "gta": "GTA"
}

CASUAL_UTTERANCES = {
    "hi", "hello", "hey", "hey fixi", "hello fixi", "hi fixi", "good morning", "good evening",
    "good afternoon", "hallo", "yo", "sup", "how are you", "how are you doing", "what's up",
    "whats up", "nice to meet you", "thank you", "thanks", "ok", "okay", "bye", "goodbye",
    "see you", "cool", "great", "awesome", "yes", "no", "sure", "alright"
}


def is_casual_or_question(text: str) -> bool:
    """Detects if an utterance is a question, greeting, or trivial small talk that should NOT be saved."""
    t = text.strip().lower()
    if "?" in t:
        return True
    if re.match(r"^(?:do|does|did|what|whats|what's|how|why|when|where|who|whom|whose|can|could|would|should|is|are|am i|will|tell me)\b", t):
        return True
    cleaned_punc = re.sub(r"[^\w\s]", "", t).strip()
    return cleaned_punc in CASUAL_UTTERANCES


def normalize_game_name(raw: str) -> str:
    cleaned = raw.strip(".,! ").lower()
    return GAME_SPELLINGS.get(cleaned, raw.strip(".,! ").capitalize())


class MemoryExtractor:
    def __init__(self, llm_provider=None, memory_manager=None):
        self.llm = llm_provider
        self.memory_manager = memory_manager

    def extract_with_rules(self, text: str) -> Dict[str, Any]:
        """
        Deterministic, high-precision local rule extractor for reliable, instant extraction.
        Never extracts from questions or trivial greetings.
        """
        facts: Dict[str, str] = {}
        summary: Optional[str] = None
        lower = text.strip()

        if is_casual_or_question(lower):
            return {"has_important_info": False, "facts": {}, "episodic_summary": None}

        # Check explicit save / remember prefix
        explicit_remember = False
        inner_text = lower
        rem_match = re.search(r"\b(?:please\s+)?remember\s+(?:that\s+)?(?:i\s+)?([^.,!\n]+)", lower, re.IGNORECASE) or \
                    re.search(r"\bsave\s+(?:it as|that|this)?\s+([^.,!\n]+)", lower, re.IGNORECASE) or \
                    re.search(r"\bdon't forget\s+(?:that\s+)?([^.,!\n]+)", lower, re.IGNORECASE)
        if rem_match:
            explicit_remember = True
            inner_text = rem_match.group(1).strip()
            summary = f"User asked to remember: {inner_text}"

        # 1. Age extraction
        age_match = re.search(r"\b(?:i am|i'm|iam|my age is)\s+(\d{1,3})(?:\s+years\s+old|\s+yo|\s+year\s+old)?\b", inner_text, re.IGNORECASE) or \
                    re.search(r"\b(\d{1,3})\s+years\s+old\b", inner_text, re.IGNORECASE)
        if age_match:
            val = age_match.group(1)
            if 5 <= int(val) <= 120:
                facts["age"] = val
                if not summary:
                    summary = f"User is {val} years old."

        # 2. Games / Gaming extraction
        outdoor_match = re.search(r"\b(?:my outdoor game is|my outdoor sport is|outdoor game is|i play outdoor games like|i play outdoors)\s+([A-Za-z0-9\s_-]+)", inner_text, re.IGNORECASE)
        indoor_match = re.search(r"\b(?:my indoor game is|indoor game is|i play indoor games like|i play indoors)\s+([A-Za-z0-9\s_-]+)", inner_text, re.IGNORECASE)
        sport_match = re.search(r"\b(?:my favorite sport is|my sport is|i play the sport|my fav sport is)\s+([A-Za-z0-9\s_-]+)", inner_text, re.IGNORECASE)
        game_match = re.search(r"\b(?:i like to play|i love to play|i love playing|i play|i enjoy playing)\s+([A-Za-z0-9\s_-]+)", inner_text, re.IGNORECASE) or \
                     re.search(r"\b(?:favorite game is|favourite game is|fav game is)\s+([A-Za-z0-9\s_-]+)", inner_text, re.IGNORECASE)

        if outdoor_match:
            raw_out = outdoor_match.group(1).strip(".,! ")
            if " and " in raw_out:
                raw_out = raw_out.split(" and ")[0].strip()
            norm_out = normalize_game_name(raw_out)
            facts["outdoor_game"] = norm_out
            facts["sport"] = norm_out
            if not summary or explicit_remember:
                summary = f"User's outdoor game is {norm_out}."
        elif indoor_match:
            raw_in = indoor_match.group(1).strip(".,! ")
            if " and " in raw_in:
                raw_in = raw_in.split(" and ")[0].strip()
            norm_in = normalize_game_name(raw_in)
            facts["indoor_game"] = norm_in
            if not summary or explicit_remember:
                summary = f"User's indoor game is {norm_in}."
        elif sport_match:
            raw_sp = sport_match.group(1).strip(".,! ")
            if " and " in raw_sp:
                raw_sp = raw_sp.split(" and ")[0].strip()
            norm_sp = normalize_game_name(raw_sp)
            facts["sport"] = norm_sp
            facts["outdoor_game"] = norm_sp
            if not summary or explicit_remember:
                summary = f"User's favorite sport is {norm_sp}."
        elif game_match:
            raw_game = game_match.group(1).strip(".,! ")
            if " and " in raw_game:
                raw_game = raw_game.split(" and ")[0].strip()
            norm_game = normalize_game_name(raw_game)
            facts["game"] = norm_game
            facts["interest"] = f"Playing {norm_game}"
            if not summary or explicit_remember:
                summary = f"User likes to play {norm_game}."

        # 3. Name extraction
        name_match = re.search(r"\bmy name is ([A-Za-z]+)\b", inner_text, re.IGNORECASE) or \
                     re.search(r"\bcall me ([A-Za-z]+)\b", inner_text, re.IGNORECASE) or \
                     re.search(r"\b(?:i am|i'm|iam)\s+([A-Za-z]+)(?:[.!?,]|$)", inner_text, re.IGNORECASE)
        if name_match:
            cand = name_match.group(1).capitalize()
            if cand.lower() not in NAME_STOPWORDS and not re.search(r"\byears\s+old\b", inner_text, re.IGNORECASE):
                facts["name"] = cand

        # 4. Interests / Hobbies / Favorite things
        fav_match = re.search(r"\bmy favorite\s+([A-Za-z]+)\s+is\s+([^.,!\n]+)", inner_text, re.IGNORECASE)
        if fav_match:
            category = f"favorite_{fav_match.group(1).lower()}"
            facts[category] = fav_match.group(2).strip().capitalize()

        if "interest" not in facts:
            int_match = re.search(r"\bi (?:love|like|enjoy|am into) ([^.,!\n]+)", inner_text, re.IGNORECASE)
            if int_match:
                item = int_match.group(1).strip(".,! ")
                if " and " in item:
                    parts = item.split(" and ", 1)
                    facts["interest"] = parts[0].strip().capitalize()
                    learn_part = parts[1].strip()
                    learn_match = re.search(r"(?:i am|i'm|am)?\s*(?:learning|studying)\s+([^.,!\n]+)", learn_part, re.IGNORECASE)
                    if learn_match:
                        facts["learning"] = learn_match.group(1).strip(".,! ").capitalize()
                else:
                    facts["interest"] = item.capitalize()

        # 5. Learning / Studying / Projects
        learn_match = re.search(r"\bi(?:\'m| am|am)?\s+(?:learning|studying)\s+([^.,!\n]+)", inner_text, re.IGNORECASE) or \
                      re.search(r"\bi(?:\'m| am|am)?\s+working on\s+([^.,!\n]+)", inner_text, re.IGNORECASE)
        if learn_match and "learning" not in facts:
            facts["learning"] = learn_match.group(1).strip(".,! ").capitalize()

        # 6. Occupation / Studies
        occ_match = re.search(r"\bi work as (?:a |an )?([^.,!\n]+)", inner_text, re.IGNORECASE) or \
                    re.search(r"\bmy job is (?:a |an )?([^.,!\n]+)", inner_text, re.IGNORECASE) or \
                    re.search(r"\bi(?:\'m| am|am)?\s+(?:a|an)\s+([A-Za-z\s]+)\s+student\b", inner_text, re.IGNORECASE)
        if occ_match:
            facts["occupation"] = occ_match.group(1).strip(".,! ").capitalize()

        # 7. Location
        loc_match = re.search(r"\b(?:i live in|i'm from|i am from)\s+([A-Za-z\s]+)(?:[.,!]|$)", inner_text, re.IGNORECASE)
        if loc_match:
            facts["location"] = loc_match.group(1).strip(".,! ").capitalize()

        # 8. Preferences
        pref_match = re.search(r"\b(?:i prefer|my preference is|i like|i don't like)\s+([^.,!\n]+)", inner_text, re.IGNORECASE)
        if pref_match and "preference" not in facts and not facts.get("interest"):
            facts["preference"] = pref_match.group(1).strip().capitalize()

        if facts and not summary:
            parts = [f"{k}: {v}" for k, v in facts.items() if k != "name"]
            if parts:
                summary = f"User shared {', '.join(parts)}"

        has_info = bool(facts or summary)
        return {
            "has_important_info": has_info,
            "facts": facts,
            "episodic_summary": summary
        }

    def extract_with_llm(self, text: str, recent_turns: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Uses local LLM to extract facts, protected by ground-truth validation:
        Any extracted value MUST have its words actually present in user's text.
        """
        if self.llm is None or is_casual_or_question(text):
            return {"has_important_info": False, "facts": {}, "episodic_summary": None}

        prompt = f"""You are Fixi's Memory Extractor.
Extract only explicit personal facts stated by the user (name, age, game, interest, occupation).
If the user did not state personal facts, output {{"has_important_info": false}}.
User text: "{text}"
JSON:"""
        try:
            raw_response = self.llm.generate(prompt=prompt, temperature=0.1)
            json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                if isinstance(parsed, dict) and parsed.get("facts"):
                    verified_facts = {}
                    text_words = set(re.findall(r"\w+", text.lower()))
                    for k, v in parsed["facts"].items():
                        if not v or not isinstance(v, str):
                            continue
                        val_lower = v.strip().lower()
                        # Reject blacklisted filler words
                        if val_lower in {"not specified", "personal identity", "learning new things", "none", "unknown", "n/a", "friend", "fixi", "bixi"}:
                            continue
                        if k == "name" and val_lower in NAME_STOPWORDS:
                            continue
                        # Ground-truth verification: at least one word from the value MUST be in the user's text!
                        val_words = set(re.findall(r"\w+", val_lower))
                        if val_words and val_words.intersection(text_words):
                            verified_facts[k] = v.strip()

                    if verified_facts:
                        return {
                            "has_important_info": True,
                            "facts": verified_facts,
                            "episodic_summary": parsed.get("episodic_summary") or None
                        }
        except Exception:
            pass

        return {"has_important_info": False, "facts": {}, "episodic_summary": None}

    def process_and_store(
        self,
        user_id: str,
        user_input: str,
        recent_turns: Optional[List[Dict[str, str]]] = None,
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        Extracts information from user input and saves to user's isolated SQLite memory.
        Guarantees:
        - Never stores junk from greetings or questions.
        - Never overwrites user name with robot name.
        - Strictly isolated by user_id.
        """
        if not user_id or not user_input.strip():
            return {"stored": False, "reason": "empty_input"}

        if is_casual_or_question(user_input):
            return {"stored": False, "reason": "casual_or_question"}

        # 1. High-precision rule extraction
        rule_res = self.extract_with_rules(user_input)
        extracted = rule_res

        # 2. LLM extraction fallback only if rules found nothing
        if not extracted.get("has_important_info") and use_llm and self.llm is not None:
            llm_res = self.extract_with_llm(user_input, recent_turns)
            if llm_res.get("has_important_info"):
                extracted = llm_res

        if not extracted.get("has_important_info"):
            return {"stored": False, "reason": "no_important_info"}

        mgr = self.memory_manager or memory_manager

        # Store Level C stable facts
        facts = extracted.get("facts", {})
        for key, val in facts.items():
            if val and isinstance(val, str):
                mgr.set_user_fact(user_id=user_id, key=key, value=val)

        # Store Level B episodic memory if meaningful summary exists
        summary = extracted.get("episodic_summary")
        if summary:
            mgr.add_episodic_memory(
                user_id=user_id,
                summary=summary,
                details=f"Extracted from statement: '{user_input}'",
                category="conversation_learning",
                importance=1.5
            )

        return {
            "stored": True,
            "user_id": user_id,
            "facts": facts,
            "episodic_summary": summary
        }


memory_extractor = MemoryExtractor()
