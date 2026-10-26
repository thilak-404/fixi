"""
Response Orchestration for Fixi.
Combines personalized memory retrieval, direct self-attribute resolution,
prompt injection with strict identity separation, and high-precision memory extraction.
"""

import re
from typing import Dict, Any, Optional, List
from app.ai.llm import BaseLLMProvider, llm_provider
from app.ai.prompts import build_system_prompt_with_context
from app.memory.memory_manager import memory_manager
from app.memory.extraction import memory_extractor


def safe_get_fact(facts: Dict[str, str], *keys: str) -> Optional[str]:
    """
    Safely retrieves a fact from the Memory Vault.
    Guarantees that the retrieved variable is non-empty, non-whitespace,
    and not an unrendered template placeholder (such as '{outdoor_game}').
    """
    for k in keys:
        val = facts.get(k)
        if val is not None:
            val_str = str(val).strip()
            # Verify variable is not empty, not a literal None/null, and not an unrendered template placeholder
            if val_str and val_str.lower() not in {"none", "null", "undefined"} and not (val_str.startswith("{") and val_str.endswith("}")):
                return val_str
    return None


def safe_format_template(
    template: str,
    context: Dict[str, Any],
    fallbacks: Optional[Dict[str, str]] = None
) -> str:
    """
    Safely formats a string template with variables from the Memory Vault.
    Guarantees that if a specific fact (like {outdoor_game}) is missing or fails
    to load into the text template, it does NOT leave a blank space or unrendered
    placeholder. Instead, it substitutes a natural fallback (e.g. 'your favorite sport')
    or clean wording.
    """
    DEFAULT_FALLBACKS = {
        "outdoor_game": "your favorite sport",
        "sport": "your favorite sport",
        "sports": "your favorite sports",
        "game": "your favorite game",
        "games": "your favorite games",
        "indoor_game": "your favorite indoor game",
        "hobby": "your favorite hobby",
        "hobbies": "your favorite hobbies",
        "interest": "your interests",
        "learning": "what you are learning",
        "occupation": "your profession",
        "age": "your age",
        "name": "Friend",
        "user_name": "Friend"
    }
    effective_fallbacks = {**DEFAULT_FALLBACKS, **(fallbacks or {})}

    def replacer(match):
        var_name = match.group(1).strip()
        val = context.get(var_name)
        if val is not None and str(val).strip():
            val_str = str(val).strip()
            if not (val_str.startswith("{") and val_str.endswith("}")):
                return val_str
        # If missing, empty, or placeholder, use natural fallback
        return effective_fallbacks.get(var_name, f"your {var_name.replace('_', ' ')}")

    # Replace all {placeholder} patterns
    formatted = re.sub(r"\{([a-zA-Z0-9_]+)\}", replacer, template)
    # Ensure no empty spaces before punctuation like "playing , Arun" or double spaces
    formatted = re.sub(r"\s+([,;.!?])", r"\1", formatted)
    formatted = re.sub(r"\s+", " ", formatted).strip()
    return formatted


def resolve_personal_memory_query(
    query: str,
    user_name: str,
    facts: Dict[str, str],
    episodic: List[Dict[str, Any]]
) -> Optional[str]:
    """
    Directly answers personal questions about the user from their isolated SQLite memory
    with 100% precision and zero LLM hallucination.
    Always verifies that retrieved variables are non-empty before generating response.
    Never leaves blank spaces or empty placeholders.
    """
    lower = query.strip().lower()

    # 1. Outdoor Game questions (Explicit check for outdoor_game with natural fallback)
    if re.search(r"\b(?:what is my outdoor game|whats my outdoor game|what's my outdoor game|do you know my outdoor game|know my outdoor game|tell me my outdoor game|what outdoor game|which outdoor game|outdoor game do i play|my outdoor game)\b", lower):
        outdoor_game = safe_get_fact(facts, "outdoor_game", "outdoor_sport")
        if outdoor_game:
            return f"You told me that your outdoor game is {outdoor_game}, {user_name}!"
        # If outdoor_game is missing or empty, do not leave a blank space: use natural fallback or ask directly
        sport = safe_get_fact(facts, "sport", "favorite_sport", "game")
        if sport:
            return f"I remember you enjoy {sport}, but I don't have your specific outdoor game saved yet, {user_name}. What is your favorite outdoor sport?"
        return f"I don't have your outdoor game saved yet, {user_name}. What is your favorite sport?"

    # 2. Indoor Game questions
    if re.search(r"\b(?:what is my indoor game|whats my indoor game|what's my indoor game|what indoor game|which indoor game|indoor game do i play|my indoor game)\b", lower):
        indoor_game = safe_get_fact(facts, "indoor_game")
        if indoor_game:
            return f"You told me that your indoor game is {indoor_game}, {user_name}!"
        return f"I don't have your indoor game saved yet, {user_name}. What indoor game do you enjoy playing?"

    # 3. Sport / Favorite Sport questions
    if re.search(r"\b(?:what is my sport|whats my sport|what's my sport|what sport|which sport|sport do i play|what is my favorite sport|whats my favorite sport|my favorite sport|my sport)\b", lower):
        sport = safe_get_fact(facts, "sport", "favorite_sport", "outdoor_game")
        if sport:
            return f"You told me that your favorite sport is {sport}, {user_name}!"
        return f"I don't have your favorite sport saved yet, {user_name}. What sport do you like to play?"

    # 4. Games / What do I play
    if re.search(r"\b(?:what game|what games|what do i play|what do i like to play|which game|game do i like|games do i like)\b", lower):
        game = safe_get_fact(facts, "game", "games", "favorite_game", "outdoor_game", "indoor_game", "sport")
        if game:
            return f"You told me that you love playing {game}, {user_name}!"
        interest = safe_get_fact(facts, "interest")
        if interest and "play" in interest.lower():
            return f"You told me that you enjoy {interest}, {user_name}!"
        return f"I don't have your favorite sport or game saved yet, {user_name}. What games do you enjoy?"

    # 5. Age questions
    if re.search(r"\b(?:what is my age|whats my age|what's my age|how old am i|do you know my age|know my age|tell me my age|my age\b)", lower):
        age = safe_get_fact(facts, "age")
        if age:
            return f"Yes, {user_name}! You are {age} years old."
        return f"I don't know your age yet, {user_name}. How old are you?"

    # 6. Name / Identity questions
    if re.search(r"\b(?:what is my name|whats my name|what's my name|who am i|do you know who i am|do you know my name|do you remember my name|who i am)\b", lower):
        name = safe_get_fact(facts, "name") or (user_name if user_name and user_name.lower() != "friend" else None)
        if name:
            return f"You are {name}!"
        return "I don't have your name saved in my memory yet. What is your name?"

    # 7. Hobbies / Interests
    if re.search(r"\b(?:what are my hobbies|what is my hobby|what are my interests|what is my interest|what do i like to do|what do i like)\b", lower):
        items = []
        g = safe_get_fact(facts, "game", "outdoor_game", "sport")
        if g:
            items.append(f"playing {g}")
        it = safe_get_fact(facts, "interest")
        if it and "play" not in it.lower():
            items.append(it.lower())
        lr = safe_get_fact(facts, "learning")
        if lr:
            items.append(f"learning {lr}")
        if items:
            return f"You like {', and '.join(items)}, {user_name}!"
        return f"I haven't learned your favorite sport or hobbies yet, {user_name}. Tell me what you're interested in!"

    # 8. Full memory summary ("What do you remember about me?")
    if re.search(r"\b(?:what do you remember|what do you know about me|remember about me|what have i told you|tell me what you remember|my memories)\b", lower):
        items = []
        age = safe_get_fact(facts, "age")
        if age:
            items.append(f"you are {age} years old")
        outdoor_game = safe_get_fact(facts, "outdoor_game")
        if outdoor_game:
            items.append(f"your outdoor game is {outdoor_game}")
        game = safe_get_fact(facts, "game")
        if game and game != outdoor_game:
            items.append(f"you love playing {game}")
        sport = safe_get_fact(facts, "sport")
        if sport and sport != outdoor_game and sport != game:
            items.append(f"your favorite sport is {sport}")
        interest = safe_get_fact(facts, "interest")
        if interest and "play" not in interest.lower():
            items.append(f"you like {interest}")
        learning = safe_get_fact(facts, "learning")
        if learning:
            items.append(f"you're learning {learning}")
        occupation = safe_get_fact(facts, "occupation")
        if occupation:
            items.append(f"you work as a {occupation}")
        for k, v in facts.items():
            if k not in ["name", "age", "game", "games", "interest", "learning", "occupation", "outdoor_game", "sport"]:
                val = safe_get_fact(facts, k)
                if val:
                    items.append(f"{k.replace('_', ' ')} is {val}")

        ep_part = ""
        if episodic:
            ep_summary = episodic[0].get("summary", "")
            if ep_summary and ep_summary.strip():
                ep_part = f" Also, I remember: {ep_summary.strip()}"

        if items:
            return f"I remember that you are {user_name}, and {' and '.join(items)}.{ep_part}"
        if episodic and ep_part:
            return f"I remember that you are {user_name}.{ep_part}"
        return f"I remember your name is {user_name}, but we haven't discussed your specific interests yet!"

    # 9. Generic attribute questions ("What is my {attribute}?", "Do you know my {attribute}?")
    attr_match = re.search(r"\b(?:what is my|whats my|what's my|do you know my|tell me my)\s+([a-z\s_-]+?)(?:\?|$)", lower)
    if attr_match:
        raw_attr = attr_match.group(1).strip()
        if raw_attr not in ["age", "name", "game", "games", "sport", "sports", "hobby", "hobbies", "interest", "interests"]:
            normalized_key = raw_attr.replace(" ", "_")
            val = safe_get_fact(facts, normalized_key, raw_attr)
            if val:
                return f"Your {raw_attr} is {val}, {user_name}!"
            # If missing or failed to load into template, never leave a blank space. Use natural fallback or ask user directly:
            if "outdoor" in raw_attr or "sport" in raw_attr:
                return f"I don't have your {raw_attr} saved yet, {user_name}. What is your favorite sport?"
            elif "game" in raw_attr:
                return f"I don't have your {raw_attr} saved yet, {user_name}. What is your favorite game?"
            else:
                return f"I don't have your {raw_attr} saved in my memory yet, {user_name}. Could you tell me what it is?"

    return None


def generate_teaching_acknowledgement(
    extracted_facts: Dict[str, str],
    user_name: str,
    original_text: str
) -> Optional[str]:
    """
    Generates a natural confirmation response when the user explicitly teaches Fixi a new fact.
    Verifies that extracted facts contain valid, non-empty values before generating response.
    """
    lower = original_text.strip().lower()
    if "?" in lower or re.match(r"^(?:do|does|did|what|whats|how|is|are|am i|who)\b", lower):
        return None

    if "outdoor_game" in extracted_facts and safe_get_fact(extracted_facts, "outdoor_game"):
        return f"Awesome, {user_name}! I've saved that your outdoor game is {extracted_facts['outdoor_game']}."
    if "sport" in extracted_facts and safe_get_fact(extracted_facts, "sport"):
        return f"Great, {user_name}! I have remembered that your favorite sport is {extracted_facts['sport']}."
    if "age" in extracted_facts and safe_get_fact(extracted_facts, "age"):
        return f"Got it, {user_name}! I have remembered that you are {extracted_facts['age']} years old."
    if "game" in extracted_facts and safe_get_fact(extracted_facts, "game"):
        return f"Awesome, {user_name}! I've saved that you like to play {extracted_facts['game']}."
    if "interest" in extracted_facts and "learning" in extracted_facts:
        it = safe_get_fact(extracted_facts, "interest")
        lr = safe_get_fact(extracted_facts, "learning")
        if it and lr:
            return f"That's great, {user_name}! I'll remember that you like {it} and you're learning {lr}."
    if "interest" in extracted_facts and safe_get_fact(extracted_facts, "interest"):
        return f"Nice! I'll remember that you like {extracted_facts['interest']}, {user_name}."
    if "learning" in extracted_facts and safe_get_fact(extracted_facts, "learning"):
        return f"Sounds exciting, {user_name}! I'll remember that you are learning {extracted_facts['learning']}."
    if "occupation" in extracted_facts and safe_get_fact(extracted_facts, "occupation"):
        return f"Understood, {user_name}! I've noted that you work as a {extracted_facts['occupation']}."
    if re.search(r"\b(?:remember|save)\b", lower):
        return f"Got it, {user_name}! I have saved that to your memory."

    return None


class ResponseGenerator:
    def __init__(self, llm: Optional[BaseLLMProvider] = None):
        self.llm = llm or llm_provider
        self.memory_manager = memory_manager
        self.extractor = memory_extractor
        self.extractor.llm = self.llm

    def clean_text_for_speech(self, text: str, user_name: str = "Friend", facts: Optional[Dict[str, str]] = None) -> str:
        """
        Cleans LLM response for clear, natural speech synthesis and resolves any template remnants.
        Verifies that no blank spaces or unrendered placeholders ({variable}) remain.
        """
        context = {"user_name": user_name, "name": user_name, **(facts or {})}
        resolved = safe_format_template(text, context)
        cleaned = re.sub(r"\*[^*]+\*", "", resolved)
        cleaned = re.sub(r"[*_~`#]", "", cleaned)
        cleaned = re.sub(r"\s+([,;.!?])", r"\1", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def generate_response(
        self,
        user_id: Optional[str],
        user_name: Optional[str],
        user_input: str
    ) -> Dict[str, Any]:
        """
        Generates a personalized response for the current person.
        Enforces strict memory isolation: retrieves memory ONLY for user_id.
        """
        user_input_clean = user_input.strip()
        if not user_input_clean:
            return {"response": "I didn't quite hear that.", "extracted": {}}

        effective_name = user_name or "Friend"

        # 1. Retrieve Current User Memory (Strictly Isolated by user_id)
        facts = {}
        episodic = []
        recent_turns = []

        if user_id:
            facts = self.memory_manager.get_user_facts(user_id)
            episodic = self.memory_manager.search_episodic_memories(user_id, user_input_clean)
            recent_turns = self.memory_manager.get_recent_conversation(user_id)
            if not effective_name or effective_name == "Friend":
                effective_name = facts.get("name", "Friend")

        # 2. Check for Direct Personal Memory Queries (100% precision, zero hallucination)
        if user_id:
            direct_answer = resolve_personal_memory_query(
                query=user_input_clean,
                user_name=effective_name,
                facts=facts,
                episodic=episodic
            )
            if direct_answer:
                self.memory_manager.add_short_term_turn(user_id, "user", user_input_clean)
                self.memory_manager.add_short_term_turn(user_id, "fixi", direct_answer)
                return {
                    "response": direct_answer,
                    "user_id": user_id,
                    "facts_used": facts,
                    "extracted": {}
                }

        # 3. Extract and store any new facts or memories from user's statement
        extracted = {}
        if user_id:
            extracted = self.extractor.process_and_store(
                user_id=user_id,
                user_input=user_input_clean,
                recent_turns=recent_turns,
                use_llm=True
            )
            if extracted.get("stored"):
                # Refresh facts dictionary with newly saved data
                facts = self.memory_manager.get_user_facts(user_id)
                # Check for direct teaching confirmation
                ack = generate_teaching_acknowledgement(
                    extracted_facts=extracted.get("facts", {}),
                    user_name=effective_name,
                    original_text=user_input_clean
                )
                if ack:
                    self.memory_manager.add_short_term_turn(user_id, "user", user_input_clean)
                    self.memory_manager.add_short_term_turn(user_id, "fixi", ack)
                    return {
                        "response": ack,
                        "user_id": user_id,
                        "facts_used": facts,
                        "extracted": extracted
                    }

        # 4. Construct System Prompt with isolated user memory
        sys_prompt = build_system_prompt_with_context(
            user_name=effective_name,
            user_facts=facts,
            episodic_memories=episodic
        )

        # 5. Build dialogue message sequence
        messages = [{"role": "system", "content": sys_prompt}]
        for turn in recent_turns[-4:]:
            role = "assistant" if turn["role"] == "fixi" else "user"
            messages.append({"role": role, "content": turn["content"]})
        messages.append({"role": "user", "content": user_input_clean})

        # 6. Call Local LLM for personalized natural conversation
        raw_response = self.llm.chat(messages=messages, temperature=0.7, max_tokens=150)
        clean_response = self.clean_text_for_speech(raw_response, user_name=effective_name, facts=facts)

        # 7. Record turns in isolated short-term memory
        if user_id:
            self.memory_manager.add_short_term_turn(user_id, "user", user_input_clean)
            self.memory_manager.add_short_term_turn(user_id, "fixi", clean_response)

        return {
            "response": clean_response,
            "user_id": user_id,
            "facts_used": facts,
            "extracted": extracted
        }


response_generator = ResponseGenerator()
