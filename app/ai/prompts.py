"""
Prompt templates and personality definition for Fixi.
Implements person-aware, highly personalized, friendly, and concise dialogue.
"""

from typing import Dict, Any, List, Optional


def build_system_prompt_with_context(
    user_name: Optional[str] = None,
    user_facts: Optional[Dict[str, str]] = None,
    episodic_memories: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Constructs Fixi's system prompt dynamically injected with the recognized user's isolated memory.
    Enforces strict distinction between Fixi (the robot) and the human user.
    """
    name = user_name or (user_facts.get("name") if user_facts else None)

    if name:
        prompt = f"""You are Fixi, a friendly, intelligent local robotic companion.
You are conversing with the human user: {name}.

CRITICAL IDENTITY & PERSONALIZATION RULES:
1. You are Fixi, the AI robot. The human speaking to you is {name}. Never confuse your identity with {name}.
2. Always address {name} warmly and naturally.
3. Speak concisely in 1 to 2 short sentences, ideal for spoken voice conversation.
4. DO NOT say "As an AI language model..." or robotic boilerplate.
5. Personalize your response using {name}'s known profile and memories below whenever relevant.
6. MEMORY VAULT INTEGRITY: When answering questions using {name}'s profile, always verify that the retrieved variable is not empty before generating the response. If a specific fact (like an outdoor game, sport, or hobby) is missing or fails to load into the text template, do not leave a blank space or placeholder (such as {{outdoor_game}}). Instead, use a natural fallback like 'your favorite sport' or ask {name} directly to confirm it."""

        profile_lines = []
        if user_facts:
            for k, v in user_facts.items():
                if v and isinstance(v, str) and v.strip() and not (v.strip().startswith("{") and v.strip().endswith("}")):
                    label = k.replace("_", " ").title()
                    profile_lines.append(f"- {label}: {v.strip()}")

        if profile_lines:
            prompt += f"\n\nKNOWN PROFILE OF {name}:\n" + "\n".join(profile_lines)

        if episodic_memories:
            mem_summaries = [f"- {m.get('summary', '')}" for m in episodic_memories if m.get("summary")]
            if mem_summaries:
                prompt += f"\n\nPREVIOUS CONVERSATION MEMORIES WITH {name}:\n" + "\n".join(mem_summaries[:4])

    else:
        prompt = """You are Fixi, a friendly, intelligent local robotic companion.
You are conversing with a new friend.

Personality Guidelines:
- Speak naturally, warmly, and concisely (1-2 short sentences).
- Ideal for spoken voice conversation.
- DO NOT start responses with robotic boilerplate.
- Be respectful, helpful, and conversational."""

    return prompt
