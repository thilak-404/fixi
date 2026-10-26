"""
Local LLM Provider module for Fixi.
Abstracts the LLM interface so the local model can be replaced effortlessly.
Connects locally to Ollama (Qwen, Llama, etc.) with 0 external network requests.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import httpx
from app.config.settings import settings


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generates a text completion given a prompt."""
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generates a chat completion given a list of role/content messages."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Checks if the local LLM service is running and responsive."""
        pass


class OllamaProvider(BaseLLMProvider):
    """
    Local LLM provider interfacing directly with a locally running Ollama instance.
    """
    def __init__(
        self,
        base_url: str = settings.OLLAMA_BASE_URL,
        model_name: str = settings.OLLAMA_MODEL,
        timeout: float = settings.OLLAMA_TIMEOUT
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = settings.OLLAMA_TEMPERATURE,
        max_tokens: Optional[int] = None
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        if system_prompt:
            payload["system"] = system_prompt
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(f"{self.base_url}/api/generate", json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("response", "").strip()
                return f"[LLM Error: Status {resp.status_code}]"
        except Exception as e:
            return f"[Local LLM Connection Error: {e}]"

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = settings.OLLAMA_TEMPERATURE,
        max_tokens: Optional[int] = None
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(f"{self.base_url}/api/chat", json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    msg = data.get("message", {})
                    return msg.get("content", "").strip()
                return f"[LLM Error: Status {resp.status_code}]"
        except Exception as e:
            return f"[Local LLM Connection Error: {e}]"


class MockLLMProvider(BaseLLMProvider):
    """
    Deterministic mock provider for automated unit tests and fallback demonstrations.
    """
    def __init__(self):
        self.canned_responses: Dict[str, str] = {}
        self.default_response = "I am Fixi, your friendly companion."

    def set_response(self, trigger_phrase: str, response: str) -> None:
        self.canned_responses[trigger_phrase.lower()] = response

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        prompt_lower = prompt.lower()
        for k, v in self.canned_responses.items():
            if k in prompt_lower:
                return v
        return self.default_response

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        last_msg = messages[-1]["content"] if messages else ""
        return self.generate(last_msg)


llm_provider = OllamaProvider()
