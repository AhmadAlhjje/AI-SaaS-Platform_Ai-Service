from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class LLMResult:
    content: str
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMProvider(ABC):
    """One implementation per upstream API (OpenAI, Anthropic, ...). Picked
    by app/llm/factory.py based on the model id's prefix — callers never
    instantiate a concrete provider directly."""

    @abstractmethod
    def complete(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResult:
        """Plain-text chat completion."""

    @abstractmethod
    def complete_json(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[dict, LLMResult]:
        """Chat completion constrained to return a single JSON object.
        Returns the parsed object alongside the same usage/result metadata
        as complete(). Raises LLMProviderError if the response isn't valid
        JSON — callers must not trust LLM output blindly regardless."""
