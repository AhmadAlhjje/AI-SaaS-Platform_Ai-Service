import json

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from app.core.config import settings
from app.llm.base import ChatMessage, LLMProvider, LLMResult
from app.shared.exceptions import LLMProviderError

# Ollama needs no real key — it doesn't authenticate — but the OpenAI SDK
# requires a non-empty string for its client to construct.
_DUMMY_API_KEY = "ollama"


class OllamaProvider(LLMProvider):
    """Local inference via Ollama's OpenAI-compatible endpoint
    (http://localhost:11434/v1) — free, no API key, runs on this machine.
    No usage/token counts are reported by Ollama's chat endpoint, so those
    fields are estimated rather than real."""

    def __init__(self) -> None:
        self._client = OpenAI(
            api_key=_DUMMY_API_KEY,
            base_url=settings.ollama_base_url,
            timeout=settings.llm_request_timeout_s,
        )

    def complete(self, messages: list[ChatMessage], model: str, temperature: float, max_tokens: int) -> LLMResult:
        response = self._chat(messages, model, temperature, max_tokens, json_mode=False)
        choice = response.choices[0].message.content or ""
        return self._to_result(choice, model, response)

    def complete_json(
        self, messages: list[ChatMessage], model: str, temperature: float, max_tokens: int
    ) -> tuple[dict, LLMResult]:
        response = self._chat(messages, model, temperature, max_tokens, json_mode=True)
        raw = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as error:
            raise LLMProviderError(f"model did not return valid JSON: {error}") from error

        return parsed, self._to_result(raw, model, response)

    def _chat(self, messages: list[ChatMessage], model: str, temperature: float, max_tokens: int, json_mode: bool):
        try:
            return self._client.chat.completions.create(
                model=model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"} if json_mode else None,
            )
        except APIConnectionError as error:
            raise LLMProviderError(
                f'could not reach Ollama at "{settings.ollama_base_url}" — is "ollama serve" running '
                f"and is the model pulled (ollama pull {model})? {error}"
            ) from error
        except APITimeoutError as error:
            raise LLMProviderError(f"Ollama request timed out: {error}") from error
        except APIError as error:
            raise LLMProviderError(f"Ollama request failed: {error}") from error

    def _to_result(self, content: str, model: str, response) -> LLMResult:
        usage = response.usage
        return LLMResult(
            content=content,
            model_used=response.model or model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )
