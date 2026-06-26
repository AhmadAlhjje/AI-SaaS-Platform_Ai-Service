import json
import re

from anthropic import Anthropic, APIError, APITimeoutError, AuthenticationError

from app.core.config import settings
from app.llm.base import ChatMessage, LLMProvider, LLMResult
from app.shared.exceptions import LLMProviderError

# Short ids accepted by the backend's AI configuration whitelist that need
# resolving to Anthropic's full dated model id.
MODEL_ALIASES = {
    "claude-haiku-4-5": "claude-haiku-4-5-20251001",
}

_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


class AnthropicProvider(LLMProvider):
    def __init__(self) -> None:
        if not settings.anthropic_api_key:
            raise LLMProviderError("ANTHROPIC_API_KEY is not configured")
        self._client = Anthropic(api_key=settings.anthropic_api_key, timeout=settings.llm_request_timeout_s)

    def complete(self, messages: list[ChatMessage], model: str, temperature: float, max_tokens: int) -> LLMResult:
        response = self._send(messages, model, temperature, max_tokens)
        content = "".join(block.text for block in response.content if block.type == "text")
        return self._to_result(content, model, response)

    def complete_json(
        self, messages: list[ChatMessage], model: str, temperature: float, max_tokens: int
    ) -> tuple[dict, LLMResult]:
        response = self._send(messages, model, temperature, max_tokens)
        raw = "".join(block.text for block in response.content if block.type == "text")
        parsed = self._parse_json(raw)
        return parsed, self._to_result(raw, model, response)

    def _send(self, messages: list[ChatMessage], model: str, temperature: float, max_tokens: int):
        system_text = "\n\n".join(m.content for m in messages if m.role == "system") or None
        conversation = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        resolved_model = MODEL_ALIASES.get(model, model)

        try:
            return self._client.messages.create(
                model=resolved_model,
                system=system_text,
                messages=conversation,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except AuthenticationError as error:
            raise LLMProviderError(f"Anthropic authentication failed: {error}") from error
        except APITimeoutError as error:
            raise LLMProviderError(f"Anthropic request timed out: {error}") from error
        except APIError as error:
            raise LLMProviderError(f"Anthropic request failed: {error}") from error

    def _parse_json(self, raw: str) -> dict:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        match = _JSON_OBJECT_PATTERN.search(raw)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as error:
                raise LLMProviderError(f"model did not return valid JSON: {error}") from error

        raise LLMProviderError("model did not return valid JSON")

    def _to_result(self, content: str, model: str, response) -> LLMResult:
        usage = response.usage
        prompt_tokens = usage.input_tokens if usage else 0
        completion_tokens = usage.output_tokens if usage else 0
        return LLMResult(
            content=content,
            model_used=response.model or model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
