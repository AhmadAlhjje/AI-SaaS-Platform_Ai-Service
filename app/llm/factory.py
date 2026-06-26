from functools import lru_cache

from app.llm.anthropic_provider import AnthropicProvider
from app.llm.base import LLMProvider
from app.llm.deepseek_provider import DeepSeekProvider
from app.llm.ollama_provider import OllamaProvider
from app.llm.openai_provider import OpenAiProvider
from app.shared.exceptions import LLMProviderError

# Ordered so a longer, more specific prefix never gets shadowed by a shorter
# one — not an issue today, but cheap to keep correct as providers are added.
_DEEPSEEK_PREFIXES = ("deepseek-",)
_OLLAMA_PREFIXES = ("llama", "mistral", "qwen", "gemma", "phi")
_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")
_ANTHROPIC_PREFIXES = ("claude-",)


@lru_cache(maxsize=1)
def _openai() -> LLMProvider:
    return OpenAiProvider()


@lru_cache(maxsize=1)
def _anthropic() -> LLMProvider:
    return AnthropicProvider()


@lru_cache(maxsize=1)
def _deepseek() -> LLMProvider:
    return DeepSeekProvider()


@lru_cache(maxsize=1)
def _ollama() -> LLMProvider:
    return OllamaProvider()


def get_llm_provider(model: str) -> LLMProvider:
    """Resolves a provider from a model id (e.g. "gpt-4o-mini", "claude-sonnet-4-6",
    "deepseek-chat", "llama3.1:8b"). Raises LLMProviderError for an
    unrecognized prefix or a missing API key — both are deliberate 502s, not
    crashes, since the caller picked the model."""
    lowered = model.lower()

    if lowered.startswith(_DEEPSEEK_PREFIXES):
        return _deepseek()
    if lowered.startswith(_ANTHROPIC_PREFIXES):
        return _anthropic()
    if lowered.startswith(_OPENAI_PREFIXES):
        return _openai()
    if lowered.startswith(_OLLAMA_PREFIXES):
        return _ollama()

    raise LLMProviderError(f"no provider configured for model \"{model}\"")
