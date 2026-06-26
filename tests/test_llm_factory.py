import pytest

from app.llm import factory as factory_module
from app.llm.anthropic_provider import AnthropicProvider
from app.llm.deepseek_provider import DeepSeekProvider
from app.llm.factory import get_llm_provider
from app.llm.openai_provider import OpenAiProvider
from app.shared.exceptions import LLMProviderError


@pytest.fixture(autouse=True)
def _clear_provider_cache():
    # Providers are cached with lru_cache; clear between tests so each test's
    # monkeypatched settings actually take effect on construction.
    factory_module._openai.cache_clear()
    factory_module._anthropic.cache_clear()
    factory_module._deepseek.cache_clear()
    yield
    factory_module._openai.cache_clear()
    factory_module._anthropic.cache_clear()
    factory_module._deepseek.cache_clear()


def test_get_llm_provider_picks_openai_for_gpt_models(monkeypatch):
    monkeypatch.setattr("app.llm.openai_provider.settings.openai_api_key", "sk-test")
    assert isinstance(get_llm_provider("gpt-4o-mini"), OpenAiProvider)


def test_get_llm_provider_picks_anthropic_for_claude_models(monkeypatch):
    monkeypatch.setattr("app.llm.anthropic_provider.settings.anthropic_api_key", "sk-ant-test")
    assert isinstance(get_llm_provider("claude-sonnet-4-6"), AnthropicProvider)


def test_get_llm_provider_picks_deepseek_for_deepseek_models(monkeypatch):
    monkeypatch.setattr("app.llm.deepseek_provider.settings.deepseek_api_key", "sk-deepseek-test")
    assert isinstance(get_llm_provider("deepseek-chat"), DeepSeekProvider)


def test_get_llm_provider_raises_for_unknown_prefix():
    with pytest.raises(LLMProviderError):
        get_llm_provider("some-unknown-model")


def test_get_llm_provider_raises_when_deepseek_key_missing(monkeypatch):
    monkeypatch.setattr("app.llm.deepseek_provider.settings.deepseek_api_key", None)
    with pytest.raises(LLMProviderError):
        get_llm_provider("deepseek-chat")
