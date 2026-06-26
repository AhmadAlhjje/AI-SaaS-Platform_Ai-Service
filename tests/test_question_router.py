from app.router import question_router as router_module
from app.router.question_router import HUMAN, RAG, SQL_AGENT, decide_route


def test_decide_route_returns_human_when_nothing_available(monkeypatch):
    called = []
    monkeypatch.setattr(router_module, "get_llm_provider", lambda model: called.append(model) or None)

    result = decide_route("anything", "gpt-4o-mini", has_documents=False, has_data_tables=False)

    assert result == HUMAN
    assert called == []  # no LLM call needed when nothing is available


class _FakeProvider:
    def __init__(self, payload):
        self._payload = payload

    def complete_json(self, messages, model, temperature, max_tokens):
        return self._payload, None


def test_decide_route_accepts_valid_available_route(monkeypatch):
    monkeypatch.setattr(router_module, "get_llm_provider", lambda model: _FakeProvider({"route": "RAG"}))

    result = decide_route("what is your return policy?", "gpt-4o-mini", has_documents=True, has_data_tables=False)

    assert result == RAG


def test_decide_route_rejects_unavailable_route(monkeypatch):
    # Router picks SQL_AGENT but no data tables exist — must fall back to HUMAN.
    monkeypatch.setattr(router_module, "get_llm_provider", lambda model: _FakeProvider({"route": "SQL_AGENT"}))

    result = decide_route("do you have a black t-shirt?", "gpt-4o-mini", has_documents=True, has_data_tables=False)

    assert result == HUMAN


def test_decide_route_defaults_to_human_on_provider_error(monkeypatch):
    class _RaisingProvider:
        def complete_json(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(router_module, "get_llm_provider", lambda model: _RaisingProvider())

    result = decide_route("question", "gpt-4o-mini", has_documents=True, has_data_tables=True)

    assert result == HUMAN


def test_decide_route_defaults_to_human_on_garbage_response(monkeypatch):
    monkeypatch.setattr(router_module, "get_llm_provider", lambda model: _FakeProvider({"route": "NONSENSE"}))

    result = decide_route("question", "gpt-4o-mini", has_documents=True, has_data_tables=True)

    assert result == HUMAN


def test_decide_route_picks_sql_agent_when_available(monkeypatch):
    monkeypatch.setattr(router_module, "get_llm_provider", lambda model: _FakeProvider({"route": "sql_agent"}))

    result = decide_route("do you have a black t-shirt in XL?", "gpt-4o-mini", has_documents=False, has_data_tables=True)

    assert result == SQL_AGENT
