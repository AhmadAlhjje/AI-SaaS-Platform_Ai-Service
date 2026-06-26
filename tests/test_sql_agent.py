import pytest

from app.sql_agent import query_generator as query_generator_module
from app.sql_agent.query_executor import apply_plan
from app.sql_agent.query_generator import QueryFilter, QueryPlan, generate_query_plan
from app.sql_agent.schema_introspection import ColumnSchema, normalize_columns
from app.shared.exceptions import ValidationError


def test_normalize_columns_rejects_empty_list():
    with pytest.raises(ValidationError):
        normalize_columns([])


def test_normalize_columns_rejects_unsupported_type():
    with pytest.raises(ValidationError):
        normalize_columns([{"name": "age", "type": "int"}])


def test_normalize_columns_rejects_empty_name():
    with pytest.raises(ValidationError):
        normalize_columns([{"name": "", "type": "string"}])


def test_normalize_columns_dedupes_by_name():
    result = normalize_columns([
        {"name": "age", "type": "number"},
        {"name": "age", "type": "number"},
        {"name": "city", "type": "string"},
    ])

    assert result == [ColumnSchema(name="age", type="number"), ColumnSchema(name="city", type="string")]


class _FakeProvider:
    def __init__(self, payload):
        self._payload = payload

    def complete_json(self, messages, model, temperature, max_tokens):
        return self._payload, None


def test_generate_query_plan_drops_unknown_column(monkeypatch):
    payload = {"filters": [{"column": "ghost", "operator": "eq", "value": "x"}], "limit": 10}
    monkeypatch.setattr(query_generator_module, "get_llm_provider", lambda model: _FakeProvider(payload))

    plan = generate_query_plan("question", [ColumnSchema(name="city", type="string")], "gpt-4o-mini", max_limit=100)

    assert plan.filters == []
    assert plan.limit == 10


def test_generate_query_plan_drops_unsupported_operator(monkeypatch):
    payload = {"filters": [{"column": "city", "operator": "regex", "value": "x"}], "limit": 5}
    monkeypatch.setattr(query_generator_module, "get_llm_provider", lambda model: _FakeProvider(payload))

    plan = generate_query_plan("question", [ColumnSchema(name="city", type="string")], "gpt-4o-mini", max_limit=100)

    assert plan.filters == []


def test_generate_query_plan_keeps_valid_filter(monkeypatch):
    payload = {"filters": [{"column": "age", "operator": "gte", "value": 18}], "limit": 7}
    monkeypatch.setattr(query_generator_module, "get_llm_provider", lambda model: _FakeProvider(payload))

    plan = generate_query_plan("adults", [ColumnSchema(name="age", type="number")], "gpt-4o-mini", max_limit=100)

    assert plan.filters == [QueryFilter(column="age", operator="gte", value=18)]
    assert plan.limit == 7


def test_generate_query_plan_clamps_limit_to_max(monkeypatch):
    payload = {"filters": [], "limit": 9999}
    monkeypatch.setattr(query_generator_module, "get_llm_provider", lambda model: _FakeProvider(payload))

    plan = generate_query_plan("question", [ColumnSchema(name="age", type="number")], "gpt-4o-mini", max_limit=50)

    assert plan.limit == 50


def test_generate_query_plan_falls_back_to_default_limit_on_garbage(monkeypatch):
    payload = {"filters": [], "limit": "not-a-number"}
    monkeypatch.setattr(query_generator_module, "get_llm_provider", lambda model: _FakeProvider(payload))

    plan = generate_query_plan("question", [ColumnSchema(name="age", type="number")], "gpt-4o-mini", max_limit=100)

    assert plan.limit == query_generator_module.DEFAULT_LIMIT


ROWS = [
    {"name": "alice", "age": 30, "city": "Cairo"},
    {"name": "bob", "age": 17, "city": "Giza"},
    {"name": "carol", "age": 25, "city": "Cairo"},
]


@pytest.mark.parametrize(
    "filter_, expected_names",
    [
        (QueryFilter(column="city", operator="eq", value="Cairo"), {"alice", "carol"}),
        (QueryFilter(column="city", operator="neq", value="Cairo"), {"bob"}),
        (QueryFilter(column="age", operator="gt", value=18), {"alice", "carol"}),
        (QueryFilter(column="age", operator="gte", value=25), {"alice", "carol"}),
        (QueryFilter(column="age", operator="lt", value=25), {"bob"}),
        (QueryFilter(column="age", operator="lte", value=25), {"bob", "carol"}),
        (QueryFilter(column="name", operator="contains", value="ali"), {"alice"}),
    ],
)
def test_apply_plan_operators(filter_, expected_names):
    plan = QueryPlan(filters=[filter_], limit=50)
    matched = apply_plan(plan, ROWS)
    assert {row["name"] for row in matched} == expected_names


def test_apply_plan_respects_limit():
    plan = QueryPlan(filters=[], limit=1)
    matched = apply_plan(plan, ROWS)
    assert len(matched) == 1
