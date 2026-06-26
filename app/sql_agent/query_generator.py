from dataclasses import dataclass

from app.core.logging import get_logger
from app.llm.base import ChatMessage
from app.llm.factory import get_llm_provider
from app.sql_agent.schema_introspection import ColumnSchema

logger = get_logger(__name__)

ALLOWED_OPERATORS = {"eq", "neq", "gt", "gte", "lt", "lte", "contains"}
DEFAULT_LIMIT = 50

SYSTEM_PROMPT_TEMPLATE = """You translate a natural-language question into a structured query plan over a \
single table. You must respond with a single JSON object only, no prose, no markdown fences.

Table columns (only these may be referenced):
{columns}

Allowed operators: eq, neq, gt, gte, lt, lte, contains.
"contains" only applies to string columns; numeric/boolean/date comparisons use the others.

Respond with exactly this JSON shape:
{{"filters": [{{"column": "<name>", "operator": "<op>", "value": <value>}}], "limit": <integer>}}

Rules:
- Only use column names from the list above, exactly as spelled.
- If the question implies no filter, return an empty filters array.
- "limit" defaults to {default_limit} if the question doesn't ask for a specific count.
- Never invent columns or operators outside the lists given."""


@dataclass(frozen=True)
class QueryFilter:
    column: str
    operator: str
    value: object


@dataclass(frozen=True)
class QueryPlan:
    filters: list[QueryFilter]
    limit: int


def generate_query_plan(question: str, columns: list[ColumnSchema], model: str, max_limit: int) -> QueryPlan:
    columns_by_name = {c.name: c for c in columns}
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        columns="\n".join(f"- {c.name} ({c.type})" for c in columns),
        default_limit=DEFAULT_LIMIT,
    )

    parsed, _ = get_llm_provider(model).complete_json(
        messages=[ChatMessage(role="system", content=system_prompt), ChatMessage(role="user", content=question)],
        model=model,
        temperature=0.0,
        max_tokens=1000,
    )

    filters = _validate_filters(parsed.get("filters", []), columns_by_name)
    limit = _clamp_limit(parsed.get("limit"), max_limit)
    return QueryPlan(filters=filters, limit=limit)


def _validate_filters(raw_filters: object, columns_by_name: dict[str, ColumnSchema]) -> list[QueryFilter]:
    """Never trusts the LLM's JSON blindly: any filter referencing an unknown
    column or operator is dropped (logged, not raised) rather than letting it
    reach the caller — the caller re-validates columns again regardless
    (defense in depth, same pattern as backend's ExecuteSqlQueryUseCase)."""
    if not isinstance(raw_filters, list):
        return []

    filters: list[QueryFilter] = []
    for entry in raw_filters:
        if not isinstance(entry, dict):
            continue

        column = entry.get("column")
        operator = entry.get("operator")
        value = entry.get("value")

        if column not in columns_by_name:
            logger.warning('dropping filter referencing unknown column "%s"', column)
            continue
        if operator not in ALLOWED_OPERATORS:
            logger.warning('dropping filter with unsupported operator "%s"', operator)
            continue
        if value is None:
            continue

        filters.append(QueryFilter(column=column, operator=operator, value=value))

    return filters


def _clamp_limit(raw_limit: object, max_limit: int) -> int:
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        return min(DEFAULT_LIMIT, max_limit)

    return max(1, min(limit, max_limit))
