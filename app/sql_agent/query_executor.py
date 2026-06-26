from app.sql_agent.query_generator import QueryFilter, QueryPlan

# This module never touches a real database — backend's storage model is EAV
# (DataTableRow.rowData JSON), and ExecuteSqlQueryUseCase is the only thing
# allowed to turn a QueryPlan into a parameterized lookup. This is purely an
# optional, in-memory self-check: if the caller supplies a small row sample,
# applying the plan to it lets the agent (and the caller) sanity-check the
# plan actually matches something before trusting it.


def apply_plan(plan: QueryPlan, rows: list[dict]) -> list[dict]:
    matched = [row for row in rows if _matches(row, plan.filters)]
    return matched[: plan.limit]


def _matches(row: dict, filters: list[QueryFilter]) -> bool:
    return all(_matches_filter(row.get(f.column), f) for f in filters)


def _matches_filter(actual: object, filter_: QueryFilter) -> bool:
    if actual is None:
        return False

    try:
        if filter_.operator == "eq":
            return actual == filter_.value
        if filter_.operator == "neq":
            return actual != filter_.value
        if filter_.operator == "gt":
            return actual > filter_.value
        if filter_.operator == "gte":
            return actual >= filter_.value
        if filter_.operator == "lt":
            return actual < filter_.value
        if filter_.operator == "lte":
            return actual <= filter_.value
        if filter_.operator == "contains":
            return str(filter_.value).lower() in str(actual).lower()
    except TypeError:
        return False

    return False
