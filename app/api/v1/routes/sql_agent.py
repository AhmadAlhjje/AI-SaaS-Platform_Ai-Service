from fastapi import APIRouter

from app.api.v1.schemas.sql_agent_schema import SqlAgentFilter, SqlAgentRequest, SqlAgentResponse
from app.core.config import settings
from app.sql_agent.query_executor import apply_plan
from app.sql_agent.query_generator import generate_query_plan
from app.sql_agent.schema_introspection import normalize_columns

router = APIRouter(prefix="/sql-agent", tags=["sql-agent"])


@router.post("/query", response_model=SqlAgentResponse, response_model_by_alias=True)
def query(request: SqlAgentRequest) -> SqlAgentResponse:
    columns = normalize_columns([c.model_dump() for c in request.columns])
    plan = generate_query_plan(request.question, columns, request.model, settings.sql_agent_max_limit)

    preview_row_count = None
    if request.sample_rows is not None:
        preview_row_count = len(apply_plan(plan, request.sample_rows))

    return SqlAgentResponse(
        filters=[SqlAgentFilter(column=f.column, operator=f.operator, value=f.value) for f in plan.filters],
        limit=plan.limit,
        model_used=request.model,
        preview_row_count=preview_row_count,
    )
