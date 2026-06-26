from fastapi import APIRouter

from app.api.v1.schemas.route_schema import RouteQuestionRequest, RouteQuestionResponse
from app.router.question_router import decide_route

router = APIRouter(tags=["route"])


@router.post("/route", response_model=RouteQuestionResponse, response_model_by_alias=True)
def route_question(request: RouteQuestionRequest) -> RouteQuestionResponse:
    route = decide_route(request.question, request.model, request.has_documents, request.has_data_tables)
    return RouteQuestionResponse(route=route, model_used=request.model)
