from fastapi import APIRouter

from app.api.v1.schemas.ask_schema import AskRequest, AskResponse
from app.llm.base import ChatMessage
from app.llm.factory import get_llm_provider

router = APIRouter()


@router.post("/ask", response_model=AskResponse, response_model_by_alias=True)
def ask(request: AskRequest) -> AskResponse:
    messages = []
    if request.system_prompt:
        messages.append(ChatMessage(role="system", content=request.system_prompt))
    messages.append(ChatMessage(role="user", content=request.question))

    result = get_llm_provider(request.model).complete(
        messages=messages,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    return AskResponse(
        content=result.content,
        model_used=result.model_used,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
    )
