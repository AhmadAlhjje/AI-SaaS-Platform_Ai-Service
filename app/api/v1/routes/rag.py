from fastapi import APIRouter

from app.api.v1.schemas.rag_schema import (
    RagIndexRequest,
    RagIndexResponse,
    RagQueryRequest,
    RagQueryResponse,
    RagSource,
)
from app.rag.pipeline import answer_question, index_document

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/index", response_model=RagIndexResponse, response_model_by_alias=True)
def index(request: RagIndexRequest) -> RagIndexResponse:
    result = index_document(request.company_id, request.document_id, request.text)
    return RagIndexResponse(chunk_count=result.chunk_count, model=result.model)


@router.post("/query", response_model=RagQueryResponse, response_model_by_alias=True)
def query(request: RagQueryRequest) -> RagQueryResponse:
    answer = answer_question(
        company_id=request.company_id,
        question=request.question,
        model=request.model,
        system_prompt=request.system_prompt,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_k=request.top_k,
    )
    result = answer.llm_result

    return RagQueryResponse(
        content=result.content,
        model_used=result.model_used,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
        sources=[
            RagSource(document_id=s.document_id, chunk_index=s.chunk_index, score=s.score, content=s.content)
            for s in answer.sources
        ],
    )
