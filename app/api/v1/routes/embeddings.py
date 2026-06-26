from fastapi import APIRouter

from app.api.v1.schemas.embeddings_schema import EmbeddingsRequest, EmbeddingsResponse
from app.rag.embeddings import embed_texts, embedding_model_name

router = APIRouter()


@router.post("/embeddings", response_model=EmbeddingsResponse)
def embeddings(request: EmbeddingsRequest) -> EmbeddingsResponse:
    vectors = embed_texts(request.texts)
    return EmbeddingsResponse(vectors=vectors, model=embedding_model_name())
