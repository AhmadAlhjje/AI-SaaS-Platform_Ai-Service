from typing import List

from pydantic import BaseModel, Field


class EmbeddingsRequest(BaseModel):
    texts: List[str] = Field(min_length=1)


class EmbeddingsResponse(BaseModel):
    vectors: List[List[float]]
    model: str
