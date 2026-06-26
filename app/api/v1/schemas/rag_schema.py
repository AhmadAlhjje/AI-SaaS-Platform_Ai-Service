from typing import Optional

from pydantic import BaseModel, Field


class RagIndexRequest(BaseModel):
    company_id: str = Field(min_length=1, alias="companyId")
    document_id: str = Field(min_length=1, alias="documentId")
    text: str = Field(min_length=1)

    class Config:
        populate_by_name = True


class RagIndexResponse(BaseModel):
    chunk_count: int = Field(alias="chunkCount")
    model: str

    class Config:
        populate_by_name = True


class RagQueryRequest(BaseModel):
    company_id: str = Field(min_length=1, alias="companyId")
    question: str = Field(min_length=1)
    model: str
    system_prompt: Optional[str] = Field(default=None, alias="systemPrompt")
    temperature: float = 0.7
    max_tokens: int = Field(default=1000, alias="maxTokens")
    top_k: int = Field(default=5, ge=1, le=50, alias="topK")

    class Config:
        populate_by_name = True


class RagSource(BaseModel):
    document_id: str = Field(alias="documentId")
    chunk_index: int = Field(alias="chunkIndex")
    score: float
    content: str

    class Config:
        populate_by_name = True


class RagQueryResponse(BaseModel):
    content: str
    model_used: str = Field(alias="modelUsed")
    prompt_tokens: int = Field(alias="promptTokens")
    completion_tokens: int = Field(alias="completionTokens")
    total_tokens: int = Field(alias="totalTokens")
    sources: list[RagSource]

    class Config:
        populate_by_name = True
