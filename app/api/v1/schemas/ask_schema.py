from typing import Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    system_prompt: Optional[str] = Field(default=None, alias="systemPrompt")
    model: str
    temperature: float = 0.7
    max_tokens: int = Field(default=1000, alias="maxTokens")

    class Config:
        populate_by_name = True


class AskResponse(BaseModel):
    content: str
    model_used: str = Field(alias="modelUsed")
    prompt_tokens: int = Field(alias="promptTokens")
    completion_tokens: int = Field(alias="completionTokens")
    total_tokens: int = Field(alias="totalTokens")

    class Config:
        populate_by_name = True
