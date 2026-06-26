from typing import Optional

from pydantic import BaseModel, Field


class SqlAgentColumn(BaseModel):
    name: str = Field(min_length=1)
    type: str = Field(min_length=1)


class SqlAgentRequest(BaseModel):
    question: str = Field(min_length=1)
    columns: list[SqlAgentColumn] = Field(min_length=1)
    model: str
    sample_rows: Optional[list[dict]] = Field(default=None, alias="sampleRows")

    class Config:
        populate_by_name = True


class SqlAgentFilter(BaseModel):
    column: str
    operator: str
    value: object


class SqlAgentResponse(BaseModel):
    filters: list[SqlAgentFilter]
    limit: int
    model_used: str = Field(alias="modelUsed")
    preview_row_count: Optional[int] = Field(default=None, alias="previewRowCount")

    class Config:
        populate_by_name = True
