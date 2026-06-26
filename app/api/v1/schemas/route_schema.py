from pydantic import BaseModel, Field


class RouteQuestionRequest(BaseModel):
    question: str = Field(min_length=1)
    model: str
    has_documents: bool = Field(alias="hasDocuments")
    has_data_tables: bool = Field(alias="hasDataTables")

    class Config:
        populate_by_name = True


class RouteQuestionResponse(BaseModel):
    route: str
    model_used: str = Field(alias="modelUsed")

    class Config:
        populate_by_name = True
