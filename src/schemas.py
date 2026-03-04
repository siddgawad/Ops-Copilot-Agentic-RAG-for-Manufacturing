from pydantic import BaseModel, Field 


class QueryRequest(BaseModel):
    question: str = Field(..., description="The operations question from the user")
    machine_id: str | None = Field(default=None, description="Optional ID of machine involved")


class SourceCitation(BaseModel):
    text: str = Field(..., description="The retrieved chunk text")
    source: str = Field(..., description="Source document filename")
    score: float = Field(..., description="Relevance score from hybrid search")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer from the LLM")
    sources: list[SourceCitation] = Field(default=[], description="Source citations for the answer")