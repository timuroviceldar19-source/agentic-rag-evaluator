from typing import Literal

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    document_id: str
    name: str
    file_type: str
    chunk_count: int
    uploaded_at: str


class UploadResponse(BaseModel):
    documents: list[DocumentInfo]


class SourceChunk(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page: int | None = None
    chunk_index: int
    text: str
    score: float = Field(ge=0.0, le=1.0)


class AgentTraceEvent(BaseModel):
    agent: str
    action: str
    status: Literal["ok", "warning", "error"]
    detail: str
    duration_ms: int


class EvaluationResult(BaseModel):
    relevance_score: int = Field(ge=0, le=100)
    groundedness_score: int = Field(ge=0, le=100)
    completeness_score: int = Field(ge=0, le=100)
    hallucination_risk: Literal["low", "medium", "high"]
    supported_claims: list[str]
    unsupported_claims: list[str]
    missing_evidence: list[str]
    critic_notes: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=12)


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]
    evaluation: EvaluationResult
    agent_trace: list[AgentTraceEvent]
    latency_ms: int

