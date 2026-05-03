from typing import Literal

from pydantic import BaseModel, Field

PipelineEngine = Literal["linear", "langgraph"]
GenerationMode = Literal["openai", "local_fallback"]


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


class QueryComparisonConfig(BaseModel):
    pipeline_engine: PipelineEngine
    label: str | None = Field(default=None, min_length=1, max_length=60)
    top_k: int | None = Field(default=None, ge=1, le=12)
    openai_model: str | None = Field(default=None, min_length=1, max_length=80)


def default_comparison_configs() -> list[QueryComparisonConfig]:
    return [
        QueryComparisonConfig(pipeline_engine="langgraph", label="LangGraph"),
        QueryComparisonConfig(pipeline_engine="linear", label="Linear baseline"),
    ]


class QueryComparisonRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=12)
    configs: list[QueryComparisonConfig] = Field(
        default_factory=default_comparison_configs,
        min_length=2,
        max_length=4,
    )


class QueryComparisonRun(BaseModel):
    label: str
    pipeline_engine: PipelineEngine
    model: str
    generation_mode: GenerationMode
    top_k: int
    response: QueryResponse


class QueryComparisonResponse(BaseModel):
    question: str
    runs: list[QueryComparisonRun]
