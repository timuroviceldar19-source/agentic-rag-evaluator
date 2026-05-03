from operator import add
from typing import Annotated, TypedDict

from app.models.schemas import AgentTraceEvent, EvaluationResult, SourceChunk, TokenUsage


class RAGState(TypedDict, total=False):
    question: str
    top_k: int
    started_at: float
    sources: list[SourceChunk]
    answer: str
    supported_claims: list[str]
    unsupported_claims: list[str]
    evaluation: EvaluationResult
    usage: TokenUsage
    trace: Annotated[list[AgentTraceEvent], add]
    latency_ms: int
