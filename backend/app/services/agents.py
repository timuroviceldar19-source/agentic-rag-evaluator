import re
import time

from app.core.config import Settings
from app.models.schemas import (
    AgentTraceEvent,
    EvaluationResult,
    QueryResponse,
    SourceChunk,
)
from app.services.llm import LLMClient
from app.services.vector_store import VectorStore, tokenize


class AgenticRAGPipeline:
    def __init__(self, vector_store: VectorStore, settings: Settings) -> None:
        self.vector_store = vector_store
        self.llm = LLMClient(settings)

    def run(self, question: str, top_k: int) -> QueryResponse:
        started_at = time.perf_counter()
        trace: list[AgentTraceEvent] = []

        retrieval_started = time.perf_counter()
        sources = self.vector_store.search(question, top_k=top_k)
        trace.append(
            _event(
                "Retrieval Agent",
                "Search vector store",
                "ok" if sources else "warning",
                f"Found {len(sources)} source chunks.",
                retrieval_started,
            )
        )

        answer_started = time.perf_counter()
        answer = self.llm.generate_answer(question, sources)
        trace.append(
            _event(
                "Answer Agent",
                "Generate answer",
                "ok" if answer else "warning",
                "Generated answer from retrieved context.",
                answer_started,
            )
        )

        critic_started = time.perf_counter()
        supported_claims, unsupported_claims = _critique_answer(answer, sources)
        trace.append(
            _event(
                "Critic Agent",
                "Check groundedness",
                "ok" if not unsupported_claims else "warning",
                f"{len(supported_claims)} supported, {len(unsupported_claims)} unsupported claims.",
                critic_started,
            )
        )

        eval_started = time.perf_counter()
        evaluation = _evaluate(question, answer, sources, supported_claims, unsupported_claims)
        trace.append(
            _event(
                "Evaluation Agent",
                "Score answer quality",
                "ok" if evaluation.hallucination_risk != "high" else "warning",
                (
                    f"Relevance {evaluation.relevance_score}, "
                    f"groundedness {evaluation.groundedness_score}, "
                    f"risk {evaluation.hallucination_risk}."
                ),
                eval_started,
            )
        )

        report_started = time.perf_counter()
        total_ms = int((time.perf_counter() - started_at) * 1000)
        trace.append(
            _event(
                "Report Agent",
                "Assemble response",
                "ok",
                "Prepared answer, sources, metrics, and trace.",
                report_started,
            )
        )

        return QueryResponse(
            question=question,
            answer=answer,
            sources=sources,
            evaluation=evaluation,
            agent_trace=trace,
            latency_ms=total_ms,
        )


def _event(
    agent: str,
    action: str,
    status: str,
    detail: str,
    started_at: float,
) -> AgentTraceEvent:
    return AgentTraceEvent(
        agent=agent,
        action=action,
        status=status,  # type: ignore[arg-type]
        detail=detail,
        duration_ms=int((time.perf_counter() - started_at) * 1000),
    )


def _critique_answer(answer: str, sources: list[SourceChunk]) -> tuple[list[str], list[str]]:
    source_terms = set(tokenize(" ".join(source.text for source in sources)))
    if not source_terms:
        return [], _claims(answer)

    supported: list[str] = []
    unsupported: list[str] = []
    for claim in _claims(answer):
        claim_terms = set(tokenize(claim))
        if not claim_terms:
            continue
        overlap_ratio = len(claim_terms & source_terms) / max(len(claim_terms), 1)
        if overlap_ratio >= 0.3:
            supported.append(claim)
        else:
            unsupported.append(claim)
    return supported, unsupported


def _evaluate(
    question: str,
    answer: str,
    sources: list[SourceChunk],
    supported_claims: list[str],
    unsupported_claims: list[str],
) -> EvaluationResult:
    relevance_score = _relevance_score(sources)
    groundedness_score = _groundedness_score(supported_claims, unsupported_claims)
    completeness_score = _completeness_score(question, answer, sources)

    if not sources or groundedness_score < 45:
        hallucination_risk = "high"
    elif groundedness_score < 75 or unsupported_claims:
        hallucination_risk = "medium"
    else:
        hallucination_risk = "low"

    missing_evidence = unsupported_claims[:5]
    if not sources:
        critic_notes = "No sources were retrieved, so the answer cannot be grounded."
    elif unsupported_claims:
        critic_notes = "Some answer claims have weak or missing support in the retrieved context."
    else:
        critic_notes = "The answer is supported by the retrieved context."

    return EvaluationResult(
        relevance_score=relevance_score,
        groundedness_score=groundedness_score,
        completeness_score=completeness_score,
        hallucination_risk=hallucination_risk,  # type: ignore[arg-type]
        supported_claims=supported_claims,
        unsupported_claims=unsupported_claims,
        missing_evidence=missing_evidence,
        critic_notes=critic_notes,
    )


def _claims(answer: str) -> list[str]:
    claims: list[str] = []
    for line in answer.splitlines():
        claim = line.strip().lstrip("- ").strip()
        if not claim:
            continue
        if claim.lower().startswith("based on the indexed documents"):
            continue
        claim = re.sub(r"\s*\([^)]*\)\s*$", "", claim).strip()
        if claim.startswith("(") and claim.endswith(")"):
            continue
        if len(claim) > 24:
            claims.append(claim)
    return claims


def _relevance_score(sources: list[SourceChunk]) -> int:
    if not sources:
        return 0
    top_scores = [source.score for source in sources[:3]]
    average = sum(top_scores) / len(top_scores)
    return min(100, round((average * 240) + 10))


def _groundedness_score(supported_claims: list[str], unsupported_claims: list[str]) -> int:
    total = len(supported_claims) + len(unsupported_claims)
    if total == 0:
        return 0
    return round((len(supported_claims) / total) * 100)


def _completeness_score(question: str, answer: str, sources: list[SourceChunk]) -> int:
    question_terms = set(tokenize(question))
    if not question_terms:
        return 0

    answer_terms = set(tokenize(answer))
    source_terms = set(tokenize(" ".join(source.text for source in sources)))
    covered = question_terms & (answer_terms | source_terms)
    return round((len(covered) / len(question_terms)) * 100)
