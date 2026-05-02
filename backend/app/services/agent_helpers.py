import re
import time
from typing import Literal

from app.models.schemas import AgentTraceEvent, EvaluationResult, SourceChunk
from app.services.vector_store import tokenize


def make_event(
    agent: str,
    action: str,
    status: Literal["ok", "warning", "error"],
    detail: str,
    started_at: float,
) -> AgentTraceEvent:
    return AgentTraceEvent(
        agent=agent,
        action=action,
        status=status,
        detail=detail,
        duration_ms=int((time.perf_counter() - started_at) * 1000),
    )


def critique_answer(answer: str, sources: list[SourceChunk]) -> tuple[list[str], list[str]]:
    source_terms = set(tokenize(" ".join(source.text for source in sources)))
    if not source_terms:
        return [], claims(answer)

    supported: list[str] = []
    unsupported: list[str] = []
    for claim in claims(answer):
        claim_terms = set(tokenize(claim))
        if not claim_terms:
            continue
        overlap_ratio = len(claim_terms & source_terms) / max(len(claim_terms), 1)
        if overlap_ratio >= 0.3:
            supported.append(claim)
        else:
            unsupported.append(claim)
    return supported, unsupported


def evaluate(
    question: str,
    answer: str,
    sources: list[SourceChunk],
    supported_claims: list[str],
    unsupported_claims: list[str],
) -> EvaluationResult:
    relevance = relevance_score(sources)
    groundedness = groundedness_score(supported_claims, unsupported_claims)
    completeness = completeness_score(question, answer, sources)

    if not sources or groundedness < 45:
        hallucination_risk = "high"
    elif groundedness < 75 or unsupported_claims:
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
        relevance_score=relevance,
        groundedness_score=groundedness,
        completeness_score=completeness,
        hallucination_risk=hallucination_risk,
        supported_claims=supported_claims,
        unsupported_claims=unsupported_claims,
        missing_evidence=missing_evidence,
        critic_notes=critic_notes,
    )


def claims(answer: str) -> list[str]:
    parsed_claims: list[str] = []
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
            parsed_claims.append(claim)
    return parsed_claims


def relevance_score(sources: list[SourceChunk]) -> int:
    if not sources:
        return 0
    top_scores = [source.score for source in sources[:3]]
    average = sum(top_scores) / len(top_scores)
    return min(100, round((average * 240) + 10))


def groundedness_score(supported_claims: list[str], unsupported_claims: list[str]) -> int:
    total = len(supported_claims) + len(unsupported_claims)
    if total == 0:
        return 0
    return round((len(supported_claims) / total) * 100)


def completeness_score(question: str, answer: str, sources: list[SourceChunk]) -> int:
    question_terms = set(tokenize(question))
    if not question_terms:
        return 0

    answer_terms = set(tokenize(answer))
    source_terms = set(tokenize(" ".join(source.text for source in sources)))
    covered = question_terms & (answer_terms | source_terms)
    return round((len(covered) / len(question_terms)) * 100)

