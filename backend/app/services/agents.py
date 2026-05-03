import time

from app.core.config import Settings
from app.models.schemas import (
    AgentTraceEvent,
    QueryResponse,
)
from app.services.agent_helpers import critique_answer, evaluate, make_event
from app.services.llm import LLMClient
from app.services.vector_store import VectorStore


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
            make_event(
                "Retrieval Agent",
                "Search vector store",
                "ok" if sources else "warning",
                f"Found {len(sources)} source chunks.",
                retrieval_started,
            )
        )

        answer_started = time.perf_counter()
        llm_result = self.llm.generate_answer(question, sources)
        answer = llm_result.answer
        trace.append(
            make_event(
                "Answer Agent",
                "Generate answer",
                "ok" if answer else "warning",
                "Generated answer from retrieved context.",
                answer_started,
            )
        )

        critic_started = time.perf_counter()
        supported_claims, unsupported_claims = critique_answer(answer, sources)
        trace.append(
            make_event(
                "Critic Agent",
                "Check groundedness",
                "ok" if not unsupported_claims else "warning",
                f"{len(supported_claims)} supported, {len(unsupported_claims)} unsupported claims.",
                critic_started,
            )
        )

        eval_started = time.perf_counter()
        evaluation = evaluate(question, answer, sources, supported_claims, unsupported_claims)
        trace.append(
            make_event(
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
        total_ms = max(1, int((time.perf_counter() - started_at) * 1000))
        trace.append(
            make_event(
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
            usage=llm_result.usage,
        )
