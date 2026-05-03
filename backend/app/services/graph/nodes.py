import time
from collections.abc import Callable
from typing import Literal

from app.services.agent_helpers import critique_answer, evaluate as evaluate_answer, make_event
from app.services.graph.state import RAGState
from app.services.llm import LLMClient
from app.services.vector_store import VectorStore


def make_retrieve_node(vector_store: VectorStore) -> Callable[[RAGState], dict]:
    def retrieve(state: RAGState) -> dict:
        started_at = time.perf_counter()
        sources = vector_store.search(state["question"], top_k=state["top_k"])
        return {
            "sources": sources,
            "trace": [
                make_event(
                    "Retrieval Agent",
                    "Search vector store",
                    "ok" if sources else "warning",
                    f"Found {len(sources)} source chunks.",
                    started_at,
                )
            ],
        }

    return retrieve


def make_answer_node(llm: LLMClient) -> Callable[[RAGState], dict]:
    def answer(state: RAGState) -> dict:
        started_at = time.perf_counter()
        llm_result = llm.generate_answer(state["question"], state.get("sources", []))
        return {
            "answer": llm_result.answer,
            "usage": llm_result.usage,
            "trace": [
                make_event(
                    "Answer Agent",
                    "Generate answer",
                    "ok" if llm_result.answer else "warning",
                    "Generated answer from retrieved context.",
                    started_at,
                )
            ],
        }

    return answer


def critique(state: RAGState) -> dict:
    started_at = time.perf_counter()
    supported_claims, unsupported_claims = critique_answer(
        state.get("answer", ""),
        state.get("sources", []),
    )
    return {
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "trace": [
            make_event(
                "Critic Agent",
                "Check groundedness",
                "ok" if not unsupported_claims else "warning",
                f"{len(supported_claims)} supported, {len(unsupported_claims)} unsupported claims.",
                started_at,
            )
        ],
    }


def evaluate(state: RAGState) -> dict:
    started_at = time.perf_counter()
    evaluation = evaluate_answer(
        state["question"],
        state.get("answer", ""),
        state.get("sources", []),
        state.get("supported_claims", []),
        state.get("unsupported_claims", []),
    )
    return {
        "evaluation": evaluation,
        "trace": [
            make_event(
                "Evaluation Agent",
                "Score answer quality",
                "ok" if evaluation.hallucination_risk != "high" else "warning",
                (
                    f"Relevance {evaluation.relevance_score}, "
                    f"groundedness {evaluation.groundedness_score}, "
                    f"risk {evaluation.hallucination_risk}."
                ),
                started_at,
            )
        ],
    }


def report(state: RAGState) -> dict:
    started_at = time.perf_counter()
    latency_ms = max(1, int((time.perf_counter() - state["started_at"]) * 1000))
    return {
        "latency_ms": latency_ms,
        "trace": [
            make_event(
                "Report Agent",
                "Assemble response",
                "ok",
                "Prepared answer, sources, metrics, and trace.",
                started_at,
            )
        ],
    }


def skip_answer(state: RAGState) -> dict:
    started_at = time.perf_counter()
    return {
        "answer": "",
        "trace": [
            make_event(
                "Answer Agent",
                "Skip answer generation",
                "warning",
                "No source chunks were retrieved, so answer generation was skipped.",
                started_at,
            )
        ],
    }


def skip_critique(state: RAGState) -> dict:
    started_at = time.perf_counter()
    return {
        "supported_claims": [],
        "unsupported_claims": [],
        "trace": [
            make_event(
                "Critic Agent",
                "Skip groundedness check",
                "warning",
                "No answer was generated, so groundedness checking was skipped.",
                started_at,
            )
        ],
    }


def route_after_retrieve(state: RAGState) -> Literal["answer", "skip_answer"]:
    if state.get("sources"):
        return "answer"
    return "skip_answer"
