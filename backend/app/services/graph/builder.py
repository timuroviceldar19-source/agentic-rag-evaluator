import time

from langgraph.graph import END, StateGraph

from app.core.config import Settings
from app.models.schemas import QueryResponse, TokenUsage
from app.services.agent_helpers import evaluate
from app.services.graph.nodes import (
    critique,
    evaluate as evaluate_node,
    make_answer_node,
    make_retrieve_node,
    report,
    route_after_retrieve,
    skip_answer,
    skip_critique,
)
from app.services.graph.state import RAGState
from app.services.llm import LLMClient
from app.services.vector_store import VectorStore


class LangGraphPipeline:
    def __init__(self, vector_store: VectorStore, settings: Settings) -> None:
        self.vector_store = vector_store
        self.llm = LLMClient(settings)
        self._graph = self._build()

    def run(self, question: str, top_k: int) -> QueryResponse:
        initial_state: RAGState = {
            "question": question,
            "top_k": top_k,
            "started_at": time.perf_counter(),
            "sources": [],
            "answer": "",
            "supported_claims": [],
            "unsupported_claims": [],
            "trace": [],
        }
        final_state = self._graph.invoke(initial_state)
        evaluation = final_state.get("evaluation") or evaluate(question, "", [], [], [])

        return QueryResponse(
            question=question,
            answer=final_state.get("answer", ""),
            sources=final_state.get("sources", []),
            evaluation=evaluation,
            agent_trace=final_state.get("trace", []),
            latency_ms=final_state.get("latency_ms", 0),
            usage=final_state.get("usage", TokenUsage()),
        )

    def _build(self):
        graph = StateGraph(RAGState)
        graph.add_node("retrieve_node", make_retrieve_node(self.vector_store))
        graph.add_node("answer_node", make_answer_node(self.llm))
        graph.add_node("critique_node", critique)
        graph.add_node("evaluate_node", evaluate_node)
        graph.add_node("report_node", report)
        graph.add_node("skip_answer", skip_answer)
        graph.add_node("skip_critique", skip_critique)

        graph.set_entry_point("retrieve_node")
        graph.add_conditional_edges(
            "retrieve_node",
            route_after_retrieve,
            {
                "answer": "answer_node",
                "skip_answer": "skip_answer",
            },
        )
        graph.add_edge("answer_node", "critique_node")
        graph.add_edge("critique_node", "evaluate_node")
        graph.add_edge("skip_answer", "skip_critique")
        graph.add_edge("skip_critique", "evaluate_node")
        graph.add_edge("evaluate_node", "report_node")
        graph.add_edge("report_node", END)
        return graph.compile()
