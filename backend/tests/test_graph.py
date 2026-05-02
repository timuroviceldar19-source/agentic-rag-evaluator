from pathlib import Path

from app.core.config import Settings
from app.services.agents import AgenticRAGPipeline
from app.services.document_loader import LoadedPage
from app.services.graph.builder import LangGraphPipeline
from app.services.vector_store import VectorStore


EXPECTED_AGENTS = [
    "Retrieval Agent",
    "Answer Agent",
    "Critic Agent",
    "Evaluation Agent",
    "Report Agent",
]


def _populated_store(tmp_path: Path) -> VectorStore:
    store = VectorStore(tmp_path / "store.json")
    store.add_document(
        "hiring.md",
        ".md",
        [
            LoadedPage(
                page=None,
                text=(
                    "AI Engineers should know Python, FastAPI, embeddings, "
                    "RAG, vector search, evaluation, and Docker."
                ),
            )
        ],
    )
    return store


def _empty_store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "empty.json")


def test_graph_smoke_returns_query_response(tmp_path: Path) -> None:
    pipeline = LangGraphPipeline(_populated_store(tmp_path), Settings(openai_api_key=None))
    response = pipeline.run("What should AI Engineers know?", top_k=3)

    assert response.sources
    assert response.evaluation.relevance_score > 0
    assert response.evaluation.hallucination_risk in {"low", "medium", "high"}


def test_graph_trace_has_five_agents(tmp_path: Path) -> None:
    pipeline = LangGraphPipeline(_populated_store(tmp_path), Settings(openai_api_key=None))
    response = pipeline.run("What should AI Engineers know?", top_k=3)

    assert [event.agent for event in response.agent_trace] == EXPECTED_AGENTS


def test_graph_no_sources_short_circuit(tmp_path: Path) -> None:
    pipeline = LangGraphPipeline(_empty_store(tmp_path), Settings(openai_api_key=None))
    response = pipeline.run("anything?", top_k=3)

    assert response.sources == []
    assert response.answer == ""
    assert response.evaluation.hallucination_risk == "high"
    assert [event.agent for event in response.agent_trace] == EXPECTED_AGENTS

    by_agent = {event.agent: event for event in response.agent_trace}
    assert by_agent["Answer Agent"].status == "warning"
    assert by_agent["Critic Agent"].status == "warning"


def test_graph_latency_ms_is_positive(tmp_path: Path) -> None:
    pipeline = LangGraphPipeline(_populated_store(tmp_path), Settings(openai_api_key=None))
    response = pipeline.run("What should AI Engineers know?", top_k=3)

    assert response.latency_ms > 0


def test_graph_response_schema_matches_linear(tmp_path: Path) -> None:
    settings = Settings(openai_api_key=None)
    store = _populated_store(tmp_path)

    linear_response = AgenticRAGPipeline(store, settings).run(
        "What should AI Engineers know?", top_k=3
    )
    graph_response = LangGraphPipeline(store, settings).run(
        "What should AI Engineers know?", top_k=3
    )

    assert set(linear_response.model_dump().keys()) == set(graph_response.model_dump().keys())
    assert set(linear_response.evaluation.model_dump().keys()) == set(
        graph_response.evaluation.model_dump().keys()
    )
    assert [e.agent for e in linear_response.agent_trace] == [
        e.agent for e in graph_response.agent_trace
    ]

