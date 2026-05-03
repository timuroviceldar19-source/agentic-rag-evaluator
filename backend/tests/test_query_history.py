import sqlite3
from pathlib import Path

from app.models.schemas import (
    AgentTraceEvent,
    EvaluationResult,
    QueryResponse,
    SourceChunk,
    TokenUsage,
)
from app.services.query_history import QueryHistoryStore


def test_query_history_persists_full_response(tmp_path: Path) -> None:
    store = QueryHistoryStore(f"sqlite:///{tmp_path / 'history.db'}")
    response = _query_response()

    saved = store.save_run(
        response=response,
        pipeline_engine="langgraph",
        generation_mode="local_fallback",
        model="local-extractive",
        run_label="LangGraph",
        run_type="single",
    )

    recent = store.list_recent(limit=5)

    assert saved.run_id
    assert len(recent) == 1
    assert recent[0].question == response.question
    assert recent[0].pipeline_engine == "langgraph"
    assert recent[0].generation_mode == "local_fallback"
    assert recent[0].source_count == 1
    assert recent[0].prompt_tokens == 12
    assert recent[0].completion_tokens == 8
    assert recent[0].total_tokens == 20
    assert recent[0].estimated_cost_usd == 0.000042
    assert recent[0].response.usage.total_tokens == 20
    assert recent[0].response.sources[0].document_name == "notes.md"
    assert recent[0].response.agent_trace[0].agent == "Retrieval Agent"


def test_query_history_respects_limit(tmp_path: Path) -> None:
    store = QueryHistoryStore(f"sqlite:///{tmp_path / 'history.db'}")
    for index in range(3):
        response = _query_response(question=f"What matters for RAG #{index}?")
        store.save_run(
            response=response,
            pipeline_engine="linear",
            generation_mode="local_fallback",
            model="local-extractive",
            run_label="Linear baseline",
            run_type="comparison",
        )

    recent = store.list_recent(limit=2)

    assert len(recent) == 2
    assert all(item.run_type == "comparison" for item in recent)


def test_query_history_migrates_existing_table(tmp_path: Path) -> None:
    database_path = tmp_path / "history.db"
    connection = sqlite3.connect(database_path)
    connection.execute(
        """
        CREATE TABLE query_runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            relevance_score INTEGER NOT NULL,
            groundedness_score INTEGER NOT NULL,
            completeness_score INTEGER NOT NULL,
            hallucination_risk TEXT NOT NULL,
            source_count INTEGER NOT NULL,
            latency_ms INTEGER NOT NULL,
            pipeline_engine TEXT NOT NULL,
            generation_mode TEXT NOT NULL,
            model TEXT NOT NULL,
            run_label TEXT NOT NULL,
            run_type TEXT NOT NULL,
            response_json TEXT NOT NULL
        )
        """
    )
    connection.commit()
    connection.close()

    store = QueryHistoryStore(f"sqlite:///{database_path}")
    store.save_run(
        response=_query_response(),
        pipeline_engine="langgraph",
        generation_mode="local_fallback",
        model="local-extractive",
        run_label="LangGraph",
        run_type="single",
    )

    recent = store.list_recent(limit=1)

    assert recent[0].total_tokens == 20


def _query_response(question: str = "What matters for RAG?") -> QueryResponse:
    return QueryResponse(
        question=question,
        answer="Grounded answers need retrieval, evaluation, and traceability.",
        sources=[
            SourceChunk(
                chunk_id="doc:0",
                document_id="doc",
                document_name="notes.md",
                page=None,
                chunk_index=0,
                text="RAG systems need retrieval, evaluation, and traceability.",
                score=0.9,
            )
        ],
        evaluation=EvaluationResult(
            relevance_score=90,
            groundedness_score=100,
            completeness_score=85,
            hallucination_risk="low",
            supported_claims=["Grounded answers need retrieval."],
            unsupported_claims=[],
            missing_evidence=[],
            critic_notes="Answer is grounded in the retrieved source.",
        ),
        agent_trace=[
            AgentTraceEvent(
                agent="Retrieval Agent",
                action="Search vector store",
                status="ok",
                detail="Found 1 source chunks.",
                duration_ms=1,
            )
        ],
        latency_ms=7,
        usage=TokenUsage(
            prompt_tokens=12,
            completion_tokens=8,
            total_tokens=20,
            estimated_cost_usd=0.000042,
        ),
    )
