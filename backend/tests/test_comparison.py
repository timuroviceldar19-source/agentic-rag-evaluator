from pathlib import Path

from app.core.config import Settings
from app.models.schemas import QueryComparisonRequest
from app.services.comparison import run_query_comparison
from app.services.document_loader import LoadedPage
from app.services.vector_store import VectorStore


def test_comparison_runs_default_engines(tmp_path: Path) -> None:
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
    request = QueryComparisonRequest(question="What should AI Engineers know?", top_k=3)

    response = run_query_comparison(store, Settings(openai_api_key=None), request)

    assert response.question == request.question
    assert [run.pipeline_engine for run in response.runs] == ["langgraph", "linear"]
    assert {run.generation_mode for run in response.runs} == {"local_fallback"}
    assert {run.model for run in response.runs} == {"local-extractive"}
    assert all(run.response.sources for run in response.runs)
    assert all(run.response.latency_ms > 0 for run in response.runs)
    assert all(run.response.evaluation.relevance_score > 0 for run in response.runs)
