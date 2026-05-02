from app.core.config import Settings
from app.services.agents import AgenticRAGPipeline
from app.services.document_loader import LoadedPage
from app.services.vector_store import VectorStore


def test_pipeline_returns_evaluation(tmp_path) -> None:
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

    pipeline = AgenticRAGPipeline(store, Settings(openai_api_key=None))
    response = pipeline.run("What should AI Engineers know?", top_k=3)

    assert response.sources
    assert response.evaluation.relevance_score > 0
    assert response.evaluation.hallucination_risk in {"low", "medium", "high"}

