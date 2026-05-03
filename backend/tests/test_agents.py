from app.core.config import Settings
from app.services.agents import AgenticRAGPipeline
from app.services.document_loader import LoadedPage
from app.services.llm import LLMClient
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
    assert response.usage.total_tokens == 0
    assert response.usage.estimated_cost_usd == 0


def test_llm_cost_estimate_uses_configured_rates() -> None:
    llm = LLMClient(
        Settings(
            openai_api_key=None,
            openai_input_cost_per_1m_tokens=0.15,
            openai_output_cost_per_1m_tokens=0.6,
        )
    )

    assert llm._estimate_cost(prompt_tokens=1_000, completion_tokens=500) == 0.00045
