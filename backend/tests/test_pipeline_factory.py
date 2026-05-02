from pathlib import Path

from app.core.config import Settings
from app.services.agents import AgenticRAGPipeline
from app.services.graph.builder import LangGraphPipeline
from app.services.pipeline_factory import create_pipeline
from app.services.vector_store import VectorStore


def test_factory_returns_linear_when_configured(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "store.json")
    settings = Settings(openai_api_key=None, pipeline_engine="linear")
    pipeline = create_pipeline(store, settings)
    assert isinstance(pipeline, AgenticRAGPipeline)


def test_factory_returns_langgraph_by_default(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "store.json")
    settings = Settings(openai_api_key=None)
    pipeline = create_pipeline(store, settings)
    assert isinstance(pipeline, LangGraphPipeline)

