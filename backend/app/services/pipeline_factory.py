from typing import Protocol

from app.core.config import Settings
from app.models.schemas import QueryResponse
from app.services.vector_store import VectorStore


class Pipeline(Protocol):
    def run(self, question: str, top_k: int) -> QueryResponse: ...


def create_pipeline(vector_store: VectorStore, settings: Settings) -> Pipeline:
    if settings.pipeline_engine == "linear":
        from app.services.agents import AgenticRAGPipeline

        return AgenticRAGPipeline(vector_store, settings)
    from app.services.graph.builder import LangGraphPipeline

    return LangGraphPipeline(vector_store, settings)

