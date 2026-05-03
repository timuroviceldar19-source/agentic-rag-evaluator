from app.core.config import Settings
from app.models.schemas import (
    GenerationMode,
    PipelineEngine,
    QueryComparisonRequest,
    QueryComparisonResponse,
    QueryComparisonRun,
)
from app.services.pipeline_factory import create_pipeline
from app.services.vector_store import VectorStore


def run_query_comparison(
    vector_store: VectorStore,
    settings: Settings,
    request: QueryComparisonRequest,
) -> QueryComparisonResponse:
    runs: list[QueryComparisonRun] = []
    for config in request.configs:
        top_k = config.top_k or request.top_k
        run_settings = settings.model_copy(
            update={
                "pipeline_engine": config.pipeline_engine,
                "openai_model": config.openai_model or settings.openai_model,
            }
        )
        response = create_pipeline(vector_store, run_settings).run(
            question=request.question,
            top_k=top_k,
        )
        generation_mode: GenerationMode = (
            "openai" if run_settings.openai_api_key else "local_fallback"
        )
        model = run_settings.openai_model if generation_mode == "openai" else "local-extractive"

        runs.append(
            QueryComparisonRun(
                label=config.label or _default_label(config.pipeline_engine),
                pipeline_engine=config.pipeline_engine,
                model=model,
                generation_mode=generation_mode,
                top_k=top_k,
                response=response,
            )
        )

    return QueryComparisonResponse(question=request.question, runs=runs)


def _default_label(pipeline_engine: PipelineEngine) -> str:
    if pipeline_engine == "langgraph":
        return "LangGraph"
    return "Linear baseline"
