from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse

from app.core.config import get_settings
from app.models.schemas import (
    DocumentInfo,
    GenerationMode,
    PipelineEngine,
    QueryComparisonRequest,
    QueryComparisonResponse,
    QueryHistoryItem,
    QueryRequest,
    QueryResponse,
    QueryRunType,
    UploadResponse,
)
from app.services.comparison import run_query_comparison
from app.services.document_loader import load_document
from app.services.pipeline_factory import create_pipeline
from app.services.query_history import create_query_history_store
from app.services.vector_store import create_vector_store


router = APIRouter()
settings = get_settings()
vector_store = create_vector_store(settings)
pipeline = create_pipeline(vector_store, settings)
history_store = create_query_history_store(settings.query_history_database_url)


@router.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "app": settings.app_name,
        "vector_store": settings.vector_store,
    }


@router.get("/", include_in_schema=False)
def docs_redirect() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    return vector_store.list_documents()


@router.get("/history", response_model=list[QueryHistoryItem])
def list_query_history(limit: int = Query(default=10, ge=1, le=50)) -> list[QueryHistoryItem]:
    return history_store.list_recent(limit=limit)


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)) -> UploadResponse:
    uploaded: list[DocumentInfo] = []
    for file in files:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"{file.filename} is empty")
        try:
            file_type, pages = load_document(file.filename or "document", content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if not pages:
            raise HTTPException(status_code=400, detail=f"{file.filename} has no readable text")
        uploaded.append(vector_store.add_document(file.filename or "document", file_type, pages))

    return UploadResponse(documents=uploaded)


@router.delete("/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, bool]:
    deleted = vector_store.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": True}


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    response = pipeline.run(question=request.question, top_k=request.top_k)
    _save_history(
        response=response,
        pipeline_engine=settings.pipeline_engine,
        generation_mode=_generation_mode(),
        model=_model_name(),
        run_label=settings.pipeline_engine,
        run_type="single",
    )
    return response


@router.post("/query/compare", response_model=QueryComparisonResponse)
def compare_query(request: QueryComparisonRequest) -> QueryComparisonResponse:
    response = run_query_comparison(vector_store, settings, request)
    for run in response.runs:
        _save_history(
            response=run.response,
            pipeline_engine=run.pipeline_engine,
            generation_mode=run.generation_mode,
            model=run.model,
            run_label=run.label,
            run_type="comparison",
        )
    return response


@router.post("/reset")
def reset() -> dict[str, bool]:
    vector_store.reset()
    return {"reset": True}


def _save_history(
    response: QueryResponse,
    pipeline_engine: PipelineEngine,
    generation_mode: GenerationMode,
    model: str,
    run_label: str,
    run_type: QueryRunType,
) -> None:
    history_store.save_run(
        response=response,
        pipeline_engine=pipeline_engine,
        generation_mode=generation_mode,
        model=model,
        run_label=run_label,
        run_type=run_type,
    )


def _generation_mode() -> GenerationMode:
    return "openai" if settings.openai_api_key else "local_fallback"


def _model_name() -> str:
    return settings.openai_model if settings.openai_api_key else "local-extractive"
