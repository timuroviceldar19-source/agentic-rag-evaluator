from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.models.schemas import DocumentInfo, QueryRequest, QueryResponse, UploadResponse
from app.services.document_loader import load_document
from app.services.pipeline_factory import create_pipeline
from app.services.vector_store import create_vector_store


router = APIRouter()
settings = get_settings()
vector_store = create_vector_store(settings)
pipeline = create_pipeline(vector_store, settings)


@router.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "app": settings.app_name,
        "vector_store": settings.vector_store,
    }


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    return vector_store.list_documents()


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
    return pipeline.run(question=request.question, top_k=request.top_k)


@router.post("/reset")
def reset() -> dict[str, bool]:
    vector_store.reset()
    return {"reset": True}
