from app.services.document_loader import LoadedPage
from app.core.config import Settings
from app.services.vector_store import VectorStore, create_vector_store


def test_vector_store_adds_and_searches_documents(tmp_path) -> None:
    store = VectorStore(tmp_path / "store.json")
    store.add_document(
        "notes.md",
        ".md",
        [
            LoadedPage(
                page=None,
                text="RAG systems use embeddings, retrieval, citations, and evaluation.",
            )
        ],
    )

    results = store.search("How do RAG systems use retrieval?", top_k=3)

    assert len(results) == 1
    assert results[0].document_name == "notes.md"
    assert results[0].score > 0


def test_create_vector_store_uses_local_backend_by_default() -> None:
    settings = Settings(vector_store="local")

    store = create_vector_store(settings)

    assert isinstance(store, VectorStore)
