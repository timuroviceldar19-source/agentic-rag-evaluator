import hashlib
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.models.schemas import DocumentInfo, SourceChunk
from app.services.chunking import chunk_text
from app.services.document_loader import LoadedPage


TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "for",
    "from",
    "has",
    "how",
    "in",
    "is",
    "it",
    "know",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 2 and token.lower() not in STOPWORDS
    ]


def create_vector_store(settings: Settings) -> "VectorStore | ChromaVectorStore":
    if settings.vector_store == "chroma":
        return ChromaVectorStore(settings.chroma_dir)
    return VectorStore(settings.vector_store_path)


def document_id_for(name: str, text: str) -> str:
    digest = hashlib.sha256(f"{name}\n{text}".encode("utf-8")).hexdigest()
    return digest[:16]


def embed_text(text: str, dimensions: int = 384) -> list[float]:
    vector = [0.0] * dimensions
    tokens = tokenize(text)
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        index = int(digest, 16) % dimensions
        vector[index] += 1.0

    length = math.sqrt(sum(value * value for value in vector))
    if length == 0:
        return vector
    return [value / length for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=False))


class VectorStore:
    def __init__(self, path: Path, dimensions: int = 384) -> None:
        self.path = path
        self.dimensions = dimensions
        self.documents: dict[str, dict[str, Any]] = {}
        self.chunks: list[dict[str, Any]] = []
        self._load()

    def list_documents(self) -> list[DocumentInfo]:
        return [
            DocumentInfo(
                document_id=document_id,
                name=document["name"],
                file_type=document["file_type"],
                chunk_count=sum(1 for chunk in self.chunks if chunk["document_id"] == document_id),
                uploaded_at=document["uploaded_at"],
            )
            for document_id, document in sorted(
                self.documents.items(),
                key=lambda item: item[1]["uploaded_at"],
                reverse=True,
            )
        ]

    def add_document(self, name: str, file_type: str, pages: list[LoadedPage]) -> DocumentInfo:
        source_text = "\n".join(page.text for page in pages)
        document_id = self._document_id(name, source_text)
        uploaded_at = datetime.now(UTC).isoformat()

        self.documents[document_id] = {
            "name": name,
            "file_type": file_type,
            "uploaded_at": uploaded_at,
        }

        self.chunks = [chunk for chunk in self.chunks if chunk["document_id"] != document_id]

        chunk_index = 0
        for page in pages:
            for text in chunk_text(page.text):
                chunk_id = f"{document_id}:{chunk_index}"
                self.chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "document_name": name,
                        "page": page.page,
                        "chunk_index": chunk_index,
                        "text": text,
                        "embedding": self._embed(text),
                    }
                )
                chunk_index += 1

        self._save()
        return DocumentInfo(
            document_id=document_id,
            name=name,
            file_type=file_type,
            chunk_count=chunk_index,
            uploaded_at=uploaded_at,
        )

    def delete_document(self, document_id: str) -> bool:
        if document_id not in self.documents:
            return False
        del self.documents[document_id]
        self.chunks = [chunk for chunk in self.chunks if chunk["document_id"] != document_id]
        self._save()
        return True

    def reset(self) -> None:
        self.documents = {}
        self.chunks = []
        self._save()

    def search(self, query: str, top_k: int = 5) -> list[SourceChunk]:
        query_embedding = self._embed(query)
        scored: list[tuple[float, dict[str, Any]]] = []
        for chunk in self.chunks:
            score = self._cosine(query_embedding, chunk["embedding"])
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            SourceChunk(
                chunk_id=chunk["chunk_id"],
                document_id=chunk["document_id"],
                document_name=chunk["document_name"],
                page=chunk["page"],
                chunk_index=chunk["chunk_index"],
                text=chunk["text"],
                score=round(min(score, 1.0), 4),
            )
            for score, chunk in scored[:top_k]
        ]

    def _document_id(self, name: str, text: str) -> str:
        return document_id_for(name, text)

    def _embed(self, text: str) -> list[float]:
        return embed_text(text, dimensions=self.dimensions)

    def _cosine(self, left: list[float], right: list[float]) -> float:
        return cosine_similarity(left, right)

    def _load(self) -> None:
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.documents = payload.get("documents", {})
        self.chunks = payload.get("chunks", [])

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": self.documents,
            "chunks": self.chunks,
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class ChromaVectorStore:
    def __init__(self, path: Path, dimensions: int = 384) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "ChromaDB is optional. Install it with "
                "`pip install -r requirements-chroma.txt` or set VECTOR_STORE=local."
            ) from exc

        self.path = path
        self.dimensions = dimensions
        self.documents_path = self.path / "documents.json"
        self.documents: dict[str, dict[str, Any]] = {}
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection = self.client.get_or_create_collection(
            name="rag_chunks",
            metadata={"hnsw:space": "cosine"},
        )
        self._load_documents()

    def list_documents(self) -> list[DocumentInfo]:
        chunk_counts = self._chunk_counts()
        return [
            DocumentInfo(
                document_id=document_id,
                name=document["name"],
                file_type=document["file_type"],
                chunk_count=chunk_counts.get(document_id, 0),
                uploaded_at=document["uploaded_at"],
            )
            for document_id, document in sorted(
                self.documents.items(),
                key=lambda item: item[1]["uploaded_at"],
                reverse=True,
            )
        ]

    def add_document(self, name: str, file_type: str, pages: list[LoadedPage]) -> DocumentInfo:
        source_text = "\n".join(page.text for page in pages)
        document_id = document_id_for(name, source_text)
        uploaded_at = datetime.now(UTC).isoformat()
        self.documents[document_id] = {
            "name": name,
            "file_type": file_type,
            "uploaded_at": uploaded_at,
        }

        self.collection.delete(where={"document_id": document_id})

        ids: list[str] = []
        texts: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, str | int]] = []

        chunk_index = 0
        for page in pages:
            for text in chunk_text(page.text):
                chunk_id = f"{document_id}:{chunk_index}"
                ids.append(chunk_id)
                texts.append(text)
                embeddings.append(embed_text(text, dimensions=self.dimensions))
                metadatas.append(
                    {
                        "document_id": document_id,
                        "document_name": name,
                        "page": page.page if page.page is not None else -1,
                        "chunk_index": chunk_index,
                    }
                )
                chunk_index += 1

        if ids:
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        self._save_documents()

        return DocumentInfo(
            document_id=document_id,
            name=name,
            file_type=file_type,
            chunk_count=chunk_index,
            uploaded_at=uploaded_at,
        )

    def delete_document(self, document_id: str) -> bool:
        if document_id not in self.documents:
            return False
        del self.documents[document_id]
        self.collection.delete(where={"document_id": document_id})
        self._save_documents()
        return True

    def reset(self) -> None:
        self.documents = {}
        try:
            self.client.delete_collection("rag_chunks")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name="rag_chunks",
            metadata={"hnsw:space": "cosine"},
        )
        self._save_documents()

    def search(self, query: str, top_k: int = 5) -> list[SourceChunk]:
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[embed_text(query, dimensions=self.dimensions)],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = results.get("ids", [[]])[0]
        texts = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        sources: list[SourceChunk] = []
        for chunk_id, text, metadata, distance in zip(
            ids,
            texts,
            metadatas,
            distances,
            strict=False,
        ):
            page_value = metadata.get("page", -1)
            page = None if page_value == -1 else int(page_value)
            score = max(0.0, min(1.0, 1.0 - float(distance)))
            sources.append(
                SourceChunk(
                    chunk_id=chunk_id,
                    document_id=str(metadata["document_id"]),
                    document_name=str(metadata["document_name"]),
                    page=page,
                    chunk_index=int(metadata["chunk_index"]),
                    text=text,
                    score=round(score, 4),
                )
            )
        return sources

    def _chunk_counts(self) -> dict[str, int]:
        results = self.collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for metadata in results.get("metadatas", []):
            if not metadata:
                continue
            document_id = str(metadata["document_id"])
            counts[document_id] = counts.get(document_id, 0) + 1
        return counts

    def _load_documents(self) -> None:
        if not self.documents_path.exists():
            return
        self.documents = json.loads(self.documents_path.read_text(encoding="utf-8"))

    def _save_documents(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        self.documents_path.write_text(json.dumps(self.documents, indent=2), encoding="utf-8")
