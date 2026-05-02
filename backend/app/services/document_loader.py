from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


@dataclass(frozen=True)
class LoadedPage:
    page: int | None
    text: str


def load_document(filename: str, content: bytes) -> tuple[str, list[LoadedPage]]:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{extension}'. Supported: {supported}")

    if extension == ".pdf":
        return extension, _load_pdf(content)

    return extension, [LoadedPage(page=None, text=_decode_text(content))]


def _load_pdf(content: bytes) -> list[LoadedPage]:
    reader = PdfReader(BytesIO(content))
    pages: list[LoadedPage] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(LoadedPage(page=index, text=text))
    return pages


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="ignore")

