from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Agentic RAG Evaluator"
    allowed_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    vector_store: Literal["local", "chroma"] = "local"
    pipeline_engine: Literal["linear", "langgraph"] = "langgraph"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def backend_dir(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def data_dir(self) -> Path:
        return self.backend_dir / "data"

    @property
    def vector_store_path(self) -> Path:
        return self.data_dir / "vector_store.json"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"


@lru_cache
def get_settings() -> Settings:
    return Settings()
