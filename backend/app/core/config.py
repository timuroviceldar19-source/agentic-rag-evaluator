from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Agentic RAG Evaluator"
    allowed_origins: Annotated[list[str], NoDecode] = Field(
        default=["http://localhost:5173", "http://127.0.0.1:5173"]
    )
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    vector_store: Literal["local", "chroma"] = "local"
    pipeline_engine: Literal["linear", "langgraph"] = "langgraph"
    database_url: str | None = None
    openai_input_cost_per_1m_tokens: float = 0.0
    openai_output_cost_per_1m_tokens: float = 0.0

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _split_origins(cls, value):
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

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

    @property
    def query_history_database_url(self) -> str:
        return self.database_url or f"sqlite:///{self.data_dir / 'query_history.db'}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
