import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.models.schemas import (
    GenerationMode,
    PipelineEngine,
    QueryHistoryItem,
    QueryResponse,
    QueryRunType,
)


class QueryHistoryStore:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.backend = _database_backend(database_url)
        self.placeholder = "%s" if self.backend == "postgres" else "?"
        self._schema_ready = False

    def save_run(
        self,
        response: QueryResponse,
        pipeline_engine: PipelineEngine,
        generation_mode: GenerationMode,
        model: str,
        run_label: str,
        run_type: QueryRunType,
    ) -> QueryHistoryItem:
        self._ensure_schema()
        run_id = str(uuid4())
        created_at = datetime.now(UTC).isoformat()
        response_json = json.dumps(response.model_dump(), ensure_ascii=False)
        evaluation = response.evaluation
        usage = response.usage

        values = (
            run_id,
            created_at,
            response.question,
            response.answer,
            evaluation.relevance_score,
            evaluation.groundedness_score,
            evaluation.completeness_score,
            evaluation.hallucination_risk,
            len(response.sources),
            response.latency_ms,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
            usage.estimated_cost_usd,
            pipeline_engine,
            generation_mode,
            model,
            run_label,
            run_type,
            response_json,
        )
        placeholders = ", ".join([self.placeholder] * len(values))
        with self._connect() as connection:
            connection.execute(
                (
                    "INSERT INTO query_runs ("
                    "run_id, created_at, question, answer, relevance_score, "
                    "groundedness_score, completeness_score, hallucination_risk, "
                    "source_count, latency_ms, prompt_tokens, completion_tokens, "
                    "total_tokens, estimated_cost_usd, pipeline_engine, generation_mode, "
                    "model, run_label, run_type, response_json"
                    f") VALUES ({placeholders})"
                ),
                values,
            )
            connection.commit()

        return QueryHistoryItem(
            run_id=run_id,
            created_at=created_at,
            question=response.question,
            answer=response.answer,
            relevance_score=evaluation.relevance_score,
            groundedness_score=evaluation.groundedness_score,
            completeness_score=evaluation.completeness_score,
            hallucination_risk=evaluation.hallucination_risk,
            source_count=len(response.sources),
            latency_ms=response.latency_ms,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            estimated_cost_usd=usage.estimated_cost_usd,
            pipeline_engine=pipeline_engine,
            generation_mode=generation_mode,
            model=model,
            run_label=run_label,
            run_type=run_type,
            response=response,
        )

    def list_recent(self, limit: int = 10) -> list[QueryHistoryItem]:
        self._ensure_schema()
        with self._connect() as connection:
            cursor = connection.execute(
                (
                    "SELECT run_id, created_at, question, answer, relevance_score, "
                    "groundedness_score, completeness_score, hallucination_risk, "
                    "source_count, latency_ms, prompt_tokens, completion_tokens, "
                    "total_tokens, estimated_cost_usd, pipeline_engine, generation_mode, "
                    "model, run_label, run_type, response_json "
                    f"FROM query_runs ORDER BY created_at DESC LIMIT {self.placeholder}"
                ),
                (limit,),
            )
            rows = _rows_as_dicts(cursor)

        return [_history_item_from_row(row) for row in rows]

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS query_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    relevance_score INTEGER NOT NULL,
                    groundedness_score INTEGER NOT NULL,
                    completeness_score INTEGER NOT NULL,
                    hallucination_risk TEXT NOT NULL,
                    source_count INTEGER NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_cost_usd REAL NOT NULL DEFAULT 0,
                    pipeline_engine TEXT NOT NULL,
                    generation_mode TEXT NOT NULL,
                    model TEXT NOT NULL,
                    run_label TEXT NOT NULL,
                    run_type TEXT NOT NULL,
                    response_json TEXT NOT NULL
                )
                """
            )
            self._ensure_usage_columns(connection)
            connection.commit()
        self._schema_ready = True

    def _ensure_usage_columns(self, connection: Any) -> None:
        existing_columns = self._column_names(connection)
        for name, definition in {
            "prompt_tokens": "INTEGER NOT NULL DEFAULT 0",
            "completion_tokens": "INTEGER NOT NULL DEFAULT 0",
            "total_tokens": "INTEGER NOT NULL DEFAULT 0",
            "estimated_cost_usd": "REAL NOT NULL DEFAULT 0",
        }.items():
            if name not in existing_columns:
                connection.execute(f"ALTER TABLE query_runs ADD COLUMN {name} {definition}")

    def _column_names(self, connection: Any) -> set[str]:
        if self.backend == "postgres":
            cursor = connection.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
                ("query_runs",),
            )
            return {row[0] for row in cursor.fetchall()}

        cursor = connection.execute("PRAGMA table_info(query_runs)")
        return {row[1] for row in cursor.fetchall()}

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        if self.backend == "postgres":
            import psycopg

            connection = psycopg.connect(self.database_url)
        else:
            sqlite_path = _sqlite_path(self.database_url)
            if sqlite_path != ":memory:":
                Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(sqlite_path)
            connection.row_factory = sqlite3.Row

        try:
            yield connection
        finally:
            connection.close()


def create_query_history_store(database_url: str) -> QueryHistoryStore:
    return QueryHistoryStore(database_url)


def _history_item_from_row(row: dict[str, Any]) -> QueryHistoryItem:
    response = QueryResponse.model_validate(json.loads(row["response_json"]))
    return QueryHistoryItem(
        run_id=row["run_id"],
        created_at=row["created_at"],
        question=row["question"],
        answer=row["answer"],
        relevance_score=int(row["relevance_score"]),
        groundedness_score=int(row["groundedness_score"]),
        completeness_score=int(row["completeness_score"]),
        hallucination_risk=row["hallucination_risk"],
        source_count=int(row["source_count"]),
        latency_ms=int(row["latency_ms"]),
        prompt_tokens=int(row["prompt_tokens"]),
        completion_tokens=int(row["completion_tokens"]),
        total_tokens=int(row["total_tokens"]),
        estimated_cost_usd=float(row["estimated_cost_usd"]),
        pipeline_engine=row["pipeline_engine"],
        generation_mode=row["generation_mode"],
        model=row["model"],
        run_label=row["run_label"],
        run_type=row["run_type"],
        response=response,
    )


def _rows_as_dicts(cursor: Any) -> list[dict[str, Any]]:
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]


def _database_backend(database_url: str) -> str:
    if database_url.startswith(("postgres://", "postgresql://")):
        return "postgres"
    if database_url.startswith("sqlite:///"):
        return "sqlite"
    raise ValueError("DATABASE_URL must start with postgresql://, postgres://, or sqlite:///")


def _sqlite_path(database_url: str) -> str:
    path = database_url.removeprefix("sqlite:///")
    if path == ":memory:":
        return path
    return str(Path(path))
