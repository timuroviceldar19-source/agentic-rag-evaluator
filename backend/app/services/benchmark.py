import json
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from app.core.config import Settings
from app.models.schemas import QueryResponse
from app.services.document_loader import load_document
from app.services.pipeline_factory import Pipeline, create_pipeline
from app.services.vector_store import VectorStore, tokenize


class BenchmarkQuestion(BaseModel):
    id: str
    question: str
    expected_terms: list[str] = Field(default_factory=list)


class BenchmarkDataset(BaseModel):
    name: str
    description: str
    document: str
    top_k: int = Field(default=5, ge=1, le=12)
    questions: list[BenchmarkQuestion]


class BenchmarkQuestionResult(BaseModel):
    id: str
    question: str
    answer: str
    relevance_score: int
    groundedness_score: int
    completeness_score: int
    expected_coverage_score: int
    hallucination_risk: str
    source_count: int
    latency_ms: int
    missing_expected_terms: list[str]


class BenchmarkSummary(BaseModel):
    dataset_name: str
    question_count: int
    average_relevance: int
    average_groundedness: int
    average_completeness: int
    average_expected_coverage: int
    high_risk_answers: int
    total_latency_ms: int


class BenchmarkReport(BaseModel):
    summary: BenchmarkSummary
    results: list[BenchmarkQuestionResult]


class BenchmarkRunner:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings(openai_api_key=None, vector_store="local")

    def run(self, dataset_path: Path) -> BenchmarkReport:
        dataset = self._load_dataset(dataset_path)
        document_path = self._resolve_document_path(dataset_path, dataset.document)

        with tempfile.TemporaryDirectory() as temp_dir:
            store = VectorStore(Path(temp_dir) / "benchmark_store.json")
            self._index_document(store, document_path)
            pipeline = create_pipeline(store, self.settings)

            results = [
                self._run_question(pipeline, question, dataset.top_k)
                for question in dataset.questions
            ]

        return BenchmarkReport(
            summary=self._summarize(dataset.name, results),
            results=results,
        )

    def _load_dataset(self, dataset_path: Path) -> BenchmarkDataset:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        return BenchmarkDataset.model_validate(payload)

    def _resolve_document_path(self, dataset_path: Path, document: str) -> Path:
        path = (dataset_path.parent / document).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Benchmark document not found: {path}")
        return path

    def _index_document(self, store: VectorStore, document_path: Path) -> None:
        content = document_path.read_bytes()
        file_type, pages = load_document(document_path.name, content)
        store.add_document(document_path.name, file_type, pages)

    def _run_question(
        self,
        pipeline: Pipeline,
        question: BenchmarkQuestion,
        top_k: int,
    ) -> BenchmarkQuestionResult:
        response = pipeline.run(question.question, top_k=top_k)
        coverage_score, missing_terms = expected_term_coverage(
            response,
            question.expected_terms,
        )

        return BenchmarkQuestionResult(
            id=question.id,
            question=question.question,
            answer=response.answer,
            relevance_score=response.evaluation.relevance_score,
            groundedness_score=response.evaluation.groundedness_score,
            completeness_score=response.evaluation.completeness_score,
            expected_coverage_score=coverage_score,
            hallucination_risk=response.evaluation.hallucination_risk,
            source_count=len(response.sources),
            latency_ms=response.latency_ms,
            missing_expected_terms=missing_terms,
        )

    def _summarize(
        self,
        dataset_name: str,
        results: list[BenchmarkQuestionResult],
    ) -> BenchmarkSummary:
        return BenchmarkSummary(
            dataset_name=dataset_name,
            question_count=len(results),
            average_relevance=_average([result.relevance_score for result in results]),
            average_groundedness=_average([result.groundedness_score for result in results]),
            average_completeness=_average([result.completeness_score for result in results]),
            average_expected_coverage=_average(
                [result.expected_coverage_score for result in results]
            ),
            high_risk_answers=sum(1 for result in results if result.hallucination_risk == "high"),
            total_latency_ms=sum(result.latency_ms for result in results),
        )


def expected_term_coverage(
    response: QueryResponse,
    expected_terms: list[str],
) -> tuple[int, list[str]]:
    if not expected_terms:
        return 100, []

    answer_and_sources = " ".join(
        [
            response.answer,
            *[source.text for source in response.sources],
        ]
    ).lower()
    token_set = set(tokenize(answer_and_sources))

    missing_terms = [
        term
        for term in expected_terms
        if not _term_present(term, answer_and_sources, token_set)
    ]
    covered = len(expected_terms) - len(missing_terms)
    return round((covered / len(expected_terms)) * 100), missing_terms


def format_benchmark_report(report: BenchmarkReport) -> str:
    summary = report.summary
    lines = [
        "RAG Benchmark Results",
        f"Dataset: {summary.dataset_name}",
        f"Questions: {summary.question_count}",
        f"Average relevance: {summary.average_relevance}%",
        f"Average groundedness: {summary.average_groundedness}%",
        f"Average completeness: {summary.average_completeness}%",
        f"Expected term coverage: {summary.average_expected_coverage}%",
        f"High-risk answers: {summary.high_risk_answers}",
        f"Total latency: {summary.total_latency_ms} ms",
        "",
        "Per-question results:",
    ]

    for result in report.results:
        missing = ", ".join(result.missing_expected_terms) or "none"
        lines.append(
            "- "
            f"{result.id}: relevance={result.relevance_score}%, "
            f"groundedness={result.groundedness_score}%, "
            f"coverage={result.expected_coverage_score}%, "
            f"risk={result.hallucination_risk}, "
            f"missing={missing}"
        )
    return "\n".join(lines)


def _term_present(term: str, text: str, tokens: set[str]) -> bool:
    normalized = term.lower().strip()
    if " " in normalized:
        return normalized in text
    return normalized in tokens


def _average(values: list[int]) -> int:
    if not values:
        return 0
    return round(sum(values) / len(values))
