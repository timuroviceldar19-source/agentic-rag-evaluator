from pathlib import Path

from app.services.benchmark import BenchmarkRunner, expected_term_coverage, format_benchmark_report
from app.services.document_loader import LoadedPage
from app.services.agents import AgenticRAGPipeline
from app.services.vector_store import VectorStore
from app.core.config import Settings


def test_benchmark_runner_returns_summary() -> None:
    dataset_path = Path("benchmarks/sample_questions.json").resolve()

    report = BenchmarkRunner(Settings(openai_api_key=None)).run(dataset_path)

    assert report.summary.question_count == 5
    assert report.summary.average_groundedness > 0
    assert report.summary.high_risk_answers == 0
    assert len(report.results) == 5


def test_expected_term_coverage_reports_missing_terms(tmp_path) -> None:
    store = VectorStore(tmp_path / "store.json")
    store.add_document(
        "notes.md",
        ".md",
        [LoadedPage(page=None, text="AI systems need latency and cost observability.")],
    )
    response = AgenticRAGPipeline(store, Settings(openai_api_key=None)).run(
        "What observability metrics matter?",
        top_k=3,
    )

    score, missing = expected_term_coverage(
        response,
        ["latency", "cost", "groundedness"],
    )

    assert score == 67
    assert missing == ["groundedness"]


def test_format_benchmark_report_includes_aggregate_metrics() -> None:
    dataset_path = Path("benchmarks/sample_questions.json").resolve()

    report = BenchmarkRunner(Settings(openai_api_key=None)).run(dataset_path)
    output = format_benchmark_report(report)

    assert "RAG Benchmark Results" in output
    assert "Average groundedness:" in output
    assert "Per-question results:" in output

