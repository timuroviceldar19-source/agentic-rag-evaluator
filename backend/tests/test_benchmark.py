from pathlib import Path

from app.services.benchmark import (
    BenchmarkRunner,
    expected_term_coverage,
    format_benchmark_report,
    ragas_style_scores,
)
from app.services.document_loader import LoadedPage
from app.services.agents import AgenticRAGPipeline
from app.services.vector_store import VectorStore
from app.core.config import Settings


def test_benchmark_runner_returns_summary() -> None:
    dataset_path = Path("benchmarks/sample_questions.json").resolve()

    report = BenchmarkRunner(Settings(openai_api_key=None)).run(dataset_path)

    assert report.summary.question_count == 5
    assert report.summary.average_groundedness > 0
    assert report.summary.average_faithfulness > 0
    assert report.summary.average_context_precision > 0
    assert report.summary.average_context_recall > 0
    assert report.summary.high_risk_answers == 0
    assert len(report.results) == 5
    assert all(result.ragas_scores.faithfulness > 0 for result in report.results)


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


def test_ragas_style_scores_include_context_metrics(tmp_path) -> None:
    store = VectorStore(tmp_path / "store.json")
    store.add_document(
        "notes.md",
        ".md",
        [
            LoadedPage(
                page=None,
                text="RAG systems need retrieval relevance, answer groundedness, and latency.",
            )
        ],
    )
    response = AgenticRAGPipeline(store, Settings(openai_api_key=None)).run(
        "What RAG metrics matter?",
        top_k=3,
    )

    scores = ragas_style_scores(
        response,
        ["retrieval relevance", "answer groundedness", "latency"],
    )

    assert scores.faithfulness == response.evaluation.groundedness_score
    assert scores.answer_relevance == response.evaluation.relevance_score
    assert scores.context_precision == 100
    assert scores.context_recall == 100


def test_format_benchmark_report_includes_aggregate_metrics() -> None:
    dataset_path = Path("benchmarks/sample_questions.json").resolve()

    report = BenchmarkRunner(Settings(openai_api_key=None)).run(dataset_path)
    output = format_benchmark_report(report)

    assert "RAGAS-Style RAG Benchmark Results" in output
    assert "Faithfulness:" in output
    assert "Context precision:" in output
    assert "Per-Question Results" in output
