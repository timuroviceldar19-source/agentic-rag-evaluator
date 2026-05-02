# Changelog

All notable changes to this project are documented here.

## v0.1.0 - 2026-05-02

Initial portfolio-ready release of Agentic RAG Evaluator.

### Added

- FastAPI backend for document ingestion, querying, and evaluation.
- React dashboard for uploads, questions, answers, sources, metrics, and agent trace.
- Multi-agent RAG pipeline:
  - Retrieval Agent
  - Answer Agent
  - Critic Agent
  - Evaluation Agent
  - Report Agent
- PDF, TXT, Markdown, and MD document upload support.
- Local zero-dependency vector store with JSON persistence.
- Optional ChromaDB backend via `VECTOR_STORE=chroma`.
- Optional OpenAI answer generation with local extractive fallback.
- Evaluation scores for relevance, groundedness, completeness, and hallucination risk.
- Source chunk display with similarity scores.
- RAG benchmark runner with sample benchmark dataset.
- Docker and Docker Compose setup.
- GitHub Actions CI for backend tests and frontend build.
- README screenshots, architecture notes, contributing guide, and project roadmap.

### Validation

- Backend tests: `pytest`
- Frontend build: `npm run build`
- Benchmark runner: `python scripts/run_benchmark.py`

