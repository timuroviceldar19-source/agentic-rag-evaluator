# Changelog

All notable changes to this project are documented here.

## Unreleased

### Added

- LangGraph orchestration engine. Five agents are now nodes of a `StateGraph` with an explicit conditional edge for the empty-sources case. Selectable via `PIPELINE_ENGINE`; default is `langgraph`.
- Linear engine remains available as `PIPELINE_ENGINE=linear` and as a regression baseline for benchmarks.
- Pipeline factory in `app/services/pipeline_factory.py` selects the engine.
- Tests `tests/test_graph.py` and `tests/test_pipeline_factory.py` covering the graph happy path, no-sources short-circuit, latency, schema parity with the linear engine, and factory selection.
- Model comparison endpoint `/query/compare` and dashboard view for side-by-side LangGraph versus linear baseline evaluation.

### Changed

- `app/services/agents.py` slimmed down: shared scoring logic moved to `app/services/agent_helpers.py` and is reused by both engines.
- API contract for `/query` unchanged. The agent trace still has 5 events on every call.

### Deployment

- Public demo deployed to [Vercel](https://agentic-rag-evaluator-ykzi.vercel.app) for the frontend and [Render](https://render.com) for the backend.
- `allowed_origins` setting is now configurable via the `ALLOWED_ORIGINS` env var in CSV format.
- Backend Dockerfile uses the `$PORT` env var so the same image works on Render, Fly.io, and other PaaS hosts.

### Dependencies

- Added `langgraph==0.2.60`.

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
