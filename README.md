# Agentic RAG Evaluator

![CI](https://github.com/timuroviceldar19-source/agentic-rag-evaluator/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-Dashboard-61DAFB?logo=react&logoColor=111827)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)

Production-style multi-agent RAG platform for answering questions over documents and evaluating whether the answer is grounded in evidence.

The app ingests documents, retrieves relevant chunks, generates an answer, and then runs a small multi-agent evaluation pipeline that checks whether the answer is grounded in the retrieved evidence.

## Quick Links

- [Why this project exists](#why-this-project-exists)
- [Architecture](#architecture)
- [Example run](#example-run)
- [Quick start](#quick-start)
- [Development](#development)
- [Roadmap](ROADMAP.md)

## Highlights

- Multi-agent RAG workflow with retrieval, answering, critique, evaluation, and reporting
- Source-backed answers with relevance, groundedness, completeness, and hallucination risk
- Zero-key local fallback for demos, optional OpenAI generation for stronger answers
- `VECTOR_STORE=local` by default, `VECTOR_STORE=chroma` when ChromaDB is installed
- FastAPI backend, React dashboard, Docker setup, GitHub Actions, and pytest coverage

## Screenshots

<img src="https://raw.githubusercontent.com/timuroviceldar19-source/agentic-rag-evaluator/main/docs/screenshots/01-upload-dashboard.png" alt="Document upload dashboard" width="100%">

<img src="https://raw.githubusercontent.com/timuroviceldar19-source/agentic-rag-evaluator/main/docs/screenshots/02-answer-evaluation.png" alt="Answer with evaluation scores" width="100%">

<img src="https://raw.githubusercontent.com/timuroviceldar19-source/agentic-rag-evaluator/main/docs/screenshots/03-agent-trace.png" alt="Agent trace panel" width="520">

## Why This Project Exists

Most RAG demos stop at "ask a document a question." Real AI systems need more than an answer:

- evidence-backed citations
- quality scores
- hallucination risk
- traceable agent steps
- repeatable local setup

This project is built to demonstrate practical AI engineering skills: FastAPI, vector search, document ingestion, agent orchestration, evaluation, Docker, tests, and a usable dashboard.

## Features

- Upload PDF, TXT, Markdown, or MD files
- Chunk documents and store searchable vectors
- Ask questions over the indexed knowledge base
- Retrieve source chunks with similarity scores
- Generate answers with OpenAI when `OPENAI_API_KEY` is configured
- Fall back to local extractive answers without an API key
- Run multi-agent checks:
  - Retrieval Agent
  - Answer Agent
  - Critic Agent
  - Evaluation Agent
  - Report Agent
- Show relevance, groundedness, completeness, and hallucination risk
- Display agent trace, latency, sources, and critic notes

## Architecture

```text
Frontend Dashboard
  |
FastAPI Backend
  |
Document Loader -> Chunker -> Vector Store
  |
Retrieval Agent -> Answer Agent -> Critic Agent -> Evaluation Agent -> Report Agent
```

## Tech Stack

- Backend: Python, FastAPI, Pydantic
- RAG: local hashed vector store by default, optional ChromaDB backend
- LLM: optional OpenAI API, local fallback when no key is present
- Frontend: React, TypeScript, Vite, lucide-react
- DevOps: Docker, Docker Compose, GitHub Actions
- Tests: pytest

## Example Run

Question:

```text
What skills should an AI Engineer know?
```

Answer:

```text
AI Engineer candidates should be comfortable building applied LLM systems,
not only writing prompts. Important skills include Python, FastAPI,
embeddings, vector search, retrieval augmented generation, evaluation
pipelines, and Docker.
```

Evaluation:

```text
Hallucination risk: low
Relevance: 67%
Groundedness: 100%
Completeness: 100%
```

The exact scores depend on the indexed documents and selected `top_k`.

## Quick Start

### 1. Backend

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

### Optional OpenAI Setup

Create a `.env` file in `backend/`:

```bash
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

Without a key, the app still works with a local extractive answer generator.

### Vector Store Options

By default, the backend uses a zero-dependency local vector store. This keeps the demo easy to run during interviews and CI.

To switch the same app interface to ChromaDB:

```bash
cd backend
pip install -r requirements-chroma.txt
```

Then set:

```bash
VECTOR_STORE=chroma
```

Use `VECTOR_STORE=local` to return to the built-in JSON-backed store.

For Docker, set both values before building:

```bash
VECTOR_STORE=chroma
INSTALL_CHROMA=true
docker compose up --build
```

## Docker

```bash
docker compose up --build
```

Frontend: `http://localhost:5173`  
Backend API docs: `http://localhost:8000/docs`

## API

- `GET /health`
- `GET /documents`
- `POST /documents/upload`
- `DELETE /documents/{document_id}`
- `POST /query`
- `POST /reset`

Example query:

```json
{
  "question": "What AI engineering skills are required?",
  "top_k": 5
}
```

## Development

Run backend tests:

```bash
cd backend
pytest
```

Run frontend type-check and production build:

```bash
cd frontend
npm run build
```

To add a new agent, create the logic in `backend/app/services/agents.py`, append an `AgentTraceEvent`, and expose any new output through the Pydantic response models in `backend/app/models/schemas.py`.

## Roadmap

- See [ROADMAP.md](ROADMAP.md) for issue-style milestones with scope and acceptance criteria.

- Add LangGraph orchestration
- Add RAGAS-style benchmark datasets
- Add reranking
- Add PostgreSQL history
- Add model comparison
- Add cost and token tracking
- Add MCP tool integrations
- Add authentication
- Deploy demo to a cloud provider
