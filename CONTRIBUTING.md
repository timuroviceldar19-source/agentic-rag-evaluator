# Contributing

## Local Setup

Run the backend:

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000
```

Run the frontend:

```bash
cd frontend
npm install
npm run dev
```

## Tests

Backend:

```bash
cd backend
pytest
```

Frontend:

```bash
cd frontend
npm run build
```

## Adding A New Agent

1. Add the agent behavior in `backend/app/services/agents.py`.
2. Record its work with an `AgentTraceEvent`.
3. Add response fields in `backend/app/models/schemas.py` if the UI needs to display new output.
4. Add or update tests in `backend/tests/`.
5. Update `docs/architecture.md` when the workflow changes.

## Vector Store Backends

The default backend is `VECTOR_STORE=local`.

To test ChromaDB:

```bash
cd backend
pip install -r requirements-chroma.txt
$env:VECTOR_STORE = "chroma"
uvicorn app.main:app --reload --port 8000
```
