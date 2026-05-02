# Roadmap

This roadmap tracks the next product and engineering milestones for Agentic RAG Evaluator.

## 1. Add LangGraph Orchestration

### Goal

Replace the current linear agent pipeline with a LangGraph-powered workflow while keeping the existing API response shape stable.

### Why It Matters

LangGraph makes the agent flow explicit, easier to visualize, and closer to production agent orchestration patterns.

### Scope

- Add LangGraph as an optional orchestration layer.
- Model nodes for retrieval, answer generation, critique, evaluation, and report generation.
- Preserve the existing `/query` contract.
- Add tests for the graph workflow.

### Acceptance Criteria

- `/query` still returns answer, sources, evaluation, and agent trace.
- The graph can be run locally with the existing sample document.
- README or architecture docs explain the orchestration choice.

## 2. Add RAGAS-Style Evaluation Benchmark

### Goal

Add a small benchmark workflow for evaluating RAG answer quality across a repeatable set of questions.

### Why It Matters

A portfolio RAG project is stronger when it shows repeatable evaluation, not only one-off answers.

### Scope

- Add a sample benchmark dataset with questions, expected evidence, and reference notes.
- Add metrics inspired by RAGAS: faithfulness, answer relevance, context precision, and context recall.
- Add a CLI command or test script to run the benchmark.
- Store benchmark results as JSON or Markdown.

### Acceptance Criteria

- Benchmark runs from the command line.
- Results include per-question and aggregate metrics.
- README documents how to run the benchmark.

## 3. Add PostgreSQL Query History

### Goal

Persist user questions, retrieved sources, answers, evaluation scores, latency, and agent traces in PostgreSQL.

### Why It Matters

Production AI systems need observability and history so teams can inspect failures, compare answers, and improve prompts over time.

### Scope

- Add PostgreSQL service to Docker Compose.
- Add SQLAlchemy or SQLModel models for query runs.
- Store request, response, metrics, and trace metadata.
- Add an endpoint to list previous query runs.

### Acceptance Criteria

- Query history survives backend restart.
- Dashboard can display recent runs or backend exposes them via API.
- Tests cover persistence logic.

## 4. Add Model Comparison View

### Goal

Allow users to compare answers and evaluation scores across different LLM or retrieval configurations.

### Why It Matters

AI engineers often need to choose between models based on quality, latency, and cost, not intuition.

### Scope

- Add backend support for selecting model/config per query.
- Add UI view comparing answer, sources, latency, and evaluation scores.
- Track whether the response came from OpenAI or local fallback.
- Prepare the interface for future cost tracking.

### Acceptance Criteria

- User can run the same question through at least two configurations.
- UI shows side-by-side answer and metrics.
- Results are clearly labeled by model/config.

## 5. Deploy Public Demo

### Goal

Deploy a public demo so recruiters and engineers can try the app without cloning the repository.

### Why It Matters

A working demo lowers friction and makes the project easier to evaluate quickly.

### Scope

- Choose a hosting target for frontend and backend.
- Configure environment variables securely.
- Use local fallback mode if no hosted LLM key is available.
- Add deployment instructions and demo URL to README.

### Acceptance Criteria

- Public URL opens the dashboard.
- Sample document workflow works in the deployed environment.
- README includes the demo link and deployment notes.

