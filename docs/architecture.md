# Architecture Notes

## Design Goal

The goal is not to create another generic chatbot. The system should behave like a small AI quality platform: every answer is paired with evidence, evaluation scores, and an agent trace.

## Agents

### Retrieval Agent

Searches the vector store for document chunks that match the user question.

### Answer Agent

Generates the final answer from the retrieved context. It uses OpenAI when configured and a local extractive fallback otherwise.

### Critic Agent

Splits the answer into claims and checks whether each claim is supported by retrieved context.

### Evaluation Agent

Turns the retrieval and critique results into user-facing metrics:

- relevance score
- groundedness score
- completeness score
- hallucination risk

### Report Agent

Packages the answer, sources, scores, and trace into a response object for the dashboard.

## Trade-Offs

The default `VECTOR_STORE=local` backend uses a local hashed vector store instead of a hosted vector database. This keeps the project easy to run in interviews and demos.

Set `VECTOR_STORE=chroma` to use the ChromaDB-backed implementation. The service layer keeps the same interface, so a production version could also swap this layer for Qdrant, Weaviate, or pgvector without changing the agent pipeline.

## Orchestration

The pipeline that turns a question into a graded answer can run in two engines:

- `PIPELINE_ENGINE=langgraph` (default) — the same five agents wired as nodes in a `StateGraph`. The graph adds an explicit conditional edge that short-circuits the answer and critic nodes when retrieval returns nothing, while preserving the five-event agent trace expected by the dashboard.
- `PIPELINE_ENGINE=linear` — a straight five-step Python function. Easier to step through in a debugger and a useful baseline when comparing latencies.

Both engines write into the same `QueryResponse` schema, so the dashboard and external consumers do not observe the choice. The shared scoring logic lives in `app/services/agent_helpers.py`, which both engines import — the only difference is orchestration, not metrics.

Flow:

```
retrieve ──► answer ──► critique ──► evaluate ──► report
     │                                  ▲
     └─(no sources)─► skip_answer ──► skip_critique ──┘
```
