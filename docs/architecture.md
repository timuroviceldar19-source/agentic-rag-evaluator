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
