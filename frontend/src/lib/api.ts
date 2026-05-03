export type DocumentInfo = {
  document_id: string;
  name: string;
  file_type: string;
  chunk_count: number;
  uploaded_at: string;
};

export type SourceChunk = {
  chunk_id: string;
  document_id: string;
  document_name: string;
  page: number | null;
  chunk_index: number;
  text: string;
  score: number;
};

export type EvaluationResult = {
  relevance_score: number;
  groundedness_score: number;
  completeness_score: number;
  hallucination_risk: "low" | "medium" | "high";
  supported_claims: string[];
  unsupported_claims: string[];
  missing_evidence: string[];
  critic_notes: string;
};

export type AgentTraceEvent = {
  agent: string;
  action: string;
  status: "ok" | "warning" | "error";
  detail: string;
  duration_ms: number;
};

export type QueryResponse = {
  question: string;
  answer: string;
  sources: SourceChunk[];
  evaluation: EvaluationResult;
  agent_trace: AgentTraceEvent[];
  latency_ms: number;
};

export type PipelineEngine = "linear" | "langgraph";

export type GenerationMode = "openai" | "local_fallback";

export type QueryComparisonConfig = {
  pipeline_engine: PipelineEngine;
  label?: string | null;
  top_k?: number | null;
  openai_model?: string | null;
};

export type QueryComparisonRun = {
  label: string;
  pipeline_engine: PipelineEngine;
  model: string;
  generation_mode: GenerationMode;
  top_k: number;
  response: QueryResponse;
};

export type QueryComparisonResponse = {
  question: string;
  runs: QueryComparisonRun[];
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail ?? `Request failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function fetchDocuments() {
  return request<DocumentInfo[]>("/documents");
}

export function uploadDocuments(files: FileList) {
  const form = new FormData();
  Array.from(files).forEach((file) => form.append("files", file));
  return request<{ documents: DocumentInfo[] }>("/documents/upload", {
    method: "POST",
    body: form,
  });
}

export function deleteDocument(documentId: string) {
  return request<{ deleted: boolean }>(`/documents/${documentId}`, {
    method: "DELETE",
  });
}

export function askQuestion(question: string, topK: number) {
  return request<QueryResponse>("/query", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ question, top_k: topK }),
  });
}

export function compareQuestion(question: string, topK: number) {
  return request<QueryComparisonResponse>("/query/compare", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ question, top_k: topK }),
  });
}

export function resetIndex() {
  return request<{ reset: boolean }>("/reset", { method: "POST" });
}
