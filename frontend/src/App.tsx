import {
  Activity,
  AlertTriangle,
  Brain,
  CheckCircle2,
  Database,
  FileText,
  GitCompareArrows,
  History,
  Loader2,
  Search,
  Send,
  ShieldCheck,
  Trash2,
  UploadCloud,
} from "lucide-react";
import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";
import {
  AgentTraceEvent,
  DocumentInfo,
  EvaluationResult,
  QueryComparisonResponse,
  QueryComparisonRun,
  QueryHistoryItem,
  QueryResponse,
  askQuestion,
  compareQuestion,
  deleteDocument,
  fetchDocuments,
  fetchHistory,
  resetIndex,
  uploadDocuments,
} from "./lib/api";

const sampleQuestion = "What AI engineering skills are required?";

function App() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [question, setQuestion] = useState(sampleQuestion);
  const [topK, setTopK] = useState(5);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [comparison, setComparison] = useState<QueryComparisonResponse | null>(null);
  const [history, setHistory] = useState<QueryHistoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [comparing, setComparing] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const totalChunks = useMemo(
    () => documents.reduce((sum, document) => sum + document.chunk_count, 0),
    [documents],
  );

  useEffect(() => {
    refreshDocuments();
    refreshHistory();
  }, []);

  async function refreshDocuments() {
    try {
      setDocuments(await fetchDocuments());
    } catch (err) {
      setError(errorMessage(err));
    }
  }

  async function refreshHistory() {
    try {
      setHistory(await fetchHistory());
    } catch (err) {
      setError(errorMessage(err));
    }
  }

  async function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const { files } = event.target;
    if (!files?.length) return;
    setUploading(true);
    setError(null);
    try {
      await uploadDocuments(files);
      await refreshDocuments();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  }

  async function handleAsk(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    setComparison(null);
    try {
      const response = await askQuestion(question.trim(), topK);
      setResult(response);
      await refreshHistory();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  }

  async function handleCompare() {
    if (!question.trim()) return;
    setComparing(true);
    setError(null);
    setResult(null);
    try {
      const response = await compareQuestion(question.trim(), topK);
      setComparison(response);
      await refreshHistory();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setComparing(false);
    }
  }

  function handleLoadHistory(item: QueryHistoryItem) {
    setQuestion(item.question);
    setComparison(null);
    setResult(item.response);
  }

  async function handleDelete(documentId: string) {
    setError(null);
    try {
      await deleteDocument(documentId);
      await refreshDocuments();
      setResult(null);
      setComparison(null);
    } catch (err) {
      setError(errorMessage(err));
    }
  }

  async function handleReset() {
    setError(null);
    try {
      await resetIndex();
      await refreshDocuments();
      setResult(null);
      setComparison(null);
    } catch (err) {
      setError(errorMessage(err));
    }
  }

  return (
    <main className="app-shell">
      <section className="topbar">
        <div>
          <p className="eyebrow">AI quality workspace</p>
          <h1>Agentic RAG Evaluator</h1>
        </div>
        <div className="status-pill">
          <Activity size={18} />
          <span>{documents.length} docs</span>
          <span>{totalChunks} chunks</span>
        </div>
      </section>

      {error && (
        <div className="error-banner">
          <AlertTriangle size={18} />
          <span>{error}</span>
        </div>
      )}

      <section className="workspace-grid">
        <aside className="panel sidebar-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Knowledge base</p>
              <h2>Documents</h2>
            </div>
            <button className="icon-button danger" onClick={handleReset} title="Reset index">
              <Trash2 size={18} />
            </button>
          </div>

          <label className="upload-zone">
            <input
              type="file"
              multiple
              accept=".pdf,.txt,.md,.markdown"
              onChange={handleUpload}
            />
            {uploading ? <Loader2 className="spin" size={24} /> : <UploadCloud size={24} />}
            <span>Upload PDF, TXT, or Markdown</span>
          </label>

          <div className="document-list">
            {documents.length === 0 && (
              <div className="empty-state">
                <Database size={28} />
                <span>No indexed documents</span>
              </div>
            )}
            {documents.map((document) => (
              <article className="document-row" key={document.document_id}>
                <FileText size={18} />
                <div>
                  <strong>{document.name}</strong>
                  <span>
                    {document.file_type} / {document.chunk_count} chunks
                  </span>
                </div>
                <button
                  className="icon-button"
                  onClick={() => handleDelete(document.document_id)}
                  title="Delete document"
                >
                  <Trash2 size={16} />
                </button>
              </article>
            ))}
          </div>

          <HistoryPanel history={history} onLoad={handleLoadHistory} />
        </aside>

        <section className="main-panel">
          <section className="panel query-panel">
            <form onSubmit={handleAsk}>
              <div className="question-row">
                <Search size={20} />
                <input
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="Ask a question about your documents"
                />
                <div className="query-actions">
                  <button type="submit" disabled={loading || comparing}>
                    {loading ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
                    <span>Ask</span>
                  </button>
                  <button
                    className="secondary-action"
                    type="button"
                    disabled={loading || comparing}
                    onClick={handleCompare}
                  >
                    {comparing ? (
                      <Loader2 className="spin" size={18} />
                    ) : (
                      <GitCompareArrows size={18} />
                    )}
                    <span>Compare</span>
                  </button>
                </div>
              </div>
              <div className="control-row">
                <label htmlFor="top-k">Sources</label>
                <input
                  id="top-k"
                  type="range"
                  min="1"
                  max="12"
                  value={topK}
                  onChange={(event) => setTopK(Number(event.target.value))}
                />
                <output>{topK}</output>
              </div>
            </form>
          </section>

          {comparison ? (
            <ComparisonView comparison={comparison} />
          ) : result ? (
            <ResultView result={result} />
          ) : (
            <section className="panel empty-result">
              <Brain size={34} />
              <span>Ask a question to run the agent pipeline</span>
            </section>
          )}
        </section>
      </section>
    </main>
  );
}

function HistoryPanel({
  history,
  onLoad,
}: {
  history: QueryHistoryItem[];
  onLoad: (item: QueryHistoryItem) => void;
}) {
  return (
    <section className="history-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Observability</p>
          <h2>Recent Runs</h2>
        </div>
        <History size={18} />
      </div>

      <div className="history-list">
        {history.length === 0 && (
          <div className="empty-history">
            <span>No saved runs yet</span>
          </div>
        )}
        {history.map((item) => (
          <button
            className="history-row"
            key={item.run_id}
            type="button"
            onClick={() => onLoad(item)}
          >
            <span className={`history-risk risk-${item.hallucination_risk}`}>
              {item.hallucination_risk}
            </span>
            <strong>{item.question}</strong>
            <span>
              {formatEngine(item.pipeline_engine)} / {item.latency_ms} ms /{" "}
              {item.source_count} sources
            </span>
          </button>
        ))}
      </div>
    </section>
  );
}

function ComparisonView({ comparison }: { comparison: QueryComparisonResponse }) {
  return (
    <section className="comparison-grid">
      {comparison.runs.map((run) => (
        <ComparisonCard run={run} key={`${run.pipeline_engine}-${run.label}`} />
      ))}
    </section>
  );
}

function ComparisonCard({ run }: { run: QueryComparisonRun }) {
  const { response } = run;
  const { evaluation } = response;
  const riskClass = `risk-${evaluation.hallucination_risk}`;

  return (
    <article className="panel comparison-card">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">{formatEngine(run.pipeline_engine)}</p>
          <h2>{run.label}</h2>
        </div>
        <span className="latency">{response.latency_ms} ms</span>
      </div>

      <div className="comparison-meta">
        <span>{formatGenerationMode(run.generation_mode)}</span>
        <span>{run.model}</span>
        <span>{run.top_k} sources</span>
      </div>

      <pre className="comparison-answer">{response.answer || "No answer generated."}</pre>

      <div className="comparison-quality">
        <div className="panel-heading compact">
          <h3>Evaluation</h3>
          <span className={`risk-pill ${riskClass}`}>
            <ShieldCheck size={16} />
            {evaluation.hallucination_risk}
          </span>
        </div>
        <ScoreBar label="Relevance" value={evaluation.relevance_score} tone="teal" />
        <ScoreBar label="Groundedness" value={evaluation.groundedness_score} tone="violet" />
        <ScoreBar label="Completeness" value={evaluation.completeness_score} tone="amber" />
      </div>

      <div className="comparison-sources">
        <div className="comparison-section-title">
          <strong>{response.sources.length} evidence chunks</strong>
          <span>{response.agent_trace.length} trace events</span>
        </div>
        {response.sources.slice(0, 2).map((source) => (
          <article className="comparison-source" key={source.chunk_id}>
            <strong>{source.document_name}</strong>
            <span>
              score {Math.round(source.score * 100)}%
              {source.page ? ` / page ${source.page}` : ""}
            </span>
            <p>{source.text}</p>
          </article>
        ))}
      </div>
    </article>
  );
}

function ResultView({ result }: { result: QueryResponse }) {
  return (
    <section className="result-grid">
      <article className="panel answer-panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Final answer</p>
            <h2>Response</h2>
          </div>
          <span className="latency">{result.latency_ms} ms</span>
        </div>
        <pre>{result.answer}</pre>
      </article>

      <EvaluationPanel evaluation={result.evaluation} />

      <article className="panel sources-panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Evidence</p>
            <h2>Sources</h2>
          </div>
          <span className="count-pill">{result.sources.length}</span>
        </div>
        <div className="sources-list">
          {result.sources.map((source) => (
            <article className="source-item" key={source.chunk_id}>
              <div>
                <strong>{source.document_name}</strong>
                <span>
                  score {Math.round(source.score * 100)}%
                  {source.page ? ` / page ${source.page}` : ""}
                </span>
              </div>
              <p>{source.text}</p>
            </article>
          ))}
        </div>
      </article>

      <TracePanel trace={result.agent_trace} />
    </section>
  );
}

function EvaluationPanel({ evaluation }: { evaluation: EvaluationResult }) {
  const riskClass = `risk-${evaluation.hallucination_risk}`;
  return (
    <article className="panel evaluation-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Quality checks</p>
          <h2>Evaluation</h2>
        </div>
        <span className={`risk-pill ${riskClass}`}>
          <ShieldCheck size={16} />
          {evaluation.hallucination_risk}
        </span>
      </div>
      <ScoreBar label="Relevance" value={evaluation.relevance_score} tone="teal" />
      <ScoreBar label="Groundedness" value={evaluation.groundedness_score} tone="violet" />
      <ScoreBar label="Completeness" value={evaluation.completeness_score} tone="amber" />
      <div className="critic-note">{evaluation.critic_notes}</div>
      {evaluation.unsupported_claims.length > 0 && (
        <div className="claim-list">
          {evaluation.unsupported_claims.slice(0, 3).map((claim) => (
            <p key={claim}>{claim}</p>
          ))}
        </div>
      )}
    </article>
  );
}

function ScoreBar({
  label,
  value,
  tone,
}: {
  label: string;
  value: number;
  tone: "teal" | "violet" | "amber";
}) {
  return (
    <div className="score-row">
      <div>
        <span>{label}</span>
        <strong>{value}%</strong>
      </div>
      <div className="score-track">
        <div className={`score-fill ${tone}`} style={{ width: `${value}%` }} />
      </div>
    </div>
  );
}

function TracePanel({ trace }: { trace: AgentTraceEvent[] }) {
  return (
    <article className="panel trace-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Execution</p>
          <h2>Agent Trace</h2>
        </div>
      </div>
      <div className="trace-list">
        {trace.map((event) => (
          <article className="trace-item" key={`${event.agent}-${event.action}`}>
            <CheckCircle2 size={17} className={event.status} />
            <div>
              <strong>{event.agent}</strong>
              <span>{event.action}</span>
              <p>{event.detail}</p>
            </div>
            <time>{event.duration_ms} ms</time>
          </article>
        ))}
      </div>
    </article>
  );
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : "Unexpected error";
}

function formatEngine(engine: QueryComparisonRun["pipeline_engine"]) {
  return engine === "langgraph" ? "LangGraph engine" : "Linear engine";
}

function formatGenerationMode(mode: QueryComparisonRun["generation_mode"]) {
  return mode === "openai" ? "OpenAI" : "Local fallback";
}

export default App;
