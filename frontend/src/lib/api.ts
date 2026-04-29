import type { AnalysisResponse, QueryResponse } from "@/types";

async function parseJson<T>(res: Response): Promise<T> {
  const text = await res.text();
  if (!res.ok) {
    throw new Error(text || res.statusText || `Request failed (${res.status})`);
  }
  return text ? (JSON.parse(text) as T) : ({} as T);
}

/**
 * Compliance Q&A — proxied to the FastAPI backend via `POST /api/query` (see route handler).
 */
export async function queryCompliance(question: string): Promise<QueryResponse> {
  const res = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  return parseJson<QueryResponse>(res);
}

/**
 * Document analysis — proxied via `POST /api/analyze-document` (forwards to backend when configured).
 * Throws with a human-readable message on timeout, job failure, or network error.
 */
export async function analyzeDocument(file: File): Promise<AnalysisResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/analyze-document", {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const text = await res.text();
    let message = `Analysis failed (${res.status})`;
    try {
      const body = JSON.parse(text) as { message?: string };
      if (body.message) message = body.message;
    } catch {
      // raw text is the message
      if (text) message = text;
    }
    throw new Error(message);
  }
  const text = await res.text();
  return text ? (JSON.parse(text) as AnalysisResponse) : ({} as AnalysisResponse);
}

export type IndexStatus = "unknown" | "no_run" | "running" | "ready" | "error";

/**
 * Fetch the CFR ingestion pipeline status from the backend.
 * Returns a simplified status label for display in the UI.
 */
export async function getIngestStatus(): Promise<IndexStatus> {
  try {
    const res = await fetch("/api/ingest-status", { cache: "no-store" });
    if (!res.ok) return "unknown";
    const body = (await res.json()) as { status?: string; success?: boolean };
    if (body.status === "running") return "running";
    if (body.status === "no_run") return "no_run";
    if (body.status === "unknown") return "unknown";
    // Full PipelineResult — check success flag
    if (typeof body.success === "boolean") return body.success ? "ready" : "error";
    return "unknown";
  } catch {
    return "unknown";
  }
}
