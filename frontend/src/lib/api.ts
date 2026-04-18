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
 */
export async function analyzeDocument(file: File): Promise<AnalysisResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/analyze-document", {
    method: "POST",
    body: form,
  });
  return parseJson<AnalysisResponse>(res);
}
