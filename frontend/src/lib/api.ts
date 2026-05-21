import type { AnalysisResponse, QueryResponse, SSEEvent } from "@/types";

async function parseJson<T>(res: Response): Promise<T> {
  const text = await res.text();
  if (!res.ok) {
    throw new Error(text || res.statusText || `Request failed (${res.status})`);
  }
  return text ? (JSON.parse(text) as T) : ({} as T);
}

/**
 * Compliance Q&A — SSE streaming version.
 *
 * Calls the backend pipeline and fires `onEvent` as each agent node completes,
 * enabling live progress feedback. Resolves with the final QueryResponse when
 * the stream ends, or rejects on network/backend error.
 */
export async function queryComplianceStream(
  question: string,
  onEvent: (event: SSEEvent) => void,
): Promise<QueryResponse> {
  const res = await fetch("/api/query/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Stream request failed (${res.status})`);
  }
  if (!res.body) {
    throw new Error("Response has no body — streaming not supported in this environment.");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResponse: QueryResponse | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Split on newlines; keep any incomplete trailing line in the buffer.
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const data = line.slice(6).trim();
      if (data === "[DONE]") break;

      let event: SSEEvent;
      try {
        event = JSON.parse(data) as SSEEvent;
      } catch {
        continue;
      }

      onEvent(event);

      // final_synthesizer and clarification_response carry the full answer payload.
      if (
        (event.event === "final_synthesizer" ||
          event.event === "clarification_response" ||
          // legacy node names kept for backward compatibility
          event.event === "conflict_detector" ||
          event.event === "insufficient_coverage") &&
        event.answer
      ) {
        finalResponse = event.answer;
      }
    }
  }

  if (!finalResponse) {
    throw new Error("Stream ended without a final answer from the pipeline.");
  }
  return finalResponse;
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
    if (typeof body.success === "boolean") return body.success ? "ready" : "error";
    return "unknown";
  } catch {
    return "unknown";
  }
}
