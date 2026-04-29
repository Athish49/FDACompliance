import { NextRequest, NextResponse } from "next/server";
import type { AnalysisResponse, AnalysisFinding } from "@/types";

function backendBase(): string {
  const raw =
    process.env.BACKEND_URL?.trim() ||
    process.env.NEXT_PUBLIC_API_URL?.trim() ||
    "";
  return raw.replace(/\/+$/, "");
}

function previewResponse(): AnalysisResponse {
  return {
    findings: [
      {
        id: "preview-1",
        category: "Configuration",
        severity: "info",
        title: "Analysis API not available",
        description:
          "The document analyzer requires a backend that implements POST /api/analyze-document. This preview response is shown when the upstream service is missing or returns an error.",
        regulation: "21 CFR (configure backend for full review)",
        recommendation:
          "Deploy the FDA Compliance AI API and set BACKEND_URL or NEXT_PUBLIC_API_URL in Vercel.",
      },
    ],
    summary:
      "Preview mode: your file was received, but full automated analysis is not connected.",
    overall_status: "needs_review",
    analyzed_at: new Date().toISOString(),
  };
}

/**
 * Map a backend severity string to the frontend severity enum.
 *
 * Backend values (from document_analysis/graph.py):  "critical" | "high" | "medium" | "low"
 * Frontend values (see types/index.ts AnalysisFinding): "critical" | "warning" | "info" | "pass"
 *
 * If the backend adds a new severity value, a warning is logged server-side
 * and the value falls through to "info" so the UI doesn't break silently.
 */
function mapSeverity(
  sev: string
): "critical" | "warning" | "info" | "pass" {
  switch (sev) {
    case "critical":
      return "critical";
    case "high":
      return "warning";
    case "medium":
      return "info";
    case "low":
      return "pass";
    default:
      console.warn(
        `[analyze-document] Unknown backend severity value "${sev}" — defaulting to "info". ` +
        `Update mapSeverity() in route.ts if the backend severity enum has changed.`
      );
      return "info";
  }
}

/**
 * Map backend overall_status to frontend overall_status.
 * Backend adds "partially_compliant" which the frontend calls "needs_review".
 */
function mapOverallStatus(
  status: string
): "compliant" | "non_compliant" | "needs_review" {
  if (status === "compliant") return "compliant";
  if (status === "non_compliant") return "non_compliant";
  return "needs_review"; // partially_compliant, error, or unknown
}

/**
 * Human-readable title derived from violation_type + claim_type.
 */
function violationTitle(violation: Record<string, unknown>): string {
  const vtype = violation.violation_type as string | undefined;
  const ctype = violation.claim_type as string | undefined;

  const typeLabel =
    vtype === "missing"
      ? "Missing"
      : vtype === "incorrect"
      ? "Incorrect"
      : vtype === "prohibited"
      ? "Prohibited"
      : vtype === "insufficient_disclosure"
      ? "Insufficient Disclosure"
      : "Potential Violation";

  if (ctype === "missing_required_element") {
    const claimText = violation.claim_text as string | undefined;
    const elementName = claimText?.replace("[Missing element] ", "") ?? "Required Element";
    return `${elementName}`;
  }

  const categoryLabel = (ctype ?? "Claim")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());

  return `${typeLabel} ${categoryLabel}`;
}

/**
 * Transform backend ViolationReport into frontend AnalysisResponse.
 */
function transformReport(report: Record<string, unknown>): AnalysisResponse {
  const violations = (report.violations as Record<string, unknown>[]) ?? [];
  const severitySummary = (report.severity_summary as Record<string, number>) ?? {};
  const totalViolations = (report.total_violations as number) ?? violations.length;
  const docName = (report.document_name as string) ?? "document";

  const findings: AnalysisFinding[] = violations.map((v, i) => ({
    id: `v-${i + 1}`,
    category:
      v.claim_type === "missing_required_element"
        ? "Missing Required Element"
        : ((v.claim_type as string) ?? "Label Claim")
            .replace(/_/g, " ")
            .replace(/\b\w/g, (c) => c.toUpperCase()),
    severity: mapSeverity(v.severity as string),
    title: violationTitle(v),
    description: (v.explanation as string) ?? "",
    regulation: (v.cfr_citation as string) ?? "21 CFR Part 101",
    recommendation:
      v.violation_type === "missing"
        ? "Add the missing required element to the label."
        : v.violation_type === "prohibited"
        ? "Remove or replace this prohibited claim."
        : "Review and correct the claim to meet CFR requirements.",
  }));

  // Build a human-readable summary
  const parts: string[] = [];
  if (severitySummary.critical > 0)
    parts.push(`${severitySummary.critical} critical`);
  if (severitySummary.high > 0)
    parts.push(`${severitySummary.high} high-priority`);
  if (severitySummary.medium > 0)
    parts.push(`${severitySummary.medium} medium`);
  if (severitySummary.low > 0) parts.push(`${severitySummary.low} low`);

  const summary =
    totalViolations === 0
      ? `No violations detected in ${docName}. The document appears to meet FDA labeling requirements.`
      : `Found ${totalViolations} potential issue${totalViolations > 1 ? "s" : ""} in ${docName}: ${parts.join(", ")}.`;

  return {
    findings,
    summary,
    overall_status: mapOverallStatus(report.overall_status as string),
    analyzed_at: (report.analyzed_at as string) ?? new Date().toISOString(),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Poll result — discriminated union so the POST handler can respond precisely
// ─────────────────────────────────────────────────────────────────────────────

type PollResult =
  | { ok: true; report: Record<string, unknown> }
  | { ok: false; reason: "timeout" | "job_failed" | "network_error"; detail?: string };

/**
 * Poll the backend job status until completed or failed (max 120s, 2s intervals).
 * Returns a discriminated PollResult instead of null so the caller can give
 * the user a precise error message rather than silently falling back to preview data.
 */
async function pollJob(base: string, jobId: string): Promise<PollResult> {
  const maxAttempts = 60; // 60 × 2s = 120s
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise((r) => setTimeout(r, 2000));
    let res: Response;
    try {
      res = await fetch(`${base}/api/jobs/${jobId}`);
    } catch (err) {
      return {
        ok: false,
        reason: "network_error",
        detail: err instanceof Error ? err.message : String(err),
      };
    }
    if (!res.ok) {
      return { ok: false, reason: "network_error", detail: `HTTP ${res.status}` };
    }
    const body = (await res.json()) as {
      status: string;
      result?: Record<string, unknown>;
      error?: string;
    };
    if (body.status === "completed" && body.result) {
      return { ok: true, report: body.result };
    }
    if (body.status === "failed") {
      return { ok: false, reason: "job_failed", detail: body.error };
    }
    // "queued" or "running" → keep polling
  }
  return { ok: false, reason: "timeout" };
}

export async function POST(req: NextRequest) {
  const base = backendBase();
  const form = await req.formData();

  // Backend not configured — show preview so the UI isn't broken in local dev
  if (!base) {
    return NextResponse.json(previewResponse());
  }

  let submitRes: Response;
  try {
    submitRes = await fetch(`${base}/api/analyze-document`, {
      method: "POST",
      body: form,
    });
  } catch {
    // Backend unreachable — show preview
    return NextResponse.json(previewResponse());
  }

  if (!submitRes.ok) {
    // Backend running but returned an error — show preview
    return NextResponse.json(previewResponse());
  }

  const submitBody = (await submitRes.json()) as {
    job_id?: string;
    findings?: unknown; // legacy: backend might return AnalysisResponse directly
  };

  // Async job path: poll until done
  if (submitBody.job_id) {
    const poll = await pollJob(base, submitBody.job_id);

    if (!poll.ok) {
      if (poll.reason === "timeout") {
        return NextResponse.json(
          {
            error: "timeout",
            message:
              "Analysis took over 2 minutes. The document may be too large — try a smaller file or a plain-text extract.",
          },
          { status: 504 }
        );
      }
      if (poll.reason === "job_failed") {
        return NextResponse.json(
          {
            error: "job_failed",
            message: poll.detail
              ? `Analysis failed: ${poll.detail}`
              : "Analysis failed on the server. Check the document format and try again.",
          },
          { status: 502 }
        );
      }
      // network_error
      return NextResponse.json(
        {
          error: "network_error",
          message:
            poll.detail
              ? `Lost connection to the analysis service: ${poll.detail}`
              : "Lost connection to the analysis service. Check the backend is running and try again.",
        },
        { status: 502 }
      );
    }

    return NextResponse.json(transformReport(poll.report));
  }

  // Legacy / direct-response path
  if (submitBody.findings) {
    return NextResponse.json(submitBody);
  }

  return NextResponse.json(previewResponse());
}
