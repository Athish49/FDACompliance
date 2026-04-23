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
 * Backend: "critical" | "high" | "medium" | "low"
 * Frontend: "critical" | "warning" | "info" | "pass"
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

/**
 * Poll the backend job status until completed or failed (max 120s, 2s intervals).
 */
async function pollJob(
  base: string,
  jobId: string
): Promise<Record<string, unknown> | null> {
  const maxAttempts = 60; // 60 × 2s = 120s
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise((r) => setTimeout(r, 2000));
    const res = await fetch(`${base}/api/jobs/${jobId}`);
    if (!res.ok) return null;
    const body = (await res.json()) as {
      status: string;
      result?: Record<string, unknown>;
      error?: string;
    };
    if (body.status === "completed" && body.result) {
      return body.result;
    }
    if (body.status === "failed") {
      return null;
    }
    // "queued" or "running" → keep polling
  }
  return null; // timeout
}

export async function POST(req: NextRequest) {
  const base = backendBase();
  const form = await req.formData();

  if (!base) {
    return NextResponse.json(previewResponse());
  }

  try {
    // Step 1: Submit document → get job_id
    const submitRes = await fetch(`${base}/api/analyze-document`, {
      method: "POST",
      body: form,
    });

    if (!submitRes.ok) {
      return NextResponse.json(previewResponse());
    }

    const submitBody = (await submitRes.json()) as {
      job_id?: string;
      // legacy: backend might return AnalysisResponse directly
      findings?: unknown;
    };

    // Step 2: If backend returned a job_id, poll for completion
    if (submitBody.job_id) {
      const report = await pollJob(base, submitBody.job_id);
      if (!report) {
        return NextResponse.json(previewResponse());
      }
      return NextResponse.json(transformReport(report));
    }

    // Fallback: backend returned a complete response (legacy or direct mode)
    if (submitBody.findings) {
      return NextResponse.json(submitBody);
    }

    return NextResponse.json(previewResponse());
  } catch {
    return NextResponse.json(previewResponse());
  }
}
