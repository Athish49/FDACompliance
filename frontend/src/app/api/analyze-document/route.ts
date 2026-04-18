import { NextRequest, NextResponse } from "next/server";
import type { AnalysisResponse } from "@/types";

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

export async function POST(req: NextRequest) {
  const base = backendBase();
  const form = await req.formData();

  if (base) {
    try {
      const upstream = await fetch(`${base}/api/analyze-document`, {
        method: "POST",
        body: form,
      });
      if (upstream.ok) {
        const text = await upstream.text();
        return new NextResponse(text, {
          status: upstream.status,
          headers: { "Content-Type": "application/json" },
        });
      }
    } catch {
      // fall through to preview
    }
  }

  return NextResponse.json(previewResponse());
}
