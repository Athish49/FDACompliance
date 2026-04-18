import { NextRequest, NextResponse } from "next/server";

function backendBase(): string {
  const raw =
    process.env.BACKEND_URL?.trim() ||
    process.env.NEXT_PUBLIC_API_URL?.trim() ||
    "";
  return raw.replace(/\/+$/, "");
}

export async function POST(req: NextRequest) {
  const base = backendBase();
  if (!base) {
    return NextResponse.json(
      {
        error:
          "Backend URL not configured. Set BACKEND_URL or NEXT_PUBLIC_API_URL to your FastAPI server.",
      },
      { status: 503 },
    );
  }

  const body = await req.text();
  const upstream = await fetch(`${base}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
  });

  const text = await upstream.text();
  return new NextResponse(text, {
    status: upstream.status,
    headers: { "Content-Type": "application/json" },
  });
}
