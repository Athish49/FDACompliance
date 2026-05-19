import { NextRequest } from "next/server";

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
    return new Response(
      JSON.stringify({ error: "Backend URL not configured." }),
      { status: 503, headers: { "Content-Type": "application/json" } },
    );
  }

  const body = await req.text();

  let upstream: Response;
  try {
    upstream = await fetch(`${base}/api/query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });
  } catch (err) {
    return new Response(
      JSON.stringify({ error: `Backend unreachable: ${err}` }),
      { status: 502, headers: { "Content-Type": "application/json" } },
    );
  }

  if (!upstream.ok || !upstream.body) {
    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: { "Content-Type": "application/json" },
    });
  }

  // Forward the SSE stream directly — Next.js App Router passes ReadableStream through.
  return new Response(upstream.body, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "X-Accel-Buffering": "no",
    },
  });
}
