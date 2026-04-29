import { NextResponse } from "next/server";

function backendBase(): string {
  const raw =
    process.env.BACKEND_URL?.trim() ||
    process.env.NEXT_PUBLIC_API_URL?.trim() ||
    "";
  return raw.replace(/\/+$/, "");
}

export async function GET() {
  const base = backendBase();
  if (!base) return NextResponse.json({ status: "unknown" });

  try {
    const res = await fetch(`${base}/api/ingest/status`, {
      next: { revalidate: 0 },
    });
    if (!res.ok) return NextResponse.json({ status: "unknown" });
    return NextResponse.json(await res.json());
  } catch {
    return NextResponse.json({ status: "unknown" });
  }
}
