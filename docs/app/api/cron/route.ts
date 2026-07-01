import { NextResponse } from "next/server";

export async function GET(request: Request) {
  const authHeader = request.headers.get("authorization");
  const cronSecret = process.env.CRON_SECRET;
  if (!cronSecret || authHeader !== `Bearer ${cronSecret}`) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const hookUrl = process.env.VERCEL_DEPLOY_HOOK;
  if (!hookUrl) {
    return NextResponse.json(
      { error: "VERCEL_DEPLOY_HOOK not configured" },
      { status: 500 }
    );
  }

  const res = await fetch(hookUrl, { method: "POST" });
  if (!res.ok) {
    return NextResponse.json(
      { error: "Failed to trigger deploy" },
      { status: 502 }
    );
  }

  return NextResponse.json({ ok: true, triggered: new Date().toISOString() });
}
