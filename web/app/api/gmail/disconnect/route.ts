export const runtime = 'nodejs';

export async function POST(req: Request) {
  let body: any = {};
  try {
    body = await req.json();
  } catch {}

  const userId = body?.userId || '';
  const connectionId = body?.connectionId || '';
  const connectionRequestId = body?.connectionRequestId || '';

  const targets = Array.from(new Set([
    process.env.PY_SERVER_URL || 'http://localhost:8001',
    'http://localhost:8001',
    'http://localhost:8002'
  ]));

  const payload: any = {};
  if (userId) payload.user_id = userId;
  if (connectionId) payload.connection_id = connectionId;
  if (connectionRequestId) payload.connection_request_id = connectionRequestId;

  let primaryData: any = null;
  let primaryStatus = 200;

  for (const base of targets) {
    const url = `${base.replace(/\/$/, '')}/api/v1/gmail/disconnect`;
    try {
      const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await resp.json().catch(() => ({}));
      if (!primaryData) {
        primaryData = data;
        primaryStatus = resp.status;
      }
    } catch {}
  }

  if (primaryData) {
    return new Response(JSON.stringify(primaryData), {
      status: primaryStatus,
      headers: { 'Content-Type': 'application/json; charset=utf-8' },
    });
  }

  return new Response(
    JSON.stringify({ ok: false, error: 'Upstream error' }),
    { status: 502, headers: { 'Content-Type': 'application/json; charset=utf-8' } }
  );
}
