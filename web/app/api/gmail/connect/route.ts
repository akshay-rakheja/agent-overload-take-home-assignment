export const runtime = 'nodejs';

export async function POST(req: Request) {
  let body: any = {};
  try {
    body = await req.json();
  } catch {}
  let userId = body?.userId || '';
  const authConfigId = body?.authConfigId || '';

  if (userId.startsWith('web-')) {
    userId = '';
  }

  const targets = Array.from(new Set([
    process.env.PY_SERVER_URL || 'http://localhost:8001',
    'http://localhost:8001',
    'http://localhost:8002'
  ]));

  let primaryData: any = null;
  let primaryStatus = 200;

  for (const base of targets) {
    const url = `${base.replace(/\/$/, '')}/api/v1/gmail/connect`;
    try {
      const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ user_id: userId, auth_config_id: authConfigId }),
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
