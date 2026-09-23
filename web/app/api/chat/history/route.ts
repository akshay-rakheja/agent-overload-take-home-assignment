export const runtime = 'nodejs';

function resolveServerBase(system?: string | null): string {
  if (system === 'baseline') {
    return process.env.BASELINE_SERVER_URL || 'http://localhost:8001';
  }
  if (system === 'enhanced' || system === 'enhanced_deterministic' || system === 'enhanced_jev' || system === 'jev') {
    return process.env.ENHANCED_SERVER_URL || 'http://localhost:8002';
  }
  return process.env.PY_SERVER_URL || 'http://localhost:8001';
}

async function forward(method: 'GET' | 'DELETE', system?: string | null, includeSystemQuery: boolean = true) {
  const serverBase = resolveServerBase(system);
  const query = includeSystemQuery && system ? `?system=${encodeURIComponent(system)}` : '';
  const historyPath = `${serverBase.replace(/\/$/, '')}/api/v1/chat/history${query}`;

  try {
    const res = await fetch(historyPath, {
      method,
      headers: { Accept: 'application/json' },
      cache: 'no-store',
    });

    const bodyText = await res.text();
    const headers = new Headers({ 'Content-Type': 'application/json; charset=utf-8' });
    return new Response(bodyText || '{}', { status: res.status, headers });
  } catch (error: any) {
    const message = error?.message || 'Failed to reach Python server';
    return new Response(JSON.stringify({ error: message }), {
      status: 502,
      headers: { 'Content-Type': 'application/json; charset=utf-8' },
    });
  }
}

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const system = searchParams.get('system');
  return forward('GET', system);
}

export async function DELETE(req: Request) {
  const { searchParams } = new URL(req.url);
  const system = searchParams.get('system');
  if (system) {
    return forward('DELETE', system);
  }
  const [resBaseline, resDet, resJev, resAll] = await Promise.all([
    forward('DELETE', 'baseline'),
    forward('DELETE', 'enhanced_deterministic'),
    forward('DELETE', 'enhanced_jev'),
    forward('DELETE', 'enhanced', false),
  ]);
  return resJev.ok ? resJev : resDet.ok ? resDet : resBaseline;
}
