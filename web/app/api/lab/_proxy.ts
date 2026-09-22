import { z } from 'zod';
import { GmailLinkSchema, GmailStatusSchema, LabPreflightSchema, PairedRunResultSchema, RunHandleSchema, ScenarioListSchema, StartRunRequestSchema } from '@/lib/lab/schema';

const origin = 'http://127.0.0.1:8002';
const allowedRequestOrigins = new Set(['http://127.0.0.1:3000', 'http://localhost:3000', 'http://n']);
const allowedClientOrigins = new Set(['http://127.0.0.1:3000', 'http://localhost:3000']);
const endpoints: Record<string, z.ZodTypeAny> = {
  'GET /lab/preflight': LabPreflightSchema,
  'GET /lab/scenarios': ScenarioListSchema,
  'POST /lab/runs': RunHandleSchema,
  'GET /lab/gmail/status': GmailStatusSchema,
  'POST /lab/gmail/link': GmailLinkSchema,
};
const reply = (data: unknown, status = 200) => Response.json(data, { status, headers: { 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' } });
class ProxyError extends Error {
  constructor(readonly status: number, message: string, readonly code?: 'INVALID_UPSTREAM_RESPONSE') { super(message); }
}

async function upstream(path: string, schema: z.ZodTypeAny, signal: AbortSignal, method = 'GET', body?: unknown) {
  const response = await fetch(`${origin}/api/v1${path}`, {
    method, redirect: 'error', cache: 'no-store', signal,
    headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  if (!response.ok) {
    const status = [403, 404, 409, 422, 429].includes(response.status) ? response.status : 502;
    throw new ProxyError(status, status === 404 ? 'Lab endpoint unavailable' : status === 409 ? 'Run blocked by the lab service' : 'Lab service unavailable');
  }
  let payload: unknown;
  try { payload = await response.json(); } catch (error) {
    if (error instanceof SyntaxError) throw new ProxyError(502, 'Lab response could not be verified', 'INVALID_UPSTREAM_RESPONSE');
    // Preserve body-stream failures for the sanitized transport/timeout boundary.
    throw error;
  }
  const parsed = schema.safeParse(payload);
  if (!parsed.success) throw new ProxyError(502, 'Lab response could not be verified', 'INVALID_UPSTREAM_RESPONSE');
  return { data: parsed.data, status: response.status };
}

export async function proxyLab(request: Request, path: string, method = 'GET'): Promise<Response> {
  const runId = path.startsWith('/lab/runs/') ? path.slice('/lab/runs/'.length) : null;
  const schema = method === 'GET' && runId && z.string().uuid().safeParse(runId).success ? PairedRunResultSchema : endpoints[`${method} ${path}`];
  if (!schema) return reply({ error: 'Invalid lab path' }, 400);
  const reqOrigin = new URL(request.url).origin;
  const clientOrigin = request.headers.get('origin');
  if (!allowedRequestOrigins.has(reqOrigin) || (method === 'POST' && (!clientOrigin || !allowedClientOrigins.has(clientOrigin)))) {
    return reply({ error: 'Local origin required' }, 403);
  }
  let body: unknown;
  if (method === 'POST') {
    try {
      const raw = await request.text();
      if (new TextEncoder().encode(raw).length > 16384) return reply({ error: 'Invalid run request' }, 400);
      const validator = path === '/lab/runs' ? StartRunRequestSchema : z.object({}).strict();
      const parsed = validator.safeParse(JSON.parse(raw.trim() || '{}'));
      if (!parsed.success) return reply({ error: 'Invalid run request' }, 400);
      body = parsed.data;
    } catch { return reply({ error: 'Invalid run request' }, 400); }
  }
  if (request.signal.aborted) return reply({ error: 'Request cancelled' }, 499);
  const controller = new AbortController();
  const cancel = () => controller.abort();
  request.signal.addEventListener('abort', cancel, { once: true });
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, 10000);
  try {
    if (method === 'POST' && path === '/lab/runs') {
      const check = await upstream('/lab/preflight', LabPreflightSchema, controller.signal);
      if (!check.data.runnable) return reply({ error: 'Run blocked by the lab service' }, 409);
    }
    const result = await upstream(path, schema, controller.signal, method, body);
    return reply(result.data, result.status);
  } catch (error) {
    if (request.signal.aborted) return reply({ error: 'Request cancelled' }, 499);
    if (timedOut) return reply({ error: 'Request timed out', code: 'UPSTREAM_TIMEOUT' }, 504);
    if (error instanceof ProxyError) return reply({ error: error.message, ...(error.code ? { code: error.code } : {}) }, error.status);
    return reply({ error: 'Lab service unavailable', code: 'UPSTREAM_TRANSPORT' }, 502);
  } finally {
    clearTimeout(timer);
    request.signal.removeEventListener('abort', cancel);
  }
}
