// @vitest-environment node
import { afterEach, expect, it, vi } from 'vitest';
import backend from '../../../tests/fixtures/backend.json';
import { preflight } from '../../../tests/fixtures/preflight';
import { proxyLab } from './_proxy';
import { GET as readRun } from './runs/[runId]/route';
import { POST as createRun } from './runs/route';
import { GET as readPreflight } from './preflight/route';
import { GET as readScenarios } from './scenarios/route';
import { GET as readGmail } from './gmail/status/route';
import { POST as linkGmail } from './gmail/link/route';
import { pollRun, startRun } from '../../../lib/lab/client';

afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });
const request = (path = 'runs', body?: unknown, origin = 'http://127.0.0.1:3000') => new Request(`http://127.0.0.1:3000/api/lab/${path}`, body === undefined ? {} : { method: 'POST', headers: { origin, 'content-type': 'application/json' }, body: JSON.stringify(body) });
const json = (data: unknown, status = 200) => new Response(JSON.stringify(data), { status });

it('uses the fixed enhanced origin and rejects URLs, traversal, query strings and invalid UUIDs', async () => {
  const targets: string[] = [];
  vi.stubGlobal('fetch', async (url: string, init: RequestInit) => { targets.push(url); expect(init.redirect).toBe('error'); return json(backend.scenarios); });
  expect((await readScenarios(request('scenarios'))).status).toBe(200);
  for (const path of ['http://evil.test', '/lab/../gmail/connect', '/lab/scenarios?url=http://evil.test', '/lab/runs/not-a-uuid', '//evil.test']) {
    expect((await proxyLab(request(), path)).status).toBe(400);
  }
  expect((await readRun(request(), { params: { runId: '../scenarios' } })).status).toBe(400);
  expect(targets).toEqual(['http://127.0.0.1:8002/api/v1/lab/scenarios']);
});

it('checks preflight before forwarding the exact idempotent run body', async () => {
  const writes: unknown[] = [];
  vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
    if (url.endsWith('preflight')) return json(preflight);
    writes.push(JSON.parse(init.body as string));
    expect(init.headers).not.toHaveProperty('authorization');
    return json(backend.handle, 202);
  });
  const body = { request_id: backend.handle.request_id, scenario_ids: ['fixture'] };
  expect((await createRun(request('runs', body))).status).toBe(202);
  expect(writes).toEqual([body]);
});

it('refuses cross-origin writes, unknown body fields and unsafe preflight without starting', async () => {
  const methods: string[] = [];
  vi.stubGlobal('fetch', async (_url: string, init: RequestInit) => { methods.push(init.method!); return json({ ...preflight, runnable: false, blockers: ['Gmail unsafe'] }); });
  const body = { request_id: backend.handle.request_id, scenario_ids: ['fixture'] };
  expect((await createRun(request('runs', body, 'https://evil.test'))).status).toBe(403);
  expect((await createRun(request('runs', { ...body, access_token: 'private' }))).status).toBe(400);
  expect((await createRun(request('runs', body))).status).toBe(409);
  expect(methods).toEqual(['GET']);
});

it('sanitizes missing prospective endpoints, provider errors, and invalid successful payloads', async () => {
  vi.stubGlobal('fetch', async () => json({ detail: 'Bearer private-token' }, 404));
  for (const response of [await readPreflight(request('preflight')), await readGmail(request('gmail/status')), await linkGmail(request('gmail/link', {}))]) {
    expect(response.status).toBe(404);
    expect(await response.text()).toBe('{"error":"Lab endpoint unavailable"}');
  }
  vi.stubGlobal('fetch', async () => json({ ...backend.scenarios, token: 'private' }));
  const invalid = await readScenarios(request('scenarios'));
  expect(invalid.status).toBe(502);
  expect(await invalid.json()).toEqual({ error: 'Lab response could not be verified', code: 'INVALID_UPSTREAM_RESPONSE' });
});

it('bounds upstream reads and passes navigation cancellation through', async () => {
  vi.useFakeTimers();
  vi.stubGlobal('fetch', (_url: string, init: RequestInit) => new Promise((_resolve, reject) => init.signal?.addEventListener('abort', () => reject(new Error('private')))));
  const pending = readPreflight(request('preflight'));
  await vi.advanceTimersByTimeAsync(10000);
  const timedOut = await pending;
  expect(timedOut.status).toBe(504);
  expect(await timedOut.json()).toEqual({ error: 'Request timed out', code: 'UPSTREAM_TIMEOUT' });
  const controller = new AbortController();
  const cancelled = readPreflight(new Request('http://127.0.0.1:3000/api/lab/preflight', { signal: controller.signal }));
  controller.abort();
  expect((await cancelled).status).toBe(499);
});

it('marks a backend connection failure with a stable sanitized transport code', async () => {
  vi.stubGlobal('fetch', async () => { throw new TypeError('Bearer private-upstream-error'); });
  const response = await readScenarios(request('scenarios'));
  expect(response.status).toBe(502);
  expect(await response.json()).toEqual({ error: 'Lab service unavailable', code: 'UPSTREAM_TRANSPORT' });
});

it('retries a backend connection failure through the actual proxy and retains terminal evidence', async () => {
  vi.useFakeTimers();
  const methods: string[] = [];
  vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
    if (url.startsWith('/api/lab/')) return readRun(new Request(`http://127.0.0.1:3000${url}`, { signal: init.signal }), { params: { runId: backend.handle.run_id } });
    methods.push(init.method!);
    if (methods.length === 1) throw new TypeError('upstream connection lost');
    return json(backend.run);
  });
  const pending = pollRun(backend.handle.run_id, new AbortController().signal).catch((error) => error);
  await vi.runAllTimersAsync();
  expect((await pending).status).toBe('partial_failure');
  expect(methods).toEqual(['GET', 'GET']);
});

it('caps backend transport retries through the proxy at three retries', async () => {
  vi.useFakeTimers();
  const methods: string[] = [];
  vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
    if (url.startsWith('/api/lab/')) return readRun(new Request(`http://127.0.0.1:3000${url}`, { signal: init.signal }), { params: { runId: backend.handle.run_id } });
    methods.push(init.method!);
    throw new TypeError('private backend error');
  });
  const pending = pollRun(backend.handle.run_id, new AbortController().signal).catch((error) => error);
  await vi.runAllTimersAsync();
  expect((await pending).message).toBe('Lab service unavailable');
  expect(methods).toEqual(['GET', 'GET', 'GET', 'GET']);
});

it('does not retry a schema rejection from the actual proxy', async () => {
  const reads: number[] = [];
  vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
    if (url.startsWith('/api/lab/')) return readRun(new Request(`http://127.0.0.1:3000${url}`, { signal: init.signal }), { params: { runId: backend.handle.run_id } });
    reads.push(1);
    return json({ private_payload: 'not a run' });
  });
  await expect(pollRun(backend.handle.run_id, new AbortController().signal)).rejects.toMatchObject({ message: 'Lab response could not be verified', retryable: false });
  expect(reads).toHaveLength(1);
});

it('never retries a start whose backend response is lost through the actual proxy', async () => {
  const writes: unknown[] = [];
  vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
    if (url === '/api/lab/runs') return createRun(request('runs', JSON.parse(init.body as string)));
    if (url.endsWith('preflight')) return json(preflight);
    writes.push(JSON.parse(init.body as string));
    throw new TypeError('private lost start response');
  });
  const body = { request_id: backend.handle.request_id, scenario_ids: ['fixture'] };
  await expect(startRun(body)).rejects.toMatchObject({ message: 'Lab service unavailable', retryable: false });
  expect(writes).toEqual([body]);
});
