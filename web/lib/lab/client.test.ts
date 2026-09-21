import { afterEach, describe, expect, it, vi } from 'vitest';
import backend from '../../tests/fixtures/backend.json';
import { preflight } from '../../tests/fixtures/preflight';
import { getPreflight, listScenarios, startRun, getRun, pollRun } from './client';

afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });
const json = (data: unknown, status = 200) => new Response(JSON.stringify(data), { status });
const runAt = (status: string) => ({ ...backend.run, status });

describe('lab requests', () => {
  it('reads only the same-origin preflight and scenario paths', async () => {
    const requests: string[] = [];
    vi.stubGlobal('fetch', async (url: string) => { requests.push(url); return json(url.endsWith('preflight') ? preflight : backend.scenarios); });
    expect((await getPreflight()).runnable).toBe(true);
    expect((await listScenarios()).scenario_count).toBe(14);
    expect(requests).toEqual(['/api/lab/preflight', '/api/lab/scenarios']);
  });
  it('submits the exact idempotent body once without retrying a lost response', async () => {
    const sent: unknown[] = [];
    vi.stubGlobal('fetch', async (_url: string, init: RequestInit) => { sent.push(JSON.parse(init.body as string)); throw new Error('Bearer private-provider-error'); });
    const request = { request_id: backend.handle.request_id, scenario_ids: ['fixture'] };
    await expect(startRun(request)).rejects.toThrow('Connection unavailable');
    expect(sent).toEqual([request]);
  });
  it('never displays upstream errors or schema payloads', async () => {
    vi.stubGlobal('fetch', async () => json({ detail: 'Bearer private-error' }, 500));
    await expect(getRun(backend.handle.run_id)).rejects.toThrow('Lab service unavailable');
    vi.stubGlobal('fetch', async () => json({ access_token: 'private' }));
    await expect(getPreflight()).rejects.toThrow('Lab response could not be verified');
  });
  it('aborts reads at their timeout and on external cancellation', async () => {
    vi.useFakeTimers();
    vi.stubGlobal('fetch', (_url: string, init: RequestInit) => new Promise((_resolve, reject) => init.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')))));
    const timeout = getPreflight();
    const checked = expect(timeout).rejects.toThrow('Request timed out');
    await vi.advanceTimersByTimeAsync(10000);
    await checked;
    const controller = new AbortController();
    const aborted = getRun(backend.handle.run_id, controller.signal);
    const cancellation = expect(aborted).rejects.toMatchObject({ name: 'AbortError' });
    controller.abort();
    await cancellation;
  });
});

describe('read-only polling', () => {
  it.each(['complete', 'partial_failure', 'blocked'])('emits every stage and stops at %s with capped backoff', async (terminal) => {
    vi.useFakeTimers();
    const stages = ['queued', 'resetting', 'baseline_running', 'enhanced_running', 'grading', terminal];
    const times: number[] = [];
    vi.stubGlobal('fetch', async (_url: string, init: RequestInit) => { expect(init.method).toBe('GET'); times.push(Date.now()); return json(runAt(stages[times.length - 1])); });
    const seen: string[] = [];
    const polling = pollRun(backend.handle.run_id, new AbortController().signal, (run) => seen.push(run.status));
    await vi.runAllTimersAsync();
    expect((await polling).status).toBe(terminal);
    expect(seen).toEqual(stages);
    expect(times.map((time, index) => index ? time - times[index - 1] : 0)).toEqual([0, 250, 500, 1000, 2000, 2000]);
  });
  it('retries transport reads, preserves the successful side, and never posts', async () => {
    vi.useFakeTimers();
    const methods: string[] = [];
    vi.stubGlobal('fetch', async (_url: string, init: RequestInit) => { methods.push(init.method!); if (methods.length === 1) throw new TypeError('network'); return json(backend.run); });
    const pending = pollRun(backend.handle.run_id, new AbortController().signal);
    await vi.runAllTimersAsync();
    const result = await pending;
    expect(methods).toEqual(['GET', 'GET']);
    expect(result.pairs[0].outcomes[0].results[0].final_response.value).toBe('Fabricated fixture response.');
    expect(result.pairs[0].outcomes[1].reason).toBe('Enhanced side exceeded its time limit.');
  });
  it('stops instead of retrying unverified or rejected responses', async () => {
    const reads: number[] = [];
    vi.stubGlobal('fetch', async () => { reads.push(1); return json({ invalid: true }); });
    await expect(pollRun(backend.handle.run_id, new AbortController().signal)).rejects.toThrow('Lab response could not be verified');
    expect(reads).toHaveLength(1);
  });
  it('cancels backoff on navigation without further reads', async () => {
    vi.useFakeTimers();
    const reads: number[] = [];
    vi.stubGlobal('fetch', async () => { reads.push(1); return json(runAt('queued')); });
    const controller = new AbortController();
    const pending = pollRun(backend.handle.run_id, controller.signal);
    const checked = expect(pending).rejects.toMatchObject({ name: 'AbortError' });
    await vi.advanceTimersByTimeAsync(0);
    controller.abort();
    await checked;
    await vi.runAllTimersAsync();
    expect(reads).toHaveLength(1);
  });
});
