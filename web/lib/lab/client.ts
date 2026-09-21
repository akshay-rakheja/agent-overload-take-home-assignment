import { z } from 'zod';
import { LabPreflightSchema, LabProxyErrorSchema, ScenarioListSchema, StartRunRequestSchema, RunHandleSchema, PairedRunResultSchema, type StartRunRequest, type PairedRunResult } from './schema';

export class LabClientError extends Error {
  constructor(message: string, readonly retryable = false) { super(message); this.name = 'LabClientError'; }
}

export function errorMessage(error: unknown): string {
  return error instanceof LabClientError ? error.message : 'Lab request could not be completed';
}

const abortError = () => new DOMException('Request cancelled', 'AbortError');

async function request<T>(path: string, schema: z.ZodType<T>, signal?: AbortSignal, body?: StartRunRequest): Promise<T> {
  if (signal?.aborted) throw abortError();
  const controller = new AbortController();
  const cancel = () => controller.abort();
  signal?.addEventListener('abort', cancel, { once: true });
  let timedOut = false;
  const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, 10000);
  try {
    const response = await fetch(`/api/lab/${path}`, {
      method: body ? 'POST' : 'GET', cache: 'no-store', signal: controller.signal,
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      ...(body ? { body: JSON.stringify(body) } : {}),
    });
    if (!response.ok) {
      const proxyError = LabProxyErrorSchema.safeParse(await response.json().catch(() => null));
      if (proxyError.success) {
        const retryableRead = !body && (
          (response.status === 502 && proxyError.data.code === 'UPSTREAM_TRANSPORT') ||
          (response.status === 504 && proxyError.data.code === 'UPSTREAM_TIMEOUT')
        );
        throw new LabClientError(proxyError.data.error, retryableRead);
      }
      const message = response.status === 404 ? 'Lab endpoint unavailable' : response.status === 409 ? 'Run blocked by the lab service' : 'Lab service unavailable';
      throw new LabClientError(message);
    }
    let payload: unknown;
    try { payload = await response.json(); } catch { throw new LabClientError('Lab response could not be verified'); }
    const parsed = schema.safeParse(payload);
    if (!parsed.success) throw new LabClientError('Lab response could not be verified');
    return parsed.data;
  } catch (error) {
    if (signal?.aborted) throw abortError();
    if (timedOut) throw new LabClientError('Request timed out', !body);
    if (error instanceof LabClientError) throw error;
    throw new LabClientError('Connection unavailable', !body);
  } finally {
    clearTimeout(timeout);
    signal?.removeEventListener('abort', cancel);
  }
}

export const getPreflight = (signal?: AbortSignal) => request('preflight', LabPreflightSchema, signal);
export const listScenarios = (signal?: AbortSignal) => request('scenarios', ScenarioListSchema, signal);
export function startRun(body: StartRunRequest, signal?: AbortSignal) {
  const parsed = StartRunRequestSchema.safeParse(body);
  if (!parsed.success) return Promise.reject(new LabClientError('Invalid run request'));
  // The caller owns request_id for this intent. A lost POST is never retried here.
  return request('runs', RunHandleSchema, signal, parsed.data);
}
export function getRun(runId: string, signal?: AbortSignal) {
  if (!z.string().uuid().safeParse(runId).success) return Promise.reject(new LabClientError('Invalid run ID'));
  return request(`runs/${runId}`, PairedRunResultSchema, signal);
}

function delay(milliseconds: number, signal: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    if (signal.aborted) { reject(abortError()); return; }
    const cancel = () => { clearTimeout(timer); reject(abortError()); };
    const timer = setTimeout(() => { signal.removeEventListener('abort', cancel); resolve(); }, milliseconds);
    signal.addEventListener('abort', cancel, { once: true });
  });
}

export async function pollRun(runId: string, signal: AbortSignal, onUpdate?: (run: PairedRunResult) => void): Promise<PairedRunResult> {
  let interval = 250;
  let failedReads = 0;
  while (!signal.aborted) {
    try {
      const run = await getRun(runId, signal);
      failedReads = 0;
      onUpdate?.(run);
      if (['complete', 'partial_failure', 'blocked'].includes(run.status)) return run;
    } catch (error) {
      if (!(error instanceof LabClientError) || !error.retryable || ++failedReads > 3) throw error;
    }
    await delay(interval, signal);
    interval = Math.min(interval * 2, 2000);
  }
  throw abortError();
}
