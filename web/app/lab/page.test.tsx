import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import backend from '../../tests/fixtures/backend.json';
import { preflight } from '../../tests/fixtures/preflight';
import Page from './page';
import type { PairedRunResult } from '../../lib/lab/schema';

afterEach(() => vi.unstubAllGlobals());
const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status });

it('keeps Run disabled when preflight is absent or unavailable', async () => {
  vi.stubGlobal('fetch', async (url: string) => url.endsWith('scenarios') ? json(backend.scenarios) : json({ detail: 'private error' }, 404));
  render(<Page />);
  expect(screen.getByRole('button', { name: 'Run scenario' })).toBeDisabled();
  expect(await screen.findByText('Lab endpoint unavailable')).toBeVisible();
  expect(screen.getByRole('button', { name: 'Run scenario' })).toBeDisabled();
});

it('prevents double submission and retains both sides of a partial failure', async () => {
  const writes: unknown[] = [];
  let release: (response: Response) => void = () => {};
  vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
    if (url.endsWith('preflight')) return json(preflight);
    if (url.endsWith('scenarios')) return json(backend.scenarios);
    if (init.method === 'POST') { writes.push(JSON.parse(init.body as string)); return new Promise<Response>((resolve) => { release = resolve; }); }
    return json(backend.run);
  });
  render(<Page />);
  const button = screen.getByRole('button', { name: 'Run scenario' });
  await waitFor(() => expect(button).toBeEnabled());
  fireEvent.click(button);
  fireEvent.click(button);
  await waitFor(() => expect(writes).toHaveLength(1));
  expect(writes[0]).toMatchObject({ scenario_ids: [backend.scenarios.scenarios[0].scenario_id], request_id: expect.stringMatching(/^[0-9a-f-]{36}$/) });
  await act(async () => release(json(backend.handle, 202)));
  expect(await screen.findByText('Partial failure')).toBeVisible();
  expect(within(screen.getByRole('region', { name: 'Baseline evidence' })).getByText('reference: SEC-7419; timestamp: 2026-09-18 04:12 UTC; location: Lisbon; device: Pixel 10; verification phrase: indigo-orbit')).toBeVisible();
  expect(within(screen.getByRole('region', { name: 'Enhanced evidence' })).getByText('Enhanced side exceeded its time limit.')).toBeVisible();
});

it('aborts active reads when navigating away', async () => {
  const signals: AbortSignal[] = [];
  vi.stubGlobal('fetch', (_url: string, init: RequestInit) => { signals.push(init.signal as AbortSignal); return new Promise<Response>((_resolve, reject) => init.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')))); });
  const view = render(<Page />);
  view.unmount();
  expect(signals.length).toBeGreaterThan(0);
  expect(signals.every((signal) => signal.aborted)).toBe(true);
});

it('does not permit another submission after an unconfirmed start', async () => {
  const writes: number[] = [];
  vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
    if (url.endsWith('preflight')) return json(preflight);
    if (url.endsWith('scenarios')) return json(backend.scenarios);
    if (init.method === 'POST') { writes.push(1); throw new TypeError('network'); }
    return json(backend.run);
  });
  render(<Page />);
  const button = screen.getByRole('button', { name: 'Run scenario' });
  await waitFor(() => expect(button).toBeEnabled());
  fireEvent.click(button);
  expect(await screen.findByText(/The start could not be confirmed/)).toBeVisible();
  expect(button).toBeDisabled();
  fireEvent.click(button);
  expect(writes).toHaveLength(1);
});

it('can resume a failed status read without submitting another scenario', async () => {
  const methods: string[] = [];
  let failRead = true;
  vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
    if (url.endsWith('preflight')) return json(preflight);
    if (url.endsWith('scenarios')) return json(backend.scenarios);
    methods.push(init.method!);
    if (init.method === 'POST') return json(backend.handle, 202);
    return failRead ? json({}, 503) : json(backend.run);
  });
  render(<Page />);
  const button = screen.getByRole('button', { name: 'Run scenario' });
  await waitFor(() => expect(button).toBeEnabled());
  fireEvent.click(button);
  const resume = await screen.findByRole('button', { name: 'Resume status checks' });
  expect(button).toBeDisabled();
  failRead = false;
  fireEvent.click(resume);
  expect(await screen.findByText('Partial failure')).toBeVisible();
  expect(methods).toEqual(['POST', 'GET', 'GET']);
});

it('composes complete evidence, aggregate outcomes, and repetition navigation without hiding failures', async () => {
  const repeated = structuredClone(backend.evidence_run) as unknown as PairedRunResult;
  repeated.pairs.push({
    ...structuredClone(repeated.pairs[0]),
    scheduled: { ...repeated.pairs[0].scheduled, pair_id: '77777777-7777-4777-8777-777777777777', repetition: 2 },
    outcomes: [
      { ...structuredClone(repeated.pairs[0].outcomes[0]), status: 'failure', reason: 'Baseline fixture failed safely.', results: [] },
      structuredClone(repeated.pairs[0].outcomes[1]),
    ],
  });
  repeated.status = 'partial_failure';
  vi.stubGlobal('fetch', async (url: string, init: RequestInit) => {
    if (url.endsWith('preflight')) return json(preflight);
    if (url.endsWith('scenarios')) return json(backend.scenarios);
    if (init.method === 'POST') return json(backend.handle, 202);
    return json(repeated);
  });
  render(<Page />);
  const button = screen.getByRole('button', { name: 'Run scenario' });
  await waitFor(() => expect(button).toBeEnabled());
  fireEvent.click(button);
  expect(await screen.findByRole('heading', { name: 'Run summary' })).toBeVisible();
  expect(screen.getByText('2 repetitions')).toBeVisible();
  expect(screen.getByText('1 failed side')).toBeVisible();
  expect(screen.getByRole('button', { name: 'Repetition 1, 0 failures' })).toHaveAttribute('aria-current', 'true');
  fireEvent.click(screen.getByRole('button', { name: 'Repetition 2, 1 failure' }));
  expect(screen.getByRole('button', { name: 'Repetition 2, 1 failure' })).toHaveAttribute('aria-current', 'true');
  expect(screen.getByText('Baseline fixture failed safely.')).toBeVisible();
  expect(screen.getAllByTestId('evidence-layer')).toHaveLength(20);
});
