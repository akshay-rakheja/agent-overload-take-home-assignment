'use client';

import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import { PreflightPanel } from '@/components/lab/PreflightPanel';
import { ScenarioRunner } from '@/components/lab/ScenarioRunner';
import { ComparisonGrid } from '@/components/lab/ComparisonGrid';
import { errorMessage, getPreflight, listScenarios, pollRun, startRun } from '@/lib/lab/client';
import type { LabPreflight, PairedRunResult, RunHandle, Scenario } from '@/lib/lab/schema';
import { GmailLinkSchema } from '@/lib/lab/schema';
import './lab.css';

const statusLabels = {
  queued: 'Queued', resetting: 'Resetting fixtures', baseline_running: 'Baseline running',
  enhanced_running: 'Enhanced running', deterministic_running: 'Deterministic running',
  jev_running: 'Jev running', grading: 'Grading evidence', complete: 'Complete',
  partial_failure: 'Partial failure', blocked: 'Blocked',
};

export default function LabPage() {
  const [preflight, setPreflight] = useState<LabPreflight | null>(null);
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [uncertainStart, setUncertainStart] = useState(false);
  const [handle, setHandle] = useState<RunHandle | null>(null);
  const [run, setRun] = useState<PairedRunResult | null>(null);
  const [readStopped, setReadStopped] = useState(false);
  const [pairIndex, setPairIndex] = useState(0);
  const [connectMessage, setConnectMessage] = useState<string | null>(null);
  const lifecycle = useRef<AbortController | null>(null);
  const inFlight = useRef(false);

  useEffect(() => {
    const clearAnnouncerRole = () => {
      const container = document.querySelector('next-route-announcer');
      const el = container?.shadowRoot?.querySelector('#__next-route-announcer__');
      if (el && el.getAttribute('role') === 'alert') el.removeAttribute('role');
    };
    clearAnnouncerRole();
    const observer = new MutationObserver(clearAnnouncerRole);
    observer.observe(document.body, { childList: true, subtree: true });
    const timer = setInterval(clearAnnouncerRole, 50);
    return () => {
      observer.disconnect();
      clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    lifecycle.current = controller;
    void Promise.allSettled([getPreflight(controller.signal), listScenarios(controller.signal)]).then(([check, catalog]) => {
      if (controller.signal.aborted) return;
      if (check.status === 'fulfilled') setPreflight(check.value);
      else setError(errorMessage(check.reason));
      if (catalog.status === 'fulfilled') setScenarios(catalog.value.scenarios);
      else setError(errorMessage(catalog.reason));
    });
    return () => controller.abort();
  }, []);

  async function watchRun(current: RunHandle, signal: AbortSignal) {
    setReadStopped(false);
    try {
      await pollRun(current.run_id, signal, setRun);
      setBusy(false);
      inFlight.current = false;
    } catch (failure) {
      if (signal.aborted) return;
      setError(errorMessage(failure));
      setReadStopped(true);
      // The run may still be active. Only resume reads; never unlock a new start.
    }
  }

  async function begin(scenarioId: string) {
    const signal = lifecycle.current?.signal;
    if (!signal || signal.aborted || !preflight?.runnable || inFlight.current || uncertainStart) return;
    inFlight.current = true;
    setBusy(true);
    setError(null);
    setRun(null);
    setHandle(null);
    setPairIndex(0);
    let submitted = false;
    try {
      const fresh = await getPreflight(signal);
      if (signal.aborted) return;
      setPreflight(fresh);
      if (!fresh.runnable) { setBusy(false); inFlight.current = false; return; }
      const request = { request_id: crypto.randomUUID(), scenario_ids: [scenarioId] };
      submitted = true;
      const started = await startRun(request, signal);
      if (signal.aborted) return;
      setHandle(started);
      await watchRun(started, signal);
    } catch (failure) {
      if (signal.aborted) return;
      setError(errorMessage(failure));
      if (submitted) setUncertainStart(true);
      else { setBusy(false); inFlight.current = false; }
    }
  }

  const status = run?.status ?? handle?.status;
  const currentScenario = scenarios.find((scenario) => scenario.scenario_id === run?.pairs[pairIndex]?.scheduled.scenario_id)
    ?? scenarios.find((scenario) => scenario.scenario_id === run?.request.scenario_ids[0]);
  const outcomeStatuses = run?.pairs.flatMap((pair) => pair.outcomes.map((outcome) => outcome.status)) ?? [];
  const failedSides = outcomeStatuses.filter((outcome) => outcome !== 'success').length;
  return <main className="lab-shell">
    <header className="lab-header">
      <div><h1>Evaluation Lab</h1><p>Follow the evidence through a paired agent run.</p></div>
      <Link href="/">Open chat</Link>
    </header>
    <PreflightPanel preflight={preflight} onConnect={() => {
      const signal = lifecycle.current?.signal;
      void fetch('/api/lab/gmail/link', { method: 'POST', cache: 'no-store', signal, headers: { Accept: 'application/json' } })
        .then(async (response) => {
          const parsed = GmailLinkSchema.safeParse(await response.json());
          setConnectMessage(response.ok && parsed.success ? parsed.data.message : 'Gmail handoff could not be verified');
        }).catch(() => { if (!signal?.aborted) setConnectMessage('Gmail handoff is unavailable'); });
    }} />
    {connectMessage && <p className="gmail-handoff" role="status" aria-live="polite">{connectMessage}</p>}
    {error && <div className="lab-error" role="alert"><p>{error}</p>
      {uncertainStart && <p>The start could not be confirmed. Submission is locked to avoid running the scenario again. Check the local run records before starting another run.</p>}
      {!busy && !uncertainStart && <button onClick={async () => {
        const signal = lifecycle.current?.signal;
        try { const fresh = await getPreflight(signal); setPreflight(fresh); setError(null); }
        catch (failure) { if (!signal?.aborted) setError(errorMessage(failure)); }
      }}>Recheck preflight</button>}
      {readStopped && handle && <button onClick={() => {
        const signal = lifecycle.current?.signal;
        if (signal && !signal.aborted) { setError(null); void watchRun(handle, signal); }
      }}>Resume status checks</button>}
    </div>}
    <ScenarioRunner scenarios={scenarios} runnable={preflight?.runnable ?? false} busy={busy || uncertainStart} onStart={(id) => void begin(id)} />
    <div className="lab-live-status" role="status" aria-live="polite"><span>{status ? statusLabels[status] : busy ? 'Checking readiness' : 'Awaiting a run'}</span>{run?.execution_mode === 'offline_fake' && <span>Offline fixture run</span>}{run?.blocked_reason && <span>{run.blocked_reason}</span>}</div>
    {run && currentScenario ? <>
      <section className="run-summary" aria-labelledby="run-summary-title">
        <div><p className="eyebrow">Paired evidence</p><h2 id="run-summary-title">Run summary</h2></div>
        <dl><div><dt>Repetitions</dt><dd>{run.pairs.length} {run.pairs.length === 1 ? 'repetition' : 'repetitions'}</dd></div><div><dt>Side outcomes</dt><dd>{outcomeStatuses.filter((outcome) => outcome === 'success').length} successful</dd></div><div><dt>Failures</dt><dd>{failedSides} {failedSides === 1 ? 'failed side' : 'failed sides'}</dd></div><div><dt>Sequence scorecards</dt><dd>{run.scorecards.filter((scorecard) => scorecard.passed).length} pass / {run.scorecards.filter((scorecard) => !scorecard.passed).length} fail</dd></div></dl>
      </section>
      <nav className="repetition-nav" aria-label="Repetition results"><span>Inspect repetition</span>{run.pairs.map((pair, index) => {
        const failures = pair.outcomes.filter((outcome) => outcome.status !== 'success').length;
        return <button key={pair.scheduled.pair_id} aria-label={`Repetition ${pair.scheduled.repetition}, ${failures} ${failures === 1 ? 'failure' : 'failures'}`} aria-current={index === pairIndex ? 'true' : undefined} onClick={() => setPairIndex(index)}>Repetition {pair.scheduled.repetition}<span>{failures} {failures === 1 ? 'failure' : 'failures'}</span></button>;
      })}</nav>
      <ComparisonGrid run={run} scenario={currentScenario} pairIndex={pairIndex} />
    </> : <section className="lab-awaiting" aria-labelledby="awaiting-title"><p className="eyebrow">Evidence browser</p><h2 id="awaiting-title">One protocol, two visible paths</h2><p>Run a server-approved scenario to inspect aligned baseline and enhanced evidence by layer.</p></section>}
    {handle && <footer className="lab-footer">Run ID: <span>{handle.run_id}</span></footer>}
  </main>;
}
