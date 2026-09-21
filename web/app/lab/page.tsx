'use client';

import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import { PreflightPanel } from '@/components/lab/PreflightPanel';
import { ScenarioRunner } from '@/components/lab/ScenarioRunner';
import { errorMessage, getPreflight, listScenarios, pollRun, startRun } from '@/lib/lab/client';
import type { LabPreflight, PairedRunResult, RunHandle, Scenario } from '@/lib/lab/schema';
import './lab.css';

const statusLabels = {
  queued: 'Queued', resetting: 'Resetting fixtures', baseline_running: 'Baseline running',
  enhanced_running: 'Enhanced running', grading: 'Grading evidence', complete: 'Complete',
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
  const lifecycle = useRef<AbortController | null>(null);
  const inFlight = useRef(false);

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
  return <main className="lab-shell">
    <header className="lab-header">
      <div><h1>Evaluation Lab</h1><p>Follow the evidence through a paired agent run.</p></div>
      <Link href="/">Open chat</Link>
    </header>
    <PreflightPanel preflight={preflight} />
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
    <div className="lab-comparison">
      <section className="lab-side" aria-labelledby="baseline-title">
        <h2 id="baseline-title">Baseline evidence</h2>
        <p className="lab-side-caption">Historical system</p>
        <SideEvidence run={run} side="baseline" />
      </section>
      <section className="lab-protocol" aria-labelledby="protocol-title">
        <h2 id="protocol-title">Protocol</h2>
        <p role="status" aria-live="polite">{status ? statusLabels[status] : busy ? 'Checking readiness' : 'Awaiting a run'}</p>
        {run?.execution_mode === 'offline_fake' && <p>Offline fixture run</p>}
        {run?.transitions.length ? <ol>{run.transitions.map((transition) => <li key={transition.sequence}>{transition.status.replaceAll('_', ' ')}{transition.detail && <p>{transition.detail}</p>}</li>)}</ol> : <p className="lab-subtle">Reset, compare, then grade.</p>}
        {run?.blocked_reason && <p>{run.blocked_reason}</p>}
      </section>
      <section className="lab-side" aria-labelledby="enhanced-title">
        <h2 id="enhanced-title">Enhanced evidence</h2>
        <p className="lab-side-caption">Directory and bounded context</p>
        <SideEvidence run={run} side="enhanced" />
      </section>
    </div>
    {handle && <footer className="lab-footer">Run ID: <span>{handle.run_id}</span></footer>}
  </main>;
}

function SideEvidence({ run, side }: { run: PairedRunResult | null; side: 'baseline' | 'enhanced' }) {
  if (!run) return <p className="lab-empty">Evidence will appear here when the server reports a result.</p>;
  return <>{run.pairs.map((pair) => {
    const outcome = pair.outcomes.find((item) => item.system === side);
    return <article className="lab-outcome" key={pair.scheduled.pair_id}>
      <h3>{pair.scheduled.scenario_id} / repetition {pair.scheduled.repetition}</h3>
      {outcome ? <>
        <p className="lab-outcome-status">{outcome.status.replaceAll('_', ' ')}</p>
        {outcome.model_id && <p className="lab-subtle">Model: {outcome.model_id}</p>}
        {outcome.reason && <p className="lab-outcome-reason">{outcome.reason}</p>}
        {outcome.results.map((result) => <div key={result.turn_id} className="lab-response">
          <h4>Final response <span>({result.final_response.availability.replaceAll('_', ' ')})</span></h4>
          <p>{result.final_response.value ?? result.final_response.reason ?? 'No response available'}</p>
        </div>)}
      </> : <p>No outcome reported yet.</p>}
    </article>;
  })}{run.pairs.length === 0 && <p className="lab-empty">No outcome reported yet.</p>}</>;
}
