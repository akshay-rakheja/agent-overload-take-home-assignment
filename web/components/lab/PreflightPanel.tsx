import type { LabPreflight } from '@/lib/lab/schema';

export function PreflightPanel({ preflight, onConnect }: { preflight: LabPreflight | null; onConnect?: () => void }) {
  return <section className="lab-preflight" aria-labelledby="preflight-title">
    <div className="lab-section-heading">
      <h2 id="preflight-title">Preflight</h2>
      <span className="lab-readiness" data-ready={preflight?.runnable ?? false}>
        {preflight ? preflight.runnable ? 'Ready to run' : 'Run blocked' : 'Awaiting server verification'}
      </span>
    </div>
    {preflight ? <>
      <dl className="lab-checks">
        {(['baseline', 'enhanced'] as const).map((side) => <div key={side}>
          <dt>{side === 'baseline' ? 'Baseline' : 'Enhanced'}</dt>
          <dd>{preflight[side].reachable ? 'Backend reachable' : 'Backend unavailable'}</dd>
          <dd>{preflight[side].revision ?? 'Revision unavailable'}</dd>
          <dd>{preflight[side].model ?? 'Model unavailable'}</dd>
          <dd className="lab-subtle">Configuration: {preflight[side].config_fingerprint ?? 'unavailable'}</dd>
        </div>)}
        <div><dt>Fixtures</dt><dd>{preflight.fixture_equivalence.equivalent ? 'Equivalent' : 'Not equivalent'}</dd><dd>{preflight.fixture_equivalence.reason}</dd></div>
        <div><dt>Gmail policy</dt><dd>{preflight.gmail_safety.connected ? 'Connected' : 'Not connected'}</dd><dd>{preflight.gmail_safety.read_only ? 'Read-only policy' : 'Policy unverified'}</dd><dd>{preflight.gmail_safety.reason}</dd></div>
        <div><dt>Budget</dt><dd>{preflight.budget.safe ? 'Within server limit' : 'Not cleared'}</dd><dd>{preflight.budget.remaining_usd === null ? 'Remaining budget unavailable' : `$${preflight.budget.remaining_usd} remaining`}</dd><dd>{preflight.budget.reason}</dd></div>
      </dl>
      {!preflight.gmail_safety.connected && onConnect && <button className="gmail-connect" type="button" onClick={onConnect}>Connect Gmail</button>}
      {preflight.blockers.length > 0 && <div className="lab-blockers"><h3>Blockers</h3><ul>{preflight.blockers.map((item, index) => <li key={index}>{item}</li>)}</ul></div>}
      {preflight.warnings.length > 0 && <div className="lab-warnings"><h3>Warnings</h3><ul>{preflight.warnings.map((item, index) => <li key={index}>{item}</li>)}</ul></div>}
    </> : <p>Run stays disabled until the server verifies both systems, fixtures, model configuration, Gmail policy, and budget.</p>}
  </section>;
}
