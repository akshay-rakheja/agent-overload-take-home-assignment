import type { PairedRunResult, Scenario, SystemRunResult } from '@/lib/lab/schema';
import { CandidateEvidenceTable } from './CandidateEvidenceTable';
import { GmailEvidencePanel } from './GmailEvidencePanel';
import { HistoryDepthPanel } from './HistoryDepthPanel';
import { IdentityRoster } from './IdentityRoster';
import { LayerScorecard } from './LayerScorecard';
import { PromptExposurePanel } from './PromptExposurePanel';
import { RoutingDispatchPanel } from './RoutingDispatchPanel';
import { UsageCostPanel } from './UsageCostPanel';
import { ObservedValue } from './ObservedValue';

const systems = ['baseline', 'enhanced'] as const;

function ResultLayer({ title, result, children }: { title: string; result: SystemRunResult; children: React.ReactNode }) {
  return <section className="evidence-panel response-panel" data-testid="evidence-layer"><h4>{title}</h4>{children}</section>;
}

export function ComparisonGrid({ run, scenario, pairIndex }: { run: PairedRunResult; scenario: Scenario; pairIndex: number }) {
  const pair = run.pairs[pairIndex];
  const track: 'controlled' | 'exploratory' = scenario.track === 'natural' ? 'exploratory' : 'controlled';
  return <div className="comparison-grid">
    {systems.map((system) => {
      const outcome = pair?.outcomes.find((item) => item.system === system);
      const result = outcome?.results[0];
      const scorecard = run.scorecards.find((item) => item.system === system && item.scenario_id === pair?.scheduled.scenario_id);
      return <section className={`comparison-side ${system}`} aria-labelledby={`${system}-evidence-title`} key={system}>
        <header className="side-heading"><p>{system === 'baseline' ? 'Historical system' : 'Directory + bounded context'}</p><h2 id={`${system}-evidence-title`}>{system === 'baseline' ? 'Baseline evidence' : 'Enhanced evidence'}</h2><span className={`outcome outcome-${outcome?.status ?? 'unavailable'}`}>{outcome?.status === 'success' ? '✓ Success' : `× ${(outcome?.status ?? 'unavailable').replaceAll('_', ' ')}`}</span></header>
        {result ? <>
          <ResultLayer title="Revision & mode" result={result}><dl className="metric-grid"><ObservedValue label="Revision fingerprint" observation={result.revision} /><ObservedValue label="Run mode" observation={result.mode} /></dl></ResultLayer>
          <IdentityRoster result={result} /><PromptExposurePanel result={result} /><CandidateEvidenceTable result={result} /><RoutingDispatchPanel result={result} /><GmailEvidencePanel result={result} track={track} /><HistoryDepthPanel result={result} /><UsageCostPanel result={result} modelId={outcome?.model_id ?? null} />
          <ResultLayer title="Final response" result={result}><dl><ObservedValue label="Response" observation={result.final_response} /></dl></ResultLayer>
          <LayerScorecard scorecard={scorecard} />
        </> : <>{Array.from({ length: 10 }, (_, index) => <section className="evidence-panel unavailable-layer" data-testid="evidence-layer" key={index}><h4>{['Revision & mode', 'Identity roster', 'Prompt exposure', 'Candidate evidence', 'Routing & dispatch', 'Gmail evidence', 'History depth', 'Usage & cost', 'Final response', 'Layer scorecard'][index]}</h4><p>{index === 0 ? outcome?.reason ?? 'No result emitted for this side.' : 'Evidence unavailable for this layer.'}</p></section>)}</>}
      </section>;
    })}
    <aside className="protocol-spine" aria-labelledby="protocol-spine-title">
      <span className="track-badge">{track === 'controlled' ? 'Controlled fixture' : 'Exploratory track'}</span>
      <h2 id="protocol-spine-title">Protocol spine</h2><p>{scenario.title}</p>
      <ol><li>Reset</li><li>Baseline</li><li>Enhanced</li><li>Grade</li></ol>
      <dl><div><dt>Repetition</dt><dd>{pair?.scheduled.repetition ?? '—'}</dd></div><div><dt>Order</dt><dd>{pair?.scheduled.order.replaceAll('_', ' → ') ?? '—'}</dd></div></dl>
    </aside>
  </div>;
}
