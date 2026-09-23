'use client';

import { useEffect, useState } from 'react';
import type { PairedRunResult, Scenario, SystemRunResult } from '@/lib/lab/schema';
import { CandidateEvidenceTable } from './CandidateEvidenceTable';
import { GmailEvidencePanel } from './GmailEvidencePanel';
import { HistoryDepthPanel } from './HistoryDepthPanel';
import { IdentityRoster } from './IdentityRoster';
import { JevInspectionPanel } from './JevInspectionPanel';
import { LayerScorecard } from './LayerScorecard';
import { PromptExposurePanel } from './PromptExposurePanel';
import { RoutingDispatchPanel } from './RoutingDispatchPanel';
import { UsageCostPanel } from './UsageCostPanel';
import { ObservedValue } from './ObservedValue';

const systemConfig: Record<string, { subhead: string; title: string }> = {
  baseline: { subhead: 'Historical system', title: 'Baseline evidence' },
  enhanced: { subhead: 'Directory + bounded context', title: 'Enhanced evidence' },
  enhanced_deterministic: { subhead: 'Deterministic retrieval', title: 'Enhanced Deterministic evidence' },
  enhanced_jev: { subhead: 'Jev Map/Reduce routing', title: 'Enhanced Jev evidence' },
};

function ResultLayer({ title, result, children }: { title: string; result: SystemRunResult; children: React.ReactNode }) {
  return <section className="evidence-panel response-panel" data-testid="evidence-layer"><h4>{title}</h4>{children}</section>;
}

export function ComparisonGrid({ run, scenario, pairIndex }: { run: PairedRunResult; scenario: Scenario; pairIndex: number }) {
  const pair = run.pairs[pairIndex];
  const [turnIndex, setTurnIndex] = useState(0);
  useEffect(() => setTurnIndex(0), [pairIndex]);
  const turnCount = Math.max(1, scenario.turn_count);
  const track: 'controlled' | 'exploratory' = scenario.track === 'natural' ? 'exploratory' : 'controlled';

  const isThreeWay = Boolean(
    pair?.outcomes.some(
      (item) => item.system === 'enhanced_deterministic' || item.system === 'enhanced_jev'
    )
  );

  const displayedSystems = isThreeWay
    ? (['baseline', 'enhanced_deterministic', 'enhanced_jev'] as const)
    : (['baseline', 'enhanced'] as const);

  return (
    <>
      <nav className="turn-nav" aria-label="Turn evidence">
        <span>Inspect turn</span>
        {Array.from({ length: turnCount }, (_, index) => (
          <button
            key={index}
            type="button"
            aria-current={index === turnIndex ? 'true' : undefined}
            onClick={() => setTurnIndex(index)}
          >
            Turn {index + 1}
          </button>
        ))}
      </nav>
      <div className={`comparison-grid ${isThreeWay ? 'three-way' : ''}`}>
        {displayedSystems.map((system) => {
          const outcome = pair?.outcomes.find((item) => item.system === system);
          const result = outcome?.results[turnIndex];
          const scorecard = run.scorecards.find(
            (item) => item.pair_id === pair?.scheduled.pair_id && item.system === system
          );
          const config = systemConfig[system] ?? { subhead: 'System evidence', title: `${system} evidence` };
          return (
            <section className={`comparison-side ${system}`} aria-labelledby={`${system}-evidence-title`} key={system}>
              <header className="side-heading">
                <p>{config.subhead}</p>
                <h2 id={`${system}-evidence-title`}>{config.title}</h2>
                <span className={`outcome outcome-${outcome?.status ?? 'unavailable'}`}>
                  {outcome?.status === 'success' ? '✓ Success' : `× ${(outcome?.status ?? 'unavailable').replaceAll('_', ' ')}`}
                </span>
              </header>
              {result ? (
                <>
                  <ResultLayer title="Revision & mode" result={result}>
                    <dl className="metric-grid">
                      <ObservedValue label="Revision fingerprint" observation={result.revision} />
                      <ObservedValue label="Run mode" observation={result.mode} />
                    </dl>
                  </ResultLayer>
                  <IdentityRoster result={result} />
                  <PromptExposurePanel result={result} />
                  <CandidateEvidenceTable result={result} />
                  <RoutingDispatchPanel result={result} />
                  {isThreeWay && <JevInspectionPanel result={result} />}
                  <GmailEvidencePanel result={result} track={track} />
                  <HistoryDepthPanel result={result} />
                  <UsageCostPanel result={result} modelId={outcome?.model_id ?? null} />
                  <ResultLayer title="Final response" result={result}>
                    <dl><ObservedValue label="Response" observation={result.final_response} /></dl>
                  </ResultLayer>
                  <LayerScorecard scorecard={scorecard} turnIndex={turnIndex} />
                </>
              ) : (
                <>
                  {Array.from({ length: isThreeWay ? 11 : 10 }, (_, index) => {
                    const layerTitles = isThreeWay
                      ? ['Revision & mode', 'Identity roster', 'Prompt exposure', 'Candidate evidence', 'Routing & dispatch', 'Jev Map/Reduce inspection', 'Gmail evidence', 'History depth', 'Usage & cost', 'Final response', 'Layer scorecard']
                      : ['Revision & mode', 'Identity roster', 'Prompt exposure', 'Candidate evidence', 'Routing & dispatch', 'Gmail evidence', 'History depth', 'Usage & cost', 'Final response', 'Layer scorecard'];
                    return (
                      <section className="evidence-panel unavailable-layer" data-testid="evidence-layer" key={index}>
                        <h4>{layerTitles[index]}</h4>
                        <p>{index === 0 ? outcome?.reason ?? 'No result emitted for this side.' : 'Evidence unavailable for this layer.'}</p>
                      </section>
                    );
                  })}
                </>
              )}
            </section>
          );
        })}
        <aside className="protocol-spine" aria-labelledby="protocol-spine-title">
          <span className="track-badge">{track === 'controlled' ? 'Controlled fixture' : 'Exploratory track'}</span>
          <h2 id="protocol-spine-title">Protocol spine</h2>
          <p>{scenario.title}</p>
          <ol>
            <li>Reset</li>
            {isThreeWay ? (
              <>
                <li>Baseline</li>
                <li>Deterministic</li>
                <li>Jev</li>
              </>
            ) : (
              <>
                <li>Baseline</li>
                <li>Enhanced</li>
              </>
            )}
            <li>Grade</li>
          </ol>
          <dl>
            <div><dt>Repetition</dt><dd>{pair?.scheduled.repetition ?? '—'}</dd></div>
            <div><dt>Turn</dt><dd>{turnIndex + 1} / {turnCount}</dd></div>
            <div><dt>Order</dt><dd>{pair?.scheduled.order.replaceAll('_', ' → ') ?? '—'}</dd></div>
          </dl>
        </aside>
      </div>
    </>
  );
}
