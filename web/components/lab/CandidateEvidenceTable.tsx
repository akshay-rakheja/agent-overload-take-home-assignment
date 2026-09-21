'use client';

import { useState } from 'react';
import type { SystemRunResult } from '@/lib/lab/schema';
import { availabilityLabel, displayScalar, isRecord } from './ObservedValue';

export function CandidateEvidenceTable({ result }: { result: SystemRunResult }) {
  const [expanded, setExpanded] = useState<Record<number, boolean>>({});
  const candidates = Array.isArray(result.candidates.value) ? result.candidates.value.filter(isRecord).slice(0, 5) : [];
  return <section className="evidence-panel" data-testid="evidence-layer" aria-labelledby={`${result.turn_id}-candidates`}>
    <div className="evidence-heading"><h4 id={`${result.turn_id}-candidates`}>Candidate evidence</h4><span>{availabilityLabel(result.candidates.availability)}</span></div>
    {candidates.length ? <div className="table-scroll" tabIndex={0} aria-label={`${result.system} candidate table scroll area`}><table><caption className="sr-only">{result.system} candidates in server order</caption><thead><tr><th>Rank</th><th>Identity</th><th>Status</th><th>Score</th><th>Components</th><th>Reasons</th></tr></thead>
      <tbody>{candidates.map((candidate, index) => <tr key={String(candidate.agent_id ?? index)}>
        <td>{displayScalar(candidate.rank)}</td><td><button className="candidate-toggle" type="button" aria-expanded={expanded[index] ?? false} aria-label={`${expanded[index] ? 'Collapse' : 'Expand'} candidate ${displayScalar(candidate.name)}`} onClick={() => setExpanded((current) => ({ ...current, [index]: !current[index] }))}><strong>{displayScalar(candidate.name)}</strong></button><span className="mono-break">{displayScalar(candidate.agent_id)}</span></td>
        <td>{displayScalar(candidate.status)}</td><td>{displayScalar(candidate.score)}</td>
        <td>{expanded[index] ? isRecord(candidate.components) ? Object.entries(candidate.components).map(([key, value]) => <span className="stacked" key={key}>{key} {displayScalar(value)}</span>) : '—' : <span className="lab-subtle">Collapsed</span>}</td>
        <td>{expanded[index] ? Array.isArray(candidate.reasons) ? candidate.reasons.map((reason, reasonIndex) => <span className="stacked" key={reasonIndex}>{displayScalar(reason)}</span>) : '—' : <span className="lab-subtle">Collapsed</span>}</td>
      </tr>)}</tbody></table></div> : <p className="empty-evidence"><strong>{availabilityLabel(result.candidates.availability)}</strong>{result.candidates.reason && <> · {result.candidates.reason}</>}</p>}
  </section>;
}
