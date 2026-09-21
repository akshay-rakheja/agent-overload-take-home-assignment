'use client';

import { useState } from 'react';
import type { SystemRunResult } from '@/lib/lab/schema';
import { availabilityLabel, displayScalar, isRecord } from './ObservedValue';

export function GmailEvidencePanel({ result, track }: { result: SystemRunResult; track: 'controlled' | 'exploratory' }) {
  const [open, setOpen] = useState(false);
  const evidence = Array.isArray(result.gmail_evidence.value) ? result.gmail_evidence.value.filter(isRecord) : [];
  const expandable = track === 'controlled' && evidence.some((item) => item.fabricated === true && typeof item.full_fabricated_content === 'string');
  return <section className="evidence-panel" data-testid="evidence-layer" aria-labelledby={`${result.turn_id}-gmail`}>
    <div className="evidence-heading"><h4 id={`${result.turn_id}-gmail`}>Gmail evidence</h4><span>{availabilityLabel(result.gmail_evidence.availability)}</span></div>
    {evidence.length ? <ol className="gmail-list">{evidence.map((item, index) => <li key={index}>
      <p><strong>{item.allowed === true ? 'Read-only allowed' : 'Policy rejected'}</strong> · {displayScalar(item.operation_name)}</p>
      <dl className="metric-grid compact"><div><dt>Policy</dt><dd>{displayScalar(item.policy_code)}</dd></div><div><dt>Results</dt><dd>{displayScalar(item.result_count)}</dd></div><div><dt>Facts</dt><dd>{Array.isArray(item.fact_ids) ? item.fact_ids.map(displayScalar).join(', ') : '—'}</dd></div><div><dt>Query fingerprint</dt><dd className="mono-break">{displayScalar(item.query_sha256)}</dd></div></dl>
      {item.content_excerpt && <p>{displayScalar(item.content_excerpt)}</p>}
    </li>)}</ol> : <p className="empty-evidence">{result.gmail_evidence.reason}</p>}
    {expandable && <button className="evidence-disclosure" onClick={() => setOpen(true)}>Show fabricated evidence</button>}
    {track === 'exploratory' && evidence.length > 0 && <p className="panel-note">Exploratory evidence remains collapsed</p>}
    {open && <div className="evidence-dialog" role="dialog" aria-modal="true" aria-label="Fabricated Gmail evidence">
      <div><h5>Fabricated Gmail evidence</h5>{evidence.map((item, index) => item.fabricated === true && <p key={index}>{displayScalar(item.full_fabricated_content)}</p>)}<button autoFocus onClick={() => setOpen(false)}>Close evidence</button></div>
    </div>}
  </section>;
}
