'use client';

import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { SystemRunResult } from '@/lib/lab/schema';
import { availabilityLabel, displayScalar, isRecord } from './ObservedValue';

export function GmailEvidencePanel({ result, track }: { result: SystemRunResult; track: 'controlled' | 'exploratory' }) {
  const [open, setOpen] = useState(false);
  const opener = useRef<HTMLButtonElement>(null);
  const dialog = useRef<HTMLDivElement>(null);
  const evidence = Array.isArray(result.gmail_evidence.value) ? result.gmail_evidence.value.filter(isRecord) : [];
  const controlledContent = evidence.flatMap((item) => Array.isArray(item.controlled_fixture_evidence) ? item.controlled_fixture_evidence.filter(isRecord) : []).filter((item) => item.fabricated === true && typeof item.content === 'string');
  const expandable = track === 'controlled' && controlledContent.length > 0;
  useEffect(() => {
    if (!open) return;
    const background = document.querySelector('main.lab-shell');
    const trigger = opener.current;
    const previousAriaHidden = background?.getAttribute('aria-hidden');
    background?.setAttribute('inert', '');
    background?.setAttribute('aria-hidden', 'true');
    return () => {
      background?.removeAttribute('inert');
      if (previousAriaHidden === null) background?.removeAttribute('aria-hidden');
      else if (previousAriaHidden !== undefined) background?.setAttribute('aria-hidden', previousAriaHidden);
      trigger?.focus();
    };
  }, [open]);
  const close = () => setOpen(false);
  const handleDialogKey = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape') { event.preventDefault(); close(); return; }
    if (event.key !== 'Tab') return;
    const focusable = dialog.current?.querySelectorAll<HTMLElement>('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
    if (!focusable?.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if ((event.shiftKey && document.activeElement === first) || (!event.shiftKey && document.activeElement === last)) {
      event.preventDefault();
      (event.shiftKey ? last : first).focus();
    }
  };
  return <section className="evidence-panel" data-testid="evidence-layer" aria-labelledby={`${result.turn_id}-gmail`}>
    <div className="evidence-heading"><h4 id={`${result.turn_id}-gmail`}>Gmail evidence</h4><span>{availabilityLabel(result.gmail_evidence.availability)}</span></div>
    {evidence.length ? <ol className="gmail-list">{evidence.map((item, index) => <li key={index}>
      <p><strong>{item.allowed === false ? 'Policy rejected' : item.allowed === true ? 'Read-only allowed' : item.operation_name === 'GMAIL_FETCH_EMAILS' && item.stage === 'completed' ? 'Read-only completed' : displayScalar(item.stage)}</strong> · {displayScalar(item.operation_name)}</p>
      <dl className="metric-grid compact">
        <div><dt>Boundary</dt><dd>{displayScalar(item.boundary)}</dd></div><div><dt>Stage</dt><dd>{displayScalar(item.stage)}</dd></div>
        {item.policy_code !== undefined && <div><dt>Policy</dt><dd>{displayScalar(item.policy_code)}</dd></div>}{item.sdk_executed !== undefined && <div><dt>SDK executed</dt><dd>{displayScalar(item.sdk_executed)}</dd></div>}
        <div><dt>Results</dt><dd>{displayScalar(item.result_count)}</dd></div>{item.attachment_count !== undefined && <div><dt>Attachments</dt><dd>{displayScalar(item.attachment_count)}</dd></div>}{item.has_more !== undefined && <div><dt>More pages</dt><dd>{displayScalar(item.has_more)}</dd></div>}
        <div><dt>Facts</dt><dd>{Array.isArray(item.fact_ids) ? item.fact_ids.map(displayScalar).join(', ') : 'Unavailable'}</dd></div>{item.query_sha256 !== undefined && <div><dt>Query fingerprint</dt><dd className="mono-break">{displayScalar(item.query_sha256)}</dd></div>}
      </dl>
    </li>)}</ol> : <p className="empty-evidence">{result.gmail_evidence.reason}</p>}
    {expandable && <button ref={opener} className="evidence-disclosure" onClick={() => setOpen(true)}>Show fabricated evidence</button>}
    {track === 'exploratory' && evidence.length > 0 && <p className="panel-note">Exploratory evidence remains collapsed</p>}
    {open && createPortal(<div ref={dialog} className="evidence-dialog" role="dialog" aria-modal="true" aria-label="Fabricated Gmail evidence" onKeyDown={handleDialogKey}>
      <div><h5>Fabricated Gmail evidence</h5>{controlledContent.map((item, index) => <p key={index}>{displayScalar(item.content)}</p>)}<button autoFocus onClick={close}>Close evidence</button></div>
    </div>, document.body)}
  </section>;
}
