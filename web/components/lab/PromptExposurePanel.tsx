import type { SystemRunResult } from '@/lib/lab/schema';
import { availabilityLabel, displayScalar, isRecord } from './ObservedValue';

export function PromptExposurePanel({ result }: { result: SystemRunResult }) {
  const value = isRecord(result.prompt_exposure.value) ? result.prompt_exposure.value : {};
  return <section className="evidence-panel" data-testid="evidence-layer" aria-labelledby={`${result.turn_id}-prompt`}>
    <div className="evidence-heading"><h4 id={`${result.turn_id}-prompt`}>Prompt exposure</h4><span>{availabilityLabel(result.prompt_exposure.availability)}</span></div>
    {result.prompt_exposure.value !== null ? result.system === 'baseline' ? <dl className="metric-grid">
      <div><dt>Surface</dt><dd>Full-roster prompt</dd></div><div><dt>Names</dt><dd>{displayScalar(value.exposed_name_count)} names exposed</dd></div>
      <div><dt>Prompt characters</dt><dd>{displayScalar(value.prompt_characters)}</dd></div><div><dt>Prompt fingerprint</dt><dd className="mono-break">{displayScalar(value.prompt_xml_sha256)}</dd></div>
    </dl> : <><dl className="metric-grid"><div><dt>Surface</dt><dd>{displayScalar(value.surface)}</dd></div><div><dt>Candidates</dt><dd>{displayScalar(value.candidate_count)} candidates exposed</dd></div><div><dt>Routing action</dt><dd>{displayScalar(value.routing_action)}</dd></div><div><dt>Candidate IDs</dt><dd className="mono-break">{Array.isArray(value.candidate_ids) ? value.candidate_ids.map(displayScalar).join(', ') : 'Unavailable'}</dd></div></dl><details><summary>Candidate XML</summary><pre className="prompt-xml">{displayScalar(value.candidate_xml)}</pre></details></> : <p className="empty-evidence">{result.prompt_exposure.reason}</p>}
  </section>;
}
