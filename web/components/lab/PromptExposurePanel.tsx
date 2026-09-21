import type { SystemRunResult } from '@/lib/lab/schema';
import { availabilityLabel, displayScalar, isRecord } from './ObservedValue';

export function PromptExposurePanel({ result }: { result: SystemRunResult }) {
  const value = isRecord(result.prompt_exposure.value) ? result.prompt_exposure.value : {};
  const count = value.exposed_name_count ?? value.exposed_identity_count;
  return <section className="evidence-panel" data-testid="evidence-layer" aria-labelledby={`${result.turn_id}-prompt`}>
    <div className="evidence-heading"><h4 id={`${result.turn_id}-prompt`}>Prompt exposure</h4><span>{availabilityLabel(result.prompt_exposure.availability)}</span></div>
    {result.prompt_exposure.value !== null ? <dl className="metric-grid">
      <div><dt>Exposure mode</dt><dd>{value.full_roster === true ? 'Full roster exposed' : 'Bounded exposure'}</dd></div>
      <div><dt>Identities</dt><dd>{displayScalar(count)} identities exposed</dd></div>
      <div><dt>Prompt characters</dt><dd>{displayScalar(value.prompt_characters)}</dd></div>
      <div><dt>Prompt fingerprint</dt><dd className="mono-break">{displayScalar(value.prompt_xml_sha256)}</dd></div>
    </dl> : <p className="empty-evidence">{result.prompt_exposure.reason}</p>}
  </section>;
}
