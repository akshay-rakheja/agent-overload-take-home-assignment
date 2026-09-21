import type { ObservedValue as Observation, SystemRunResult } from '@/lib/lab/schema';
import { availabilityLabel, displayScalar, isRecord } from './ObservedValue';

function Field({ label, observation, fields }: { label: string; observation: Observation<unknown>; fields?: string[] }) {
  const value = isRecord(observation.value) ? observation.value : null;
  return <div className="routing-field"><dt>{label}</dt><dd><span className="availability-tag">{availabilityLabel(observation.availability)}</span>
    {value ? (fields ?? Object.keys(value)).map((key) => value[key] !== undefined && <span className="stacked" key={key}><small>{key.replaceAll('_', ' ')}: </small><span>{displayScalar(value[key])}</span></span>) : observation.value !== null ? displayScalar(observation.value) : observation.reason}
  </dd></div>;
}

export function RoutingDispatchPanel({ result }: { result: SystemRunResult }) {
  const selected = isRecord(result.selected_identity.value) ? result.selected_identity.value : {};
  const delta = isRecord(result.identity_delta.value) ? result.identity_delta.value : {};
  const before = delta.directory_count_before ?? delta.roster_before_count;
  const after = delta.directory_count_after ?? delta.roster_after_count;
  const duplicates = Array.isArray(result.duplicates.value) ? result.duplicates.value : null;
  return <section className="evidence-panel" data-testid="evidence-layer" aria-labelledby={`${result.turn_id}-routing`}>
    <div className="evidence-heading"><h4 id={`${result.turn_id}-routing`}>Routing & dispatch</h4><span>{availabilityLabel(result.decision.availability)}</span></div>
    <dl className="routing-list">
      <Field label="Expectation" observation={result.decision} fields={['expectation', 'reason', 'action', 'agent_id']} />
      <Field label="Deterministic recommendation" observation={result.recommendation} />
      <div className="routing-field"><dt>Authorized set</dt><dd><span className="availability-tag">{availabilityLabel(result.authorized_ids.availability)}</span>{result.authorized_ids.value?.map((id) => <span className="stacked mono-break" key={id}>{id}</span>) ?? result.authorized_ids.reason}</dd></div>
      <Field label="Attempted dispatch" observation={result.attempted_dispatch} />
      <Field label="Accepted dispatch" observation={result.accepted_dispatch} />
      <div className="routing-field"><dt>Identity outcome</dt><dd><span>{selected.reused === true ? 'Reused' : result.created_identity.value ? 'Created' : availabilityLabel(result.created_identity.availability)}</span>{selected.name && <span className="stacked">{displayScalar(selected.name)}</span>}</dd></div>
      <div className="routing-field"><dt>Identity count</dt><dd>{before !== undefined && after !== undefined ? `${displayScalar(before)} → ${displayScalar(after)}` : result.identity_delta.reason}</dd></div>
      <div className="routing-field"><dt>Duplicates</dt><dd>{duplicates?.length === 0 ? 'No duplicates reported' : duplicates ? duplicates.map((item, index) => <span className="stacked" key={index}>{displayScalar(item)}</span>) : result.duplicates.reason}</dd></div>
    </dl>
  </section>;
}
