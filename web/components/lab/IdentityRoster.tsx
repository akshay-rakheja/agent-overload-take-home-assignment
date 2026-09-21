import type { SystemRunResult } from '@/lib/lab/schema';
import { availabilityLabel, displayScalar, isRecord } from './ObservedValue';

export function IdentityRoster({ result }: { result: SystemRunResult }) {
  const exposure = isRecord(result.prompt_exposure.value) ? result.prompt_exposure.value : {};
  const identities = Array.isArray(exposure.identities) ? exposure.identities.filter(isRecord) : [];
  const selected = isRecord(result.selected_identity.value) ? result.selected_identity.value : {};
  return <section className="evidence-panel identity-panel" data-testid="evidence-layer" aria-labelledby={`${result.turn_id}-roster`}>
    <div className="evidence-heading"><h4 id={`${result.turn_id}-roster`}>Identity roster</h4><span>{availabilityLabel(result.roster_count.availability)}</span></div>
    <p className="evidence-kicker">Roster size: {result.roster_count.value === null ? '—' : displayScalar(result.roster_count.value)}</p>
    {selected.name !== undefined && <p className="selected-summary">Selected: {displayScalar(selected.name)}</p>}
    {identities.length ? <div className="table-scroll" tabIndex={0} aria-label={`${result.system} identity roster scroll area`}><table><caption className="sr-only">{result.system} identity roster</caption><thead><tr><th>Name</th><th>Purpose</th><th>Status</th><th>Uses</th></tr></thead>
      <tbody>{identities.map((identity, index) => {
        const selectedRow = identity.agent_id === selected.agent_id;
        return <tr key={String(identity.agent_id ?? index)} aria-selected={selectedRow || undefined}>
          <td><strong>{displayScalar(identity.name)}</strong>{selectedRow && <span className="selected-mark">Selected identity</span>}</td>
          <td>{displayScalar(identity.purpose)}</td><td>{displayScalar(identity.status)}</td><td>{displayScalar(identity.use_count)}</td>
        </tr>;
      })}</tbody></table></div> : <p className="empty-evidence">{result.prompt_exposure.reason ?? 'No identity roster emitted.'}</p>}
  </section>;
}
