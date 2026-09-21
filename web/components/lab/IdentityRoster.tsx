import type { SystemRunResult } from '@/lib/lab/schema';
import { availabilityLabel, displayScalar, isRecord } from './ObservedValue';

export function IdentityRoster({ result }: { result: SystemRunResult }) {
  const exposure = isRecord(result.prompt_exposure.value) ? result.prompt_exposure.value : {};
  const exposedNames = Array.isArray(exposure.exposed_names) ? exposure.exposed_names.filter((item): item is string => typeof item === 'string') : [];
  const candidates = Array.isArray(result.candidates.value) ? result.candidates.value.filter(isRecord) : [];
  const selected = isRecord(result.selected_identity.value) ? result.selected_identity.value : {};
  return <section className="evidence-panel identity-panel" data-testid="evidence-layer" aria-labelledby={`${result.turn_id}-roster`}>
    <div className="evidence-heading"><h4 id={`${result.turn_id}-roster`}>Identity roster</h4><span>{availabilityLabel(result.roster_count.availability)}</span></div>
    <p className="evidence-kicker">Roster size: {result.roster_count.value === null ? '—' : displayScalar(result.roster_count.value)}</p>
    {selected.name !== undefined && <p className="selected-summary">Selected: {displayScalar(selected.name)}</p>}
    {result.system === 'baseline' && exposedNames.length ? <div className="table-scroll" tabIndex={0} aria-label="baseline exposed names scroll area"><table><caption className="sr-only">baseline names in prompt order</caption><thead><tr><th>Name</th></tr></thead><tbody>{exposedNames.map((name, index) => {
      const selectedRow = name === selected.name;
      return <tr key={`${name}-${index}`} aria-selected={selectedRow || undefined}><td><strong>{name}</strong>{selectedRow && <span className="selected-mark">Selected identity</span>}</td></tr>;
    })}</tbody></table></div> : result.system === 'enhanced' && candidates.length ? <div className="table-scroll" tabIndex={0} aria-label="enhanced candidate identities scroll area"><table><caption className="sr-only">enhanced candidate identities in producer order</caption><thead><tr><th>Name</th><th>Stable ID</th><th>Purpose</th><th>Status</th></tr></thead><tbody>{candidates.map((identity, index) => {
      const selectedRow = identity.agent_id === selected.agent_id;
      return <tr key={String(identity.agent_id ?? index)} aria-selected={selectedRow || undefined}><td><strong>{displayScalar(identity.name)}</strong>{selectedRow && <span className="selected-mark">Selected identity</span>}</td><td className="mono-break">{displayScalar(identity.agent_id)}</td><td>{displayScalar(identity.purpose)}</td><td>{displayScalar(identity.status)}</td></tr>;
    })}</tbody></table></div> : <p className="empty-evidence">{result.prompt_exposure.reason ?? result.candidates.reason ?? 'Identity details were not emitted.'}</p>}
  </section>;
}
