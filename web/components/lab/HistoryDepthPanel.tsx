import type { SystemRunResult } from '@/lib/lab/schema';
import { availabilityLabel, displayScalar, isRecord } from './ObservedValue';

export function HistoryDepthPanel({ result }: { result: SystemRunResult }) {
  const value = isRecord(result.context_metrics.value) ? result.context_metrics.value : null;
  const facts: [string, string][] = value ? [
    ['Raw bytes', `${displayScalar(value.raw_history_bytes)} bytes`], ['Raw characters', `${displayScalar(value.raw_history_characters)} characters`], ['Raw entries', `${displayScalar(value.raw_entry_count)} entries`],
    ['Rendered characters', `${displayScalar(value.rendered_characters)} characters`], ['Rendered bytes', `${displayScalar(value.rendered_bytes)} bytes`], ['Rendered episodes', `${displayScalar(value.included_episode_count)} episodes`],
    ['Omissions', `${displayScalar(value.omitted_entry_count)} omitted`], ['Truncations', `${displayScalar(value.truncated_entry_count)} truncated`],
    ['Summary', value.summary_used === true ? 'Summary used' : 'Summary not used'],
    ['Preservation', value.preserved_raw_history === undefined ? 'Unavailable' : value.preserved_raw_history === true ? 'Raw history preserved' : 'Raw history not preserved'],
    ['Contamination', value.contamination_detected === undefined ? 'Unavailable' : value.contamination_detected === true ? 'Contamination detected' : 'No contamination detected'],
  ] : [];
  return <section className="evidence-panel" data-testid="evidence-layer" aria-labelledby={`${result.turn_id}-history`}>
    <div className="evidence-heading"><h4 id={`${result.turn_id}-history`}>History depth</h4><span>{availabilityLabel(result.context_metrics.availability)}</span></div>
    {value ? <dl className="metric-grid">{facts.map(([label, fact]) => <div key={label}><dt>{label}</dt><dd>{fact}</dd></div>)}</dl> : <p className="empty-evidence">{result.context_metrics.reason}</p>}
  </section>;
}
