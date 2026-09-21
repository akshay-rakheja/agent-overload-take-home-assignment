import type { SystemRunResult } from '@/lib/lab/schema';
import { ObservedValue, displayScalar, isRecord } from './ObservedValue';

export function UsageCostPanel({ result, modelId }: { result: SystemRunResult; modelId: string | null }) {
  const timings = Array.isArray(result.timings.value) ? result.timings.value.filter(isRecord) : [];
  const total = [...timings].reverse().find((timing) => timing.phase === 'total') ?? timings.at(-1);
  return <section className="evidence-panel" data-testid="evidence-layer" aria-labelledby={`${result.turn_id}-usage`}>
    <div className="evidence-heading"><h4 id={`${result.turn_id}-usage`}>Usage & cost</h4><span>Server reported</span></div>
    <dl className="metric-grid usage-grid">
      <div><dt>Model</dt><dd>{modelId ?? 'Unavailable'}</dd></div><div><dt>Provider</dt><dd>{total ? displayScalar(total.provider) : 'Unavailable'}</dd></div><div><dt>Latency</dt><dd>{total ? `${displayScalar(total.latency_ms)} ms` : result.timings.reason}</dd></div>
      <ObservedValue label="Input tokens" observation={result.usage.input_tokens} /><ObservedValue label="Output tokens" observation={result.usage.output_tokens} /><ObservedValue label="Cached tokens" observation={result.usage.cached_tokens} /><ObservedValue label="Total tokens" observation={result.usage.total_tokens} />
      <ObservedValue label="Cost" observation={result.cost.amount} format={(amount) => `${displayScalar(amount)} ${result.cost.currency.value ?? ''}`.trim()} />
    </dl>
  </section>;
}
