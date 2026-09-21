import type { JsonValue, ObservedValue as Observation } from '@/lib/lab/schema';

const labels = { available: 'Available', inferred: 'Inferred', not_applicable: 'Not applicable', unavailable: 'Unavailable' } as const;

export function isRecord(value: unknown): value is Record<string, JsonValue> {
  return value !== null && !Array.isArray(value) && typeof value === 'object';
}

export function displayScalar(value: unknown): string {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'number') return value.toLocaleString('en-US', { maximumFractionDigits: 6 });
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}

export function availabilityLabel(availability: Observation<unknown>['availability']) {
  return labels[availability];
}

export function ObservedValue<T>({ label, observation, format = displayScalar }: {
  label: string; observation: Observation<T>; format?: (value: T) => string;
}) {
  const supported = observation.availability === 'available' || observation.availability === 'inferred';
  return <div className="observed-value" data-availability={observation.availability}>
    <dt>{label}</dt>
    <dd>
      <span className="availability-tag">{availabilityLabel(observation.availability)}</span>
      {supported && observation.value !== null && <span className="observed-content">{format(observation.value)}</span>}
      {observation.reason && <span className="observed-reason">{observation.reason}</span>}
    </dd>
  </div>;
}
