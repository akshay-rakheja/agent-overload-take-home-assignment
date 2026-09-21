import { describe, expect, it } from 'vitest';
import { z } from 'zod';
import backend from '../../tests/fixtures/backend.json';
import { preflight } from '../../tests/fixtures/preflight';
import { AvailabilitySchema, observedValue, LabPreflightSchema, ScenarioListSchema, RunHandleSchema, PairedRunResultSchema, SystemRunResultSchema, StartRunRequestSchema } from './schema';

describe('Python serialization boundary', () => {
  it('accepts complete actual Python scenario, run, handle and result serialization', () => {
    expect(ScenarioListSchema.parse(backend.scenarios).scenario_count).toBe(14);
    expect(RunHandleSchema.parse(backend.handle).status).toBe('queued');
    expect(PairedRunResultSchema.parse(backend.run).scorecards[0].passed).toBe(false);
    expect(SystemRunResultSchema.parse(backend.system_result).selected_identity.availability).toBe('inferred');
  });
  it('preserves null, zero, inferred and unavailable without coercion', () => {
    const schema = observedValue(z.number());
    expect(schema.parse({ availability: 'available', value: 0, reason: null }).value).toBe(0);
    expect(schema.parse({ availability: 'inferred', value: 3, reason: 'From state' }).value).toBe(3);
    expect(schema.parse({ availability: 'available', value: null, reason: null }).value).toBeNull();
    expect(schema.safeParse({ availability: 'unavailable', value: 3, reason: 'Missing' }).success).toBe(false);
    expect(schema.safeParse({ availability: 'not_applicable', value: 0, reason: null }).success).toBe(false);
    expect(AvailabilitySchema.safeParse('unknown').success).toBe(false);
  });
  it.each(['schema_version', 'usage', 'cost', 'revision'])('rejects missing serialized field %s', (field) => {
    const payload: Record<string, unknown> = { ...backend.system_result };
    delete payload[field];
    expect(SystemRunResultSchema.safeParse(payload).success).toBe(false);
  });
  it.each([{ schema_version: 2 }, { extra: true }, { access_token: 'private' }])('rejects unknown versions and unexpected fields %j', (extra) => {
    expect(SystemRunResultSchema.safeParse({ ...backend.system_result, ...extra }).success).toBe(false);
  });
  it('rejects nested secret-shaped fields and unsafe text at the display boundary', () => {
    for (const value of [{ access_token: '[REDACTED]' }, { nested: { clientSecret: 'private' } }, { text: 'Bearer private-token' }]) {
      expect(SystemRunResultSchema.safeParse({ ...backend.system_result, decision: { availability: 'available', value, reason: null } }).success).toBe(false);
    }
  });
  it('accepts the documented prospective preflight and fails closed on missing proof', () => {
    expect(LabPreflightSchema.parse(preflight).runnable).toBe(true);
    expect(LabPreflightSchema.safeParse({ runnable: true }).success).toBe(false);
    expect(LabPreflightSchema.safeParse({ ...preflight, schema_version: 2 }).success).toBe(false);
  });
  it('accepts only the current backend start body and unique bounded scenario IDs', () => {
    expect(StartRunRequestSchema.safeParse({ request_id: backend.handle.request_id, scenario_ids: ['fixture'] }).success).toBe(true);
    expect(StartRunRequestSchema.safeParse({ scenarioId: 'fixture', repetitions: 3 }).success).toBe(false);
    expect(StartRunRequestSchema.safeParse({ request_id: backend.handle.request_id, scenario_ids: ['fixture', 'fixture'] }).success).toBe(false);
  });
});
