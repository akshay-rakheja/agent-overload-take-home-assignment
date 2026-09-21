import { describe, expect, it } from 'vitest';
import { z } from 'zod';
import backend from '../../tests/fixtures/backend.json';
import { preflight } from '../../tests/fixtures/preflight';
import { AvailabilitySchema, observedValue, LabPreflightSchema, ScenarioListSchema, ScenarioSchema, RunHandleSchema, PairedRunResultSchema, SystemRunResultSchema, StartRunRequestSchema } from './schema';

describe('Python serialization boundary', () => {
  it('accepts complete actual Python scenario, run, handle and result serialization', () => {
    expect(ScenarioListSchema.parse(backend.scenarios).scenario_count).toBe(14);
    expect(RunHandleSchema.parse(backend.handle).status).toBe('queued');
    expect(PairedRunResultSchema.parse(backend.run).scorecards[0].passed).toBe(false);
    expect(SystemRunResultSchema.parse(backend.system_result).selected_identity.availability).toBe('inferred');
  });
  it('requires the backend-authored scenario track enum', () => {
    const scenario = backend.scenarios.scenarios[0];
    const { track: _track, ...withoutTrack } = scenario;
    expect(ScenarioSchema.safeParse({ ...scenario, track: 'controlled' }).success).toBe(true);
    expect(ScenarioSchema.safeParse({ ...scenario, track: 'natural' }).success).toBe(true);
    expect(ScenarioSchema.safeParse(withoutTrack).success).toBe(false);
    expect(ScenarioSchema.safeParse({ ...scenario, track: 'exploratory' }).success).toBe(false);
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
  it.each([
    { authorization_code: 'private' },
    { nested: { authorizationCode: 'private' } },
    { nested: [{ 'authorization-code': 'private' }] },
    { auth_code: 'private' },
    { oauth: { code: 'private' } },
    { nested: [{ oauth_response: { flow: [{ code: 'private' }] } }] },
  ])('rejects authorization codes and contextual OAuth codes recursively: %j', (value) => {
    expect(SystemRunResultSchema.safeParse({ ...backend.system_result, decision: { availability: 'available', value, reason: null } }).success).toBe(false);
  });
  it.each([
    { authorization_code: { value: 'fixture-code' } },
    { authorizationCode: ['fixture-code'] },
    { nested: [{ 'AUTHORIZATION-CODE': { value: 'fixture-code' } }] },
    { oauth_code: { value: 'fixture-code' } },
    { authorization_client_secret: { value: 'fixture-secret' } },
    { Authorization_Response: [{ CODE: 'fixture-code' }] },
  ])('rejects secret-shaped keys regardless of object, array, nesting or casing: %j', (value) => {
    expect(SystemRunResultSchema.safeParse({
      ...backend.system_result,
      gmail_evidence: { availability: 'available', value: [value], reason: null },
    }).success).toBe(false);
  });
  it.each([
    { Authorization: 'Bearer fixture-private' },
    { 'Proxy-Authorization': 'Basic fixture-private' },
    { 'X-Api-Key': 'fixture-private' },
    { Cookie: 'session=fixture-private' },
  ])('rejects raw values inside recognized header collections: %j', (headers) => {
    expect(SystemRunResultSchema.safeParse({
      ...backend.system_result,
      gmail_evidence: { availability: 'available', value: [{ headers }], reason: null },
    }).success).toBe(false);
  });
  it('preserves ordinary diagnostic codes and OAuth status without a code', () => {
    const value = { code: 'timeout', oauth: { status: 'connected' } };
    expect(SystemRunResultSchema.safeParse({ ...backend.system_result, decision: { availability: 'available', value, reason: null } }).success).toBe(true);
  });
  it.each([
    ['mail body', { body: 'Private appointment notes' }],
    ['message ID', { message_id: 'abcdef123456' }],
    ['thread ID', { thread_id: 'fedcba654321' }],
    ['Cookie header', { headers: { Cookie: 'session=fixture' } }],
    ['authorization response code', { authorization_response: { code: 'fixture-code' } }],
    ['nested authorization code', { nested: { authorization_result: { payload: { code: 'fixture-code' } } } }],
    ['free-text key', { text: 'key=fixture-private' }],
  ])('rejects raw %s inside JSON-valued evidence', (_label, value) => {
    expect(SystemRunResultSchema.safeParse({
      ...backend.system_result,
      gmail_evidence: { availability: 'available', value: [value], reason: null },
    }).success).toBe(false);
  });
  it('preserves redacted mail fields, safe tool evidence and non-secret diagnostic JSON', () => {
    const value = [{
      body: '[REDACTED]',
      message_id: '[REDACTED]',
      thread_id: '[REDACTED]',
      headers: {
        Authorization: '[REDACTED]',
        'Proxy-Authorization': '[REDACTED]',
        'X-Api-Key': '[REDACTED]',
        Cookie: '[REDACTED]',
      },
      tool: 'gmail.search',
      count: 2,
      code: 'timeout',
      oauth: { status: 'connected' },
      authorization: { status: 'denied' },
      generic: { nested: true, values: [0, false, null] },
    }];
    expect(SystemRunResultSchema.safeParse({
      ...backend.system_result,
      gmail_evidence: { availability: 'available', value, reason: null },
    }).success).toBe(true);
  });
  it('accepts the documented prospective preflight and fails closed on missing proof', () => {
    expect(LabPreflightSchema.parse(preflight).runnable).toBe(true);
    expect(LabPreflightSchema.safeParse({ runnable: true }).success).toBe(false);
    expect(LabPreflightSchema.safeParse({ ...preflight, schema_version: 2 }).success).toBe(false);
  });
  it('rejects raw mailbox addresses and OAuth handoff URLs in displayable fields', () => {
    for (const reason of ['Mailbox person@example.test failed', 'https://accounts.example.test/oauth?code=private']) {
      expect(LabPreflightSchema.safeParse({ ...preflight, blockers: [reason] }).success).toBe(false);
    }
  });
  it('accepts only the current backend start body and unique bounded scenario IDs', () => {
    expect(StartRunRequestSchema.safeParse({ request_id: backend.handle.request_id, scenario_ids: ['fixture'] }).success).toBe(true);
    expect(StartRunRequestSchema.safeParse({ scenarioId: 'fixture', repetitions: 3 }).success).toBe(false);
    expect(StartRunRequestSchema.safeParse({ request_id: backend.handle.request_id, scenario_ids: ['fixture', 'fixture'] }).success).toBe(false);
  });
});
