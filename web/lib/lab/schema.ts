import { z } from 'zod';

// These are wire schemas: all serialized fields (including nulls) are required.
// They mirror Tasks 07–10 Python models, not the older illustrative TS sketch.
export const AvailabilitySchema = z.enum(['available', 'inferred', 'not_applicable', 'unavailable']);
const text = z.string().refine((value) => !/\bbearer\s+[\w.~+/=-]+|\bsk-[\w-]{8,}|\b(?:api[_ -]?key|access[_ -]?token|client[_ -]?secret|password|token|secret|key)\b\s*[:=]\s*\S+|[\w.+-]+@[\w.-]+\.[a-z]{2,}|https?:\/\/\S*(?:oauth|authorize|[?&](?:code|token|state)=)/i.test(value), 'Unsafe text');
const uuid = z.string().uuid();
const integer = z.number().int();
const system = z.enum(['baseline', 'enhanced']);
const nullableText = text.nullable();
const secretKey = /(?:authorizationcode|authcode|apikey|authconfigid|oauthcode|accesstoken|refreshtoken|idtoken|clientsecret|password|secret|token)$/i;
const mailKeys = new Set(['email', 'emailaddress', 'address', 'from', 'to', 'cc', 'bcc', 'sender', 'recipient', 'messageid', 'threadid', 'gmailmessageid', 'gmailthreadid', 'snippet', 'body', 'rawbody', 'htmlbody']);
export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };
const normalizedKey = (key: string) => key.replace(/[^a-z0-9]/gi, '').toLowerCase();
const isRedacted = (value: JsonValue) => value === '[REDACTED]' || value === '[REDACTED_EMAIL]';
const hasRedactedHeaders = (value: JsonValue) => {
  if (value === '[REDACTED]') return true;
  return value !== null && !Array.isArray(value) && typeof value === 'object'
    && Object.values(value).every((item) => item === '[REDACTED]');
};
function containsCodeField(value: JsonValue): boolean {
  if (Array.isArray(value)) return value.some(containsCodeField);
  if (value === null || typeof value !== 'object') return false;
  return Object.entries(value).some(([key, item]) => normalizedKey(key) === 'code' || containsCodeField(item));
}
function containsPrivateField(value: JsonValue): boolean {
  if (Array.isArray(value)) return value.some(containsPrivateField);
  if (value === null || typeof value !== 'object') return false;
  for (const [key, item] of Object.entries(value)) {
    const normalized = normalizedKey(key);
    const mailKey = mailKeys.has(normalized) || normalized.includes('address')
      || normalized.endsWith('messageid') || normalized.endsWith('threadid');
    const headerKey = normalized === 'headers' || normalized.endsWith('headers');
    const authorizationWrapper = (normalized.includes('oauth') || normalized.startsWith('authorization'))
      && item !== null && typeof item === 'object';
    if (headerKey) {
      if (!hasRedactedHeaders(item)) return true;
      continue;
    }
    if (mailKey) {
      if (!isRedacted(item)) return true;
      continue;
    }
    if (secretKey.test(normalized) || normalized === 'proxyauthorization'
      || (normalized === 'authorization' && !authorizationWrapper)
      || (authorizationWrapper && containsCodeField(item))) return true;
    if (containsPrivateField(item)) return true;
  }
  return false;
}
const jsonShape: z.ZodType<JsonValue> = z.lazy(() => z.union([
  text, z.number().finite(), z.boolean(), z.null(), z.array(jsonShape), z.record(jsonShape),
]));
const jsonValue: z.ZodType<JsonValue> = jsonShape.superRefine((value, ctx) => {
  if (containsPrivateField(value)) {
    ctx.addIssue({ code: z.ZodIssueCode.custom, message: 'Unexpected private field' });
  }
});

export function observedValue<T extends z.ZodTypeAny>(value: T) {
  return z.object({ availability: AvailabilitySchema, value: value.nullable(), reason: nullableText }).strict()
    .refine((item) => !['unavailable', 'not_applicable'].includes(item.availability) || item.value === null, 'Unsupported observations cannot carry a value');
}
const observation = observedValue(jsonValue);
const observedList = observedValue(z.array(jsonValue));
const usage = z.object({
  input_tokens: observedValue(integer), output_tokens: observedValue(integer),
  cached_tokens: observedValue(integer), total_tokens: observedValue(integer),
  known_input_tokens_subtotal: observedValue(integer), known_output_tokens_subtotal: observedValue(integer),
  known_cached_tokens_subtotal: observedValue(integer), known_total_tokens_subtotal: observedValue(integer),
}).strict();

export const SystemRunResultSchema = z.object({
  schema_version: z.literal(1), run_id: uuid, turn_id: uuid, system,
  revision: observedValue(text), mode: observedValue(text), roster_count: observedValue(integer),
  prompt_exposure: observation, candidates: observedList, decision: observation,
  recommendation: observation, authorized_ids: observedValue(z.array(text)),
  attempted_dispatch: observation, accepted_dispatch: observation,
  selected_identity: observation, created_identity: observation, identity_delta: observation,
  duplicates: observedList, gmail_evidence: observedList, final_response: observedValue(text),
  context_metrics: observation, timings: observedList, usage,
  cost: z.object({ amount: observedValue(z.number().finite()), currency: observedValue(text), known_amount_subtotal: observedValue(z.number().finite()) }).strict(),
  errors: observedList, availability_metadata: z.record(AvailabilitySchema),
}).strict();

export const ScenarioSchema = z.object({
  scenario_id: text, track: z.enum(['controlled', 'natural']), family: text, title: text, repetitions: integer.min(1), optional: z.boolean(), budget_guarded: z.boolean(),
  reset_profile: z.object({ profile_id: text, fixture_seed: integer, roster_size: z.union([z.literal(10), z.literal(100), z.literal(500), z.literal(1000)]), history_entries: integer.min(0) }).strict(),
  turn_count: integer.min(1),
}).strict();
export const ScenarioListSchema = z.object({ schema_version: z.literal(1), scenario_count: integer.min(0), scenarios: z.array(ScenarioSchema) }).strict();
export const StartRunRequestSchema = z.object({
  request_id: uuid,
  scenario_ids: z.array(z.string().trim().min(1).max(200).pipe(text)).refine((ids) => new Set(ids).size === ids.length, 'Scenario IDs must be unique'),
}).strict();
export const RunStatusSchema = z.enum(['queued', 'resetting', 'baseline_running', 'enhanced_running', 'grading', 'complete', 'partial_failure', 'blocked']);
export const RunHandleSchema = z.object({ run_id: uuid, request_id: uuid, status: RunStatusSchema }).strict();
const scheduledPair = z.object({ pair_id: uuid, scenario_id: text, repetition: integer.min(1), order: z.enum(['baseline_then_enhanced', 'enhanced_then_baseline']) }).strict();
const outcomeStatus = z.enum(['success', 'failure', 'timeout', 'budget_stop', 'malformed', 'unavailable']);
const lateCompletion = z.object({ status: outcomeStatus, model_id: nullableText, results: z.array(SystemRunResultSchema), reason: nullableText }).strict();
const sideOutcome = lateCompletion.extend({ system, attempt_count: z.literal(1), late_completion: lateCompletion.nullable() }).strict();
const stateVerification = z.object({
  system, snapshot_id: text, roster_fingerprint: text, expected_roster_fingerprint: text,
  fixture_fingerprint: text, expected_fixture_fingerprint: text,
  raw_journal_fingerprint: text, expected_raw_journal_fingerprint: text,
}).strict();
const layerGrade = z.object({
  layer: z.enum(['routing', 'response', 'gmail_safety', 'identity', 'duplicate', 'context']),
  status: z.enum(['pass', 'fail', 'missing', 'contradictory', 'not_applicable']),
  positive_evidence: z.array(text), negative_evidence: z.array(text), missing_evidence: z.array(text), contradictory_evidence: z.array(text),
}).strict();
const scorecard = z.object({
  schema_version: z.literal(1), scenario_id: text, system, routing: layerGrade, response: layerGrade,
  gmail_safety: layerGrade, identity: layerGrade, duplicate: layerGrade, context: layerGrade,
  candidate_rank: observedValue(integer), passed: z.boolean(),
}).strict();
const sequenceScorecard = z.object({
  schema_version: z.literal(1), scenario_id: text, system, turns: z.array(scorecard), identity_continuity: layerGrade, passed: z.boolean(),
}).strict();
export const PairedRunResultSchema = z.object({
  schema_version: z.literal(1), run_id: uuid, request: StartRunRequestSchema,
  execution_mode: z.enum(['measured', 'offline_fake']),
  snapshot_contracts: z.array(z.object({ profile_id: text, snapshot_id: text, baseline_roster_fingerprint: text, enhanced_roster_fingerprint: text, fixture_fingerprint: text, raw_journal_fingerprint: text }).strict()),
  status: RunStatusSchema, generation: integer.min(0), schedule: z.array(scheduledPair),
  pairs: z.array(z.object({ scheduled: scheduledPair, verifications: z.array(stateVerification), outcomes: z.array(sideOutcome) }).strict()
    .refine((pair) => [pair.verifications, pair.outcomes].every((items) => new Set(items.map((item) => item.system)).size === items.length), 'Duplicate system')),
  scorecards: z.array(sequenceScorecard),
  transitions: z.array(z.object({ sequence: integer.min(1), status: RunStatusSchema, occurred_at: text, detail: nullableText }).strict()),
  trace: z.array(z.object({ sequence: integer.min(1), kind: z.enum(['transition', 'reset', 'outcome', 'grade', 'recovery', 'late']), occurred_at: text, pair_id: uuid.nullable(), system: system.nullable(), detail: text }).strict()),
  blocked_reason: nullableText, created_at: text, updated_at: text,
}).strict();

// Prospective contracts only. No backend LabPreflight/Gmail lab route exists yet.
// A missing route or incomplete response is an error, never synthesized readiness.
const systemPreflight = z.object({ reachable: z.boolean(), revision: nullableText, model: nullableText, config_fingerprint: nullableText }).strict();
export const LabPreflightSchema = z.object({
  schema_version: z.literal(1), runnable: z.boolean(), blockers: z.array(text), warnings: z.array(text),
  baseline: systemPreflight, enhanced: systemPreflight,
  fixture_equivalence: z.object({ equivalent: z.boolean(), reason: nullableText }).strict(),
  gmail_safety: z.object({ connected: z.boolean(), read_only: z.boolean(), reason: nullableText }).strict(),
  budget: z.object({ safe: z.boolean(), remaining_usd: nullableText, reason: nullableText }).strict(),
}).strict();
export const GmailStatusSchema = z.object({ schema_version: z.literal(1), connected: z.boolean(), read_only: z.boolean(), message: text }).strict();
export const GmailLinkSchema = z.object({ schema_version: z.literal(1), available: z.boolean(), message: text }).strict();

// Proxy-owned error contract. Raw upstream error messages never cross this boundary.
export const LabProxyErrorSchema = z.discriminatedUnion('code', [
  z.object({ code: z.literal('UPSTREAM_TRANSPORT'), error: z.literal('Lab service unavailable') }).strict(),
  z.object({ code: z.literal('UPSTREAM_TIMEOUT'), error: z.literal('Request timed out') }).strict(),
  z.object({ code: z.literal('INVALID_UPSTREAM_RESPONSE'), error: z.literal('Lab response could not be verified') }).strict(),
]);

export type Availability = z.infer<typeof AvailabilitySchema>;
export type ObservedValue<T> = { availability: Availability; value: T | null; reason: string | null };
export type LabPreflight = z.infer<typeof LabPreflightSchema>;
export type Scenario = z.infer<typeof ScenarioSchema>;
export type StartRunRequest = z.infer<typeof StartRunRequestSchema>;
export type RunHandle = z.infer<typeof RunHandleSchema>;
export type PairedRunResult = z.infer<typeof PairedRunResultSchema>;
export type SystemRunResult = z.infer<typeof SystemRunResultSchema>;
