// Prospective UI contract: the backend does not yet implement /lab/preflight.
export const preflight = {
  schema_version: 1,
  runnable: true,
  blockers: [],
  warnings: ['Controlled fabricated fixtures only.'],
  baseline: { reachable: true, revision: 'fixture-baseline', model: 'fixture-model', config_fingerprint: 'fixture-config' },
  enhanced: { reachable: true, revision: 'fixture-enhanced', model: 'fixture-model', config_fingerprint: 'fixture-config' },
  fixture_equivalence: { equivalent: true, reason: null },
  gmail_safety: { connected: true, read_only: true, reason: null },
  budget: { safe: true, remaining_usd: '1.25', reason: null },
};
