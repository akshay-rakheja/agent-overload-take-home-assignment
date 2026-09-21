import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import backend from '../../tests/fixtures/backend.json';
import type { PairedRunResult, Scenario, SystemRunResult } from '@/lib/lab/schema';
import { ObservedValue } from './ObservedValue';
import { ComparisonGrid } from './ComparisonGrid';
import { IdentityRoster } from './IdentityRoster';
import { PromptExposurePanel } from './PromptExposurePanel';
import { CandidateEvidenceTable } from './CandidateEvidenceTable';
import { RoutingDispatchPanel } from './RoutingDispatchPanel';
import { GmailEvidencePanel } from './GmailEvidencePanel';
import { HistoryDepthPanel } from './HistoryDepthPanel';
import { UsageCostPanel } from './UsageCostPanel';
import { LayerScorecard } from './LayerScorecard';

const run = backend.evidence_run as PairedRunResult;
const scenario = backend.scenarios.scenarios[0] as Scenario;
const pair = run.pairs[0];
const baseline = pair.outcomes[0].results[0] as SystemRunResult;
const enhanced = pair.outcomes[1].results[0] as SystemRunResult;

describe('Evaluation Lab evidence panels', () => {
  it('distinguishes numeric zero from inferred, unavailable, and not applicable observations', () => {
    render(<dl>
      <ObservedValue label="Zero tokens" observation={{ availability: 'available', value: 0, reason: null }} />
      <ObservedValue label="Inferred route" observation={baseline.decision} />
      <ObservedValue label="Missing evidence" observation={baseline.gmail_evidence} />
      <ObservedValue label="Unsupported rank" observation={baseline.candidates} />
    </dl>);
    expect(screen.getByText('0')).toBeInTheDocument();
    expect(screen.getByText('Inferred')).toBeInTheDocument();
    expect(screen.getByText('Unavailable')).toBeInTheDocument();
    expect(screen.getByText('Not applicable')).toBeInTheDocument();
    expect(screen.queryByText(/^0(?:\.0+)?$/, { selector: '[data-availability="not_applicable"] *' })).not.toBeInTheDocument();
  });

  it('renders every identity in source order and keeps the selected identity marked', () => {
    render(<IdentityRoster result={enhanced} />);
    const rows = screen.getAllByRole('row').slice(1);
    expect(rows).toHaveLength(6);
    expect(rows.map((row) => within(row).getByText(/.+/, { selector: 'strong' }).textContent)).toEqual(['Atlas', 'Beacon', 'Cedar', 'Delta', 'Ember', 'Flint']);
    expect(screen.getByText('Selected identity')).toBeInTheDocument();
    expect(screen.getByText('Selected: Atlas')).toBeInTheDocument();
    expect(screen.getByText('Roster size: 6')).toBeInTheDocument();
  });

  it('renders full-roster and bounded prompt exposure without deriving values', () => {
    const { rerender } = render(<PromptExposurePanel result={baseline} />);
    expect(screen.getByText('Full roster exposed')).toBeInTheDocument();
    expect(screen.getByText('18,420')).toBeInTheDocument();
    expect(screen.getByText('6 identities exposed')).toBeInTheDocument();
    rerender(<PromptExposurePanel result={enhanced} />);
    expect(screen.getByText('Bounded exposure')).toBeInTheDocument();
    expect(screen.getByText('3 identities exposed')).toBeInTheDocument();
    expect(screen.getByText('enhanced-fixture-sha256')).toBeInTheDocument();
  });

  it('keeps candidate rank, score, components, and reasons in exact server array order', () => {
    render(<CandidateEvidenceTable result={enhanced} />);
    const rows = screen.getAllByRole('row').slice(1);
    expect(rows.map((row) => within(row).getByText(/.+/, { selector: 'strong' }).textContent)).toEqual(['Atlas', 'Cedar', 'Delta']);
    expect(rows.map((row) => within(row).getAllByRole('cell')[0].textContent)).toEqual(['1', '2', '3']);
    expect(rows.map((row) => within(row).getAllByRole('cell')[3].textContent)).toEqual(['0.94', '0.61', '0.37']);
    fireEvent.click(within(rows[0]).getByRole('button', { name: 'Expand candidate Atlas' }));
    fireEvent.click(within(rows[2]).getByRole('button', { name: 'Expand candidate Delta' }));
    expect(within(rows[0]).getByText('semantic 0.72')).toBeInTheDocument();
    expect(within(rows[0]).getByText('security intent match')).toBeInTheDocument();
    expect(within(rows[2]).getByText('weak review overlap')).toBeInTheDocument();
    expect(screen.getAllByRole('row')).toHaveLength(4);
  });

  it('renders baseline ranking gaps as not applicable and never as rank or score zero', () => {
    render(<CandidateEvidenceTable result={baseline} />);
    expect(screen.getAllByText('Not applicable').length).toBeGreaterThan(0);
    expect(screen.queryByText('Rank 0')).not.toBeInTheDocument();
    expect(screen.queryByText('Score 0')).not.toBeInTheDocument();
  });

  it('shows expectation, recommendation, authorization, dispatch, reuse, identity delta, and duplicates', () => {
    render(<RoutingDispatchPanel result={enhanced} />);
    expect(screen.getByText('reuse Atlas')).toBeInTheDocument();
    expect(screen.getByText('Highest deterministic score.')).toBeInTheDocument();
    expect(screen.getAllByText('00000000-0000-4000-8000-000000000001').length).toBeGreaterThan(0);
    expect(screen.getByText('accepted')).toBeInTheDocument();
    expect(screen.getByText('Reused')).toBeInTheDocument();
    expect(screen.getByText('6 → 6')).toBeInTheDocument();
    expect(screen.getByText('No duplicates reported')).toBeInTheDocument();
  });

  it('shows sanitized Gmail facts and expands only controlled fabricated content', () => {
    render(<GmailEvidencePanel result={enhanced} track="controlled" />);
    expect(screen.getByText('Read-only allowed')).toBeInTheDocument();
    expect(screen.getByText('fixture-security-001')).toBeInTheDocument();
    expect(screen.queryByText('A controlled fixture reports a new sign-in and recommends reviewing account activity.')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Show fabricated evidence' }));
    expect(screen.getByRole('dialog', { name: 'Fabricated Gmail evidence' })).toBeInTheDocument();
    expect(screen.getByText('A controlled fixture reports a new sign-in and recommends reviewing account activity.')).toBeInTheDocument();
  });

  it('does not expose a disclosure control for exploratory evidence', () => {
    render(<GmailEvidencePanel result={enhanced} track="exploratory" />);
    expect(screen.getByText('Exploratory evidence remains collapsed')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /fabricated evidence/i })).not.toBeInTheDocument();
  });

  it('renders the complete raw, rendered, omission, summary, preservation, and contamination history facts', () => {
    render(<HistoryDepthPanel result={enhanced} />);
    for (const value of ['984,220 bytes', '10,000 entries', '4,000 characters', '8 episodes', '9,968 omitted', '24 truncated', 'Summary used', 'Raw history preserved', 'No contamination detected']) {
      expect(screen.getByText(value)).toBeInTheDocument();
    }
  });

  it('renders latency, tokens, model, provider, and exact cost including zero cached tokens', () => {
    render(<UsageCostPanel result={enhanced} modelId="fixture-model-v1" />);
    expect(screen.getByText('fixture-model-v1')).toBeInTheDocument();
    expect(screen.getAllByText('fixture-provider').length).toBeGreaterThan(0);
    expect(screen.getByText('436.8 ms')).toBeInTheDocument();
    expect(screen.getByText('968')).toBeInTheDocument();
    expect(screen.getByText('126')).toBeInTheDocument();
    expect(screen.getByText('0')).toBeInTheDocument();
    expect(screen.getByText('0.0048 USD')).toBeInTheDocument();
  });

  it('renders every layer with pass/fail text and expected-versus-actual evidence', () => {
    render(<LayerScorecard scorecard={run.scorecards[1]} />);
    expect(screen.getByRole('table', { name: 'Enhanced layer scorecard' })).toBeInTheDocument();
    expect(screen.getAllByText(/Pass|Fail|Missing|Contradictory|Not applicable/).length).toBeGreaterThanOrEqual(6);
    expect(screen.getByText('Expected evidence')).toBeInTheDocument();
    expect(screen.getByText('Observed gaps or contradictions')).toBeInTheDocument();
    expect(screen.getByText('context evidence satisfied bounded-history rules')).toBeInTheDocument();
  });

  it('keeps both evidence columns aligned when one side times out', () => {
    const failed = structuredClone(run);
    failed.status = 'partial_failure';
    failed.pairs[0].outcomes[1] = { ...failed.pairs[0].outcomes[1], status: 'timeout', reason: 'Enhanced side exceeded its time limit.', results: [] };
    render(<ComparisonGrid run={failed} scenario={scenario} pairIndex={0} />);
    expect(screen.getByRole('heading', { name: 'Baseline evidence' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Enhanced evidence' })).toBeInTheDocument();
    expect(screen.getByText('Atlas found the controlled fixture and summarized the security alert.')).toBeInTheDocument();
    expect(screen.getByText('Enhanced side exceeded its time limit.')).toBeInTheDocument();
    expect(screen.getAllByTestId('evidence-layer')).toHaveLength(20);
  });

  it('composes the complete mirrored evidence story with a central protocol spine', () => {
    render(<ComparisonGrid run={run} scenario={scenario} pairIndex={0} />);
    expect(screen.getByRole('heading', { name: 'Baseline evidence' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Protocol spine' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Enhanced evidence' })).toBeInTheDocument();
    expect(screen.getAllByTestId('evidence-layer')).toHaveLength(20);
    expect(screen.getByText('Controlled fixture')).toBeInTheDocument();
    expect(screen.getByText('Exact named Instagram security reuse')).toBeInTheDocument();
  });
});
