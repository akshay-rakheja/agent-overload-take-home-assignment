import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import backend from '../../tests/fixtures/backend.json';
import { PairedRunResultSchema, ScenarioSchema, type SystemRunResult } from '@/lib/lab/schema';
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

const run = PairedRunResultSchema.parse(backend.evidence_run);
const scenario = ScenarioSchema.parse(backend.scenarios.scenarios[0]);
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

  it('renders producer-ranked candidate identities in source order and keeps the selected identity marked', () => {
    render(<IdentityRoster result={enhanced} />);
    const rows = screen.getAllByRole('row').slice(1);
    expect(rows).toHaveLength(3);
    expect(rows.map((row) => within(row).getByText(/.+/, { selector: 'strong' }).textContent)).toEqual(['Instagram Security Monitor', 'Controlled Fixture Researcher', 'Account Security Auditor']);
    expect(screen.getByText('Selected identity')).toBeInTheDocument();
    expect(screen.getByText('Selected: Instagram Security Monitor')).toBeInTheDocument();
    expect(screen.getByText('Roster size: 100')).toBeInTheDocument();
  });

  it('renders the exact baseline and enhanced producer payload names', () => {
    expect(Object.keys(baseline.prompt_exposure.value as object).sort()).toEqual(['exposed_name_count', 'exposed_names', 'prompt_characters', 'prompt_xml_sha256']);
    expect(Object.keys(enhanced.prompt_exposure.value as object).sort()).toEqual(['candidate_count', 'candidate_ids', 'candidate_xml', 'routing_action', 'surface']);
    expect(Object.keys(enhanced.candidates.value![0] as object).sort()).toEqual(['agent_id', 'name', 'purpose', 'rank', 'reasons', 'score', 'score_components', 'status']);
    expect(Object.keys(enhanced.context_metrics.value as object).sort()).toEqual(['included_episode_count', 'omitted_entry_count', 'raw_entry_count', 'raw_history_bytes', 'raw_history_characters', 'rendered_bytes', 'rendered_characters', 'summary_used', 'truncated_entry_count']);
    expect(Object.keys((enhanced.timings.value as object[]).at(-1)!).sort()).toEqual(['elapsed_ns', 'finished_monotonic_ns', 'phase', 'started_monotonic_ns']);

    const { rerender } = render(<IdentityRoster result={baseline} />);
    expect(screen.getAllByRole('row').slice(1)).toHaveLength(100);
    expect(screen.getAllByRole('row')[1].textContent).toBe('Instagram Security MonitorSelected identity');
    rerender(<IdentityRoster result={enhanced} />);
    expect(screen.getAllByRole('row').slice(1)).toHaveLength(3);
    rerender(<PromptExposurePanel result={enhanced} />);
    expect(screen.getByText('interaction_agent_candidate_xml')).toBeVisible();
    expect(screen.getByText('3 candidates exposed')).toBeVisible();
    rerender(<CandidateEvidenceTable result={enhanced} />);
    fireEvent.click(screen.getByRole('button', { name: 'Expand candidate Instagram Security Monitor' }));
    expect(screen.getByText('exact_match 0.7')).toBeVisible();
    rerender(<HistoryDepthPanel result={enhanced} />);
    expect(screen.getByText('1,229,999 bytes')).toBeVisible();
    expect(screen.getByText('1,229,999 characters')).toBeVisible();
    rerender(<UsageCostPanel result={enhanced} modelId="fixture-model-v1" />);
    expect(screen.getByText('436,800,000 ns')).toBeVisible();
  });

  it('renders full-roster and bounded prompt exposure without deriving values', () => {
    const { rerender } = render(<PromptExposurePanel result={baseline} />);
    expect(screen.getByText('Full-roster prompt')).toBeInTheDocument();
    expect(screen.getByText('18,420')).toBeInTheDocument();
    expect(screen.getByText('100 names exposed')).toBeInTheDocument();
    rerender(<PromptExposurePanel result={enhanced} />);
    expect(screen.getByText('interaction_agent_candidate_xml')).toBeInTheDocument();
    expect(screen.getByText('3 candidates exposed')).toBeInTheDocument();
    expect(screen.getByText('292943fa-641b-5ae6-a8d2-ab631be77ba8, 5f5723ea-68b5-513c-8562-b8d7145440e7, 68163898-5de0-5848-8cd9-7148270cb0ea')).toBeInTheDocument();
  });

  it('keeps candidate rank, score, components, and reasons in exact server array order', () => {
    render(<CandidateEvidenceTable result={enhanced} />);
    const rows = screen.getAllByRole('row').slice(1);
    expect(rows.map((row) => within(row).getByText(/.+/, { selector: 'strong' }).textContent)).toEqual(['Instagram Security Monitor', 'Controlled Fixture Researcher', 'Account Security Auditor']);
    expect(rows.map((row) => within(row).getAllByRole('cell')[0].textContent)).toEqual(['1', '2', '3']);
    expect(rows.map((row) => within(row).getAllByRole('cell')[3].textContent)).toEqual(['1', '0.1775', '0.10375']);
    fireEvent.click(within(rows[0]).getByRole('button', { name: 'Expand candidate Instagram Security Monitor' }));
    fireEvent.click(within(rows[2]).getByRole('button', { name: 'Expand candidate Account Security Auditor' }));
    expect(within(rows[0]).getByText('exact_match 0.7')).toBeInTheDocument();
    expect(within(rows[0]).getByText('shared multi-word phrase')).toBeInTheDocument();
    expect(within(rows[2]).getByText('lifecycle -0.03')).toBeInTheDocument();
    expect(within(rows[2]).getByText('dormant lifecycle penalty')).toBeInTheDocument();
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
    expect(screen.getAllByText('292943fa-641b-5ae6-a8d2-ab631be77ba8').length).toBeGreaterThan(0);
    expect(screen.getByText('accepted')).toBeInTheDocument();
    expect(screen.getByText('Reused')).toBeInTheDocument();
    expect(screen.getByText('100 → 100')).toBeInTheDocument();
    expect(screen.getByText('Unavailable · not emitted')).toBeInTheDocument();
  });

  it('shows sanitized Gmail facts and expands only controlled fabricated content', () => {
    render(<GmailEvidencePanel result={enhanced} track="controlled" />);
    expect(screen.getByText('Read-only allowed')).toBeInTheDocument();
    expect(screen.getByText('Read-only completed')).toBeInTheDocument();
    expect(screen.getByText('SEC-7419')).toBeInTheDocument();
    const content = 'Fabricated security notice SEC-7419. A sign-in was recorded at 2026-09-18 04:12 UTC from Lisbon on Pixel 10. Verification phrase: indigo-orbit.';
    expect(screen.queryByText(content)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Show fabricated evidence' }));
    expect(screen.getByRole('dialog', { name: 'Fabricated Gmail evidence' })).toBeInTheDocument();
    expect(screen.getByText(content)).toBeInTheDocument();
  });

  it('traps modal focus, closes with Escape, isolates the background, and restores the opener', () => {
    render(<main className="lab-shell"><GmailEvidencePanel result={enhanced} track="controlled" /></main>);
    const opener = screen.getByRole('button', { name: 'Show fabricated evidence' });
    fireEvent.click(opener);
    const dialog = screen.getByRole('dialog', { name: 'Fabricated Gmail evidence' });
    const close = screen.getByRole('button', { name: 'Close evidence' });
    expect(document.querySelector('main.lab-shell')).toHaveAttribute('inert');
    close.focus();
    fireEvent.keyDown(dialog, { key: 'Tab' });
    expect(close).toHaveFocus();
    fireEvent.keyDown(dialog, { key: 'Tab', shiftKey: true });
    expect(close).toHaveFocus();
    fireEvent.keyDown(dialog, { key: 'Escape' });
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(opener).toHaveFocus();
    expect(document.querySelector('main.lab-shell')).not.toHaveAttribute('inert');
  });

  it('selects the exact repetition scorecard and exposes every turn despite a failed peer side', () => {
    const multi: any = structuredClone(run);
    const originalPair = multi.pairs[0];
    const enhancedTurn2 = structuredClone(originalPair.outcomes[1].results[0]);
    enhancedTurn2.turn_id = '88888888-8888-4888-8888-888888888888';
    enhancedTurn2.final_response.value = 'Enhanced repetition three turn two.';
    multi.pairs = [1, 2, 3].map((repetition) => ({
      ...structuredClone(originalPair),
      scheduled: { ...originalPair.scheduled, pair_id: `${repetition}${repetition}${repetition}${repetition}${repetition}${repetition}${repetition}${repetition}-${repetition}${repetition}${repetition}${repetition}-4${repetition}${repetition}${repetition}-8${repetition}${repetition}${repetition}-${repetition}${repetition}${repetition}${repetition}${repetition}${repetition}${repetition}${repetition}${repetition}${repetition}${repetition}${repetition}`, repetition },
      outcomes: repetition === 3 ? [
        { ...structuredClone(originalPair.outcomes[0]), status: 'failure', reason: 'Baseline repetition three failed.', results: [] },
        { ...structuredClone(originalPair.outcomes[1]), results: [structuredClone(originalPair.outcomes[1].results[0]), enhancedTurn2] },
      ] : structuredClone(originalPair.outcomes),
    }));
    multi.scorecards = multi.pairs.flatMap((pair: any) => {
      const enhancedCard = structuredClone(run.scorecards[1]);
      enhancedCard.pair_id = pair.scheduled.pair_id;
      enhancedCard.repetition = pair.scheduled.repetition;
      enhancedCard.turns = [structuredClone(enhancedCard.turns[0]), structuredClone(enhancedCard.turns[0])];
      enhancedCard.turns[1].response.positive_evidence = [`Repetition ${pair.scheduled.repetition} turn two grade`];
      return pair.scheduled.repetition === 3 ? [enhancedCard] : [];
    });
    render(<ComparisonGrid run={multi} scenario={{ ...scenario, turn_count: 2 }} pairIndex={2} />);
    expect(screen.getByText('Baseline repetition three failed.')).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Turn 2' }));
    expect(screen.getByText('Enhanced repetition three turn two.')).toBeVisible();
    expect(screen.getByText('Repetition 3 turn two grade')).toBeVisible();
  });

  it('renders legacy unassociated scorecards as unavailable without guessing by system', () => {
    const legacy = structuredClone(backend.evidence_run);
    for (const scorecard of legacy.scorecards) {
      delete (scorecard as { pair_id?: string }).pair_id;
      delete (scorecard as { repetition?: number }).repetition;
    }
    const parsed = PairedRunResultSchema.parse(legacy);

    render(<ComparisonGrid run={parsed} scenario={scenario} pairIndex={0} />);

    const panels = screen.getAllByText('Layer scorecard').map((heading) => heading.closest('section')!);
    expect(panels).toHaveLength(2);
    for (const panel of panels) expect(within(panel).getByText('Unavailable')).toBeVisible();
    expect(screen.queryByRole('table', { name: /layer scorecard/i })).not.toBeInTheDocument();
  });

  it('does not expose a disclosure control for exploratory evidence', () => {
    render(<GmailEvidencePanel result={enhanced} track="exploratory" />);
    expect(screen.getByText('Exploratory evidence remains collapsed')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /fabricated evidence/i })).not.toBeInTheDocument();
  });

  it('renders the complete raw, rendered, omission, summary, preservation, and contamination history facts', () => {
    render(<HistoryDepthPanel result={enhanced} />);
    for (const value of ['1,229,999 bytes', '1,229,999 characters', '10,000 entries', '2,100 characters', '2,100 bytes', '8 episodes', '9,984 omitted', '0 truncated', 'Summary used']) {
      expect(screen.getByText(value)).toBeInTheDocument();
    }
    expect(within(screen.getByText('Preservation').parentElement!).getByText('Unavailable')).toBeInTheDocument();
    expect(within(screen.getByText('Contamination').parentElement!).getByText('Unavailable')).toBeInTheDocument();
  });

  it('renders elapsed time, tokens, model, explicit unavailable provider, and exact cost including zero cached tokens', () => {
    render(<UsageCostPanel result={enhanced} modelId="fixture-model-v1" />);
    expect(screen.getByText('fixture-model-v1')).toBeInTheDocument();
    expect(within(screen.getByText('Provider').parentElement!).getByText('Unavailable')).toBeInTheDocument();
    expect(screen.getByText('436,800,000 ns')).toBeInTheDocument();
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
    expect(screen.getByText(/reference: SEC-7419/)).toBeInTheDocument();
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
