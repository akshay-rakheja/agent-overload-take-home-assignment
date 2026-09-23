import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import backend from '../../tests/fixtures/backend.json';
import {
  PairedRunResultSchema,
  ScenarioSchema,
  SystemRunResultSchema,
  type SystemRunResult,
} from '@/lib/lab/schema';
import { JevInspectionPanel } from './JevInspectionPanel';
import { ComparisonGrid } from './ComparisonGrid';

const baseRun = PairedRunResultSchema.parse(backend.evidence_run);
const scenario = ScenarioSchema.parse(backend.scenarios.scenarios[0]);

const jevResult: SystemRunResult = SystemRunResultSchema.parse({
  ...baseRun.pairs[0].outcomes[1].results[0],
  system: 'enhanced_jev',
  jev_map_scores: {
    availability: 'available',
    value: [
      {
        agent_id: '292943fa-641b-5ae6-a8d2-ab631be77ba8',
        composite_score: 0.95,
        affinity_score: 0.95,
        continuity_score: 1.0,
        risk_score: 0.0,
        reasoning: 'exact match for Instagram security operations',
        card_digest: 'card-digest-001',
        latency_ms: 12.0,
      },
    ],
    reason: null,
  },
  jev_shortlist: {
    availability: 'available',
    value: ['292943fa-641b-5ae6-a8d2-ab631be77ba8'],
    reason: null,
  },
  jev_reduce_decision: {
    availability: 'available',
    value: {
      action: 'reuse',
      recommended_agent_id: '292943fa-641b-5ae6-a8d2-ab631be77ba8',
      confidence: 0.95,
      winner_margin: 0.35,
      rationale: 'Jev map-reduce selected Instagram Security Monitor by clear margin',
    },
    reason: null,
  },
  jev_winner_margin: {
    availability: 'available',
    value: 0.35,
    reason: null,
  },
  jev_map_latency_ms: {
    availability: 'available',
    value: 32.5,
    reason: null,
  },
  jev_reduce_latency_ms: {
    availability: 'available',
    value: 18.2,
    reason: null,
  },
  jev_total_latency_ms: {
    availability: 'available',
    value: 50.7,
    reason: null,
  },
  jev_api_calls_count: {
    availability: 'available',
    value: 3,
    reason: null,
  },
  jev_token_usage: {
    availability: 'available',
    value: {
      input_tokens: 350,
      output_tokens: 65,
    },
    reason: null,
  },
  jev_card_digest: {
    availability: 'available',
    value: 'card-digest-001',
    reason: null,
  },
});

describe('JevInspectionPanel', () => {
  it('renders complete Map/Reduce routing evidence, shortlist, and timing metrics', () => {
    render(<JevInspectionPanel result={jevResult} />);

    expect(screen.getByRole('heading', { name: 'Jev Map/Reduce inspection' })).toBeInTheDocument();
    expect(screen.getByText('Reduce Routing Decision')).toBeInTheDocument();
    expect(screen.getByText('reuse')).toBeInTheDocument();
    expect(screen.getByText('Jev map-reduce selected Instagram Security Monitor by clear margin')).toBeInTheDocument();
    expect(screen.getByText('Shortlist Candidates (1)')).toBeInTheDocument();
    expect(screen.getByText('Map Card Scores (1)')).toBeInTheDocument();

    const table = screen.getByRole('table', { name: 'Jev agent scores in source order' });
    expect(table).toBeInTheDocument();
    expect(within(table).getAllByText('0.95')).toHaveLength(2);
    expect(within(table).getByText('1')).toBeInTheDocument();

    const toggle = screen.getByRole('button', { name: /Expand reasoning for/ });
    expect(toggle).toBeInTheDocument();
    fireEvent.click(toggle);
    expect(screen.getByText('exact match for Instagram security operations')).toBeInTheDocument();

    expect(screen.getByText('Timing & Operations')).toBeInTheDocument();
    expect(screen.getByText('32.5 ms')).toBeInTheDocument();
    expect(screen.getByText('18.2 ms')).toBeInTheDocument();
    expect(screen.getByText('50.7 ms')).toBeInTheDocument();
    expect(screen.getByText('3')).toBeInTheDocument();
    expect(screen.getByText('350')).toBeInTheDocument();
    expect(screen.getByText('65')).toBeInTheDocument();
    expect(screen.getByText('card-digest-001')).toBeInTheDocument();
  });

  it('renders unavailable placeholder when Jev metrics were not emitted', () => {
    const unavResult: SystemRunResult = SystemRunResultSchema.parse({
      ...baseRun.pairs[0].outcomes[0].results[0],
      system: 'baseline',
      jev_reduce_decision: {
        availability: 'unavailable',
        value: null,
        reason: 'not emitted',
      },
    });

    render(<JevInspectionPanel result={unavResult} />);
    expect(screen.getByRole('heading', { name: 'Jev Map/Reduce inspection' })).toBeInTheDocument();
    expect(screen.getAllByText('Unavailable').length).toBeGreaterThan(0);
    expect(screen.getByText(/not emitted/)).toBeInTheDocument();
  });
});

describe('Three-Way ComparisonGrid', () => {
  it('renders 3 columns with Protocol Spine and Jev inspection in three-way mode', () => {
    const baselineOutcome = {
      ...baseRun.pairs[0].outcomes[0],
      system: 'baseline' as const,
    };
    const deterministicOutcome = {
      ...baseRun.pairs[0].outcomes[1],
      system: 'enhanced_deterministic' as const,
      results: [
        SystemRunResultSchema.parse({
          ...baseRun.pairs[0].outcomes[1].results[0],
          system: 'enhanced_deterministic',
        }),
      ],
    };
    const jevOutcome = {
      ...baseRun.pairs[0].outcomes[1],
      system: 'enhanced_jev' as const,
      results: [jevResult],
    };

    const threeWayRun = PairedRunResultSchema.parse({
      ...baseRun,
      schema_version: 2,
      schedule: [
        {
          pair_id: baseRun.pairs[0].scheduled.pair_id,
          scenario_id: scenario.scenario_id,
          repetition: 1,
          order: 'baseline_then_deterministic_then_jev',
        },
      ],
      pairs: [
        {
          scheduled: {
            pair_id: baseRun.pairs[0].scheduled.pair_id,
            scenario_id: scenario.scenario_id,
            repetition: 1,
            order: 'baseline_then_deterministic_then_jev',
          },
          verifications: [],
          outcomes: [baselineOutcome, deterministicOutcome, jevOutcome],
        },
      ],
      scorecards: [
        { ...baseRun.scorecards[0], system: 'baseline' as const },
        { ...baseRun.scorecards[1], system: 'enhanced_deterministic' as const },
        { ...baseRun.scorecards[1], system: 'enhanced_jev' as const },
      ],
    });

    render(<ComparisonGrid run={threeWayRun} scenario={scenario} pairIndex={0} />);

    // Check headings
    expect(screen.getByRole('heading', { name: 'Baseline evidence' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Enhanced Deterministic evidence' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Enhanced Jev evidence' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Protocol spine' })).toBeInTheDocument();

    // Check 3-way protocol spine steps
    expect(screen.getByText('Deterministic')).toBeInTheDocument();
    expect(screen.getByText('Jev')).toBeInTheDocument();

    // Check that 33 evidence layers are rendered (11 per column)
    expect(screen.getAllByTestId('evidence-layer')).toHaveLength(33);

    // Jev inspection panel should be rendered in the Jev column
    expect(screen.getByText('Reduce Routing Decision')).toBeInTheDocument();
    expect(screen.getByText('50.7 ms')).toBeInTheDocument();
  });
});
