import type { PairedRunResult } from '@/lib/lab/schema';
import { availabilityLabel } from './ObservedValue';

type SequenceScorecard = PairedRunResult['scorecards'][number];
const statusLabels = { pass: 'Pass', fail: 'Fail', missing: 'Missing', contradictory: 'Contradictory', not_applicable: 'Not applicable' } as const;

export function LayerScorecard({ scorecard, turnIndex = 0 }: { scorecard: SequenceScorecard | undefined; turnIndex?: number }) {
  const turn = scorecard?.turns[turnIndex];
  const grades = turn ? [turn.routing, turn.response, turn.gmail_safety, turn.identity, turn.duplicate, turn.context] : [];
  const systemLabel = scorecard?.system === 'baseline' ? 'Baseline' : scorecard?.system === 'enhanced_deterministic' ? 'Enhanced Deterministic' : scorecard?.system === 'enhanced_jev' ? 'Enhanced Jev' : 'Enhanced';
  return <section className="evidence-panel scorecard-panel" data-testid="evidence-layer" aria-labelledby={`${scorecard?.system ?? 'unknown'}-scorecard-${turnIndex}`}>
    <div className="evidence-heading"><h4 id={`${scorecard?.system ?? 'unknown'}-scorecard-${turnIndex}`}>Layer scorecard</h4><span>{scorecard ? scorecard.passed ? '✓ Sequence pass' : '× Sequence fail' : 'Unavailable'}</span></div>
    {turn ? <div className="table-scroll" tabIndex={0} aria-label={`${scorecard.system} scorecard scroll area`}><table aria-label={`${systemLabel} layer scorecard`}><thead><tr><th>Layer</th><th>Result</th><th>Expected evidence</th><th>Observed gaps or contradictions</th></tr></thead><tbody>
      {grades.map((grade) => <tr key={grade.layer}><td>{grade.layer.replaceAll('_', ' ')}</td><td><span className={`grade grade-${grade.status}`}>{grade.status === 'pass' ? '✓' : grade.status === 'not_applicable' ? '—' : '×'} {statusLabels[grade.status]}</span></td><td>{grade.positive_evidence.length ? grade.positive_evidence.map((item, index) => <span className="stacked" key={index}>{item}</span>) : 'None emitted'}</td><td>{[...grade.negative_evidence, ...grade.missing_evidence, ...grade.contradictory_evidence].map((item, index) => <span className="stacked" key={index}>{item}</span>)}{grade.negative_evidence.length + grade.missing_evidence.length + grade.contradictory_evidence.length === 0 && 'None'}</td></tr>)}
      <tr><td>candidate rank</td><td colSpan={3}>{availabilityLabel(turn.candidate_rank.availability)}{turn.candidate_rank.value !== null && ` · ${turn.candidate_rank.value}`}{turn.candidate_rank.reason && ` · ${turn.candidate_rank.reason}`}</td></tr>
    </tbody></table></div> : <p className="empty-evidence">No layer grade emitted for this side.</p>}
  </section>;
}
