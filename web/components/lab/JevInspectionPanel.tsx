'use client';

import { useState } from 'react';
import type { SystemRunResult } from '@/lib/lab/schema';
import { availabilityLabel, displayScalar, isRecord } from './ObservedValue';

export function JevInspectionPanel({ result }: { result: SystemRunResult }) {
  const [expandedReasoning, setExpandedReasoning] = useState<Record<number, boolean>>({});

  const isJev = result.system === 'enhanced_jev';
  const reduceObs = result.jev_reduce_decision;
  const mapObs = result.jev_map_scores;
  const shortlistObs = result.jev_shortlist;

  const availability = reduceObs?.availability ?? (isJev ? 'available' : 'not_applicable');
  const isAvailable = availability === 'available' || availability === 'inferred';

  const reduceValue = isRecord(reduceObs?.value) ? reduceObs.value : null;
  const mapScores = Array.isArray(mapObs?.value) ? mapObs.value.filter(isRecord) : [];
  const shortlist = Array.isArray(shortlistObs?.value) ? shortlistObs.value : [];

  const winnerMargin = result.jev_winner_margin?.value ?? (reduceValue?.winner_margin as number | undefined);
  const mapLatency = result.jev_map_latency_ms?.value;
  const reduceLatency = result.jev_reduce_latency_ms?.value;
  const totalLatency = result.jev_total_latency_ms?.value;
  const apiCalls = result.jev_api_calls_count?.value;
  const digest = result.jev_card_digest?.value ?? (reduceValue?.card_digest as string | undefined);
  const tokenUsage = isRecord(result.jev_token_usage?.value) ? result.jev_token_usage.value : null;

  return (
    <section
      className="evidence-panel jev-panel"
      data-testid="evidence-layer"
      aria-labelledby={`${result.turn_id}-jev`}
    >
      <div className="evidence-heading">
        <h4 id={`${result.turn_id}-jev`}>Jev Map/Reduce inspection</h4>
        <span>{availabilityLabel(availability)}</span>
      </div>

      {!isAvailable ? (
        <p className="empty-evidence">
          <strong>{availabilityLabel(availability)}</strong>
          {reduceObs?.reason ? (
            <> · {reduceObs.reason}</>
          ) : !isJev ? (
            <> · Not applicable for {result.system.replaceAll('_', ' ')}</>
          ) : null}
        </p>
      ) : (
        <div className="jev-content">
          {/* Reduce Decision */}
          <div className="jev-decision-block">
            <h5 className="jev-subheading">Reduce Routing Decision</h5>
            <dl className="routing-list">
              <div className="routing-field">
                <dt>Action</dt>
                <dd><strong>{reduceValue?.action ? displayScalar(reduceValue.action) : '—'}</strong></dd>
              </div>
              <div className="routing-field">
                <dt>Recommended Agent</dt>
                <dd className="mono-break">
                  {reduceValue?.recommended_agent_id ? displayScalar(reduceValue.recommended_agent_id) : 'None'}
                </dd>
              </div>
              <div className="routing-field">
                <dt>Confidence</dt>
                <dd>{reduceValue?.confidence !== undefined ? displayScalar(reduceValue.confidence) : '—'}</dd>
              </div>
              <div className="routing-field">
                <dt>Winner Margin</dt>
                <dd>{winnerMargin !== undefined && winnerMargin !== null ? displayScalar(winnerMargin) : '—'}</dd>
              </div>
            </dl>
            {reduceValue?.rationale && (
              <div className="routing-field" style={{ marginTop: '6px' }}>
                <dt>Rationale</dt>
                <dd>{displayScalar(reduceValue.rationale)}</dd>
              </div>
            )}
          </div>

          {/* Shortlist */}
          <div className="jev-shortlist-block" style={{ marginTop: '12px' }}>
            <h5 className="jev-subheading">Shortlist Candidates ({shortlist.length})</h5>
            {shortlist.length > 0 ? (
              <ul className="jev-shortlist-items" style={{ listStyle: 'none', padding: 0, margin: '4px 0 0' }}>
                {shortlist.map((item, idx) => {
                  const idStr = typeof item === 'string' ? item : isRecord(item) && item.agent_id ? displayScalar(item.agent_id) : displayScalar(item);
                  return (
                    <li key={idx} className="stacked mono-break" style={{ padding: '2px 0' }}>
                      <span className="selected-mark" style={{ display: 'inline', marginRight: '6px' }}>•</span>
                      {idStr}
                    </li>
                  );
                })}
              </ul>
            ) : (
              <p className="empty-evidence" style={{ margin: '4px 0 0' }}>No candidates shortlisted</p>
            )}
          </div>

          {/* Map Scores */}
          <div className="jev-map-block" style={{ marginTop: '12px' }}>
            <h5 className="jev-subheading">Map Card Scores ({mapScores.length})</h5>
            {mapScores.length > 0 ? (
              <div className="table-scroll" tabIndex={0} aria-label={`${result.system} Jev map scores table scroll area`}>
                <table>
                  <caption className="sr-only">Jev agent scores in source order</caption>
                  <thead>
                    <tr>
                      <th>Agent ID</th>
                      <th>Composite</th>
                      <th>Affinity</th>
                      <th>Continuity</th>
                      <th>Risk</th>
                      <th>Reasoning</th>
                    </tr>
                  </thead>
                  <tbody>
                    {mapScores.map((score, index) => {
                      const agentIdStr = displayScalar(score.agent_id);
                      const isExpanded = expandedReasoning[index] ?? false;
                      return (
                        <tr key={String(score.agent_id ?? index)}>
                          <td>
                            <span className="mono-break">{agentIdStr}</span>
                          </td>
                          <td><strong>{displayScalar(score.composite_score)}</strong></td>
                          <td>{displayScalar(score.affinity_score)}</td>
                          <td>{displayScalar(score.continuity_score)}</td>
                          <td>{displayScalar(score.risk_score)}</td>
                          <td>
                            {score.reasoning ? (
                              <button
                                className="candidate-toggle"
                                type="button"
                                aria-expanded={isExpanded}
                                aria-label={`${isExpanded ? 'Collapse' : 'Expand'} reasoning for ${agentIdStr}`}
                                onClick={() => setExpandedReasoning((curr) => ({ ...curr, [index]: !curr[index] }))}
                              >
                                {isExpanded ? displayScalar(score.reasoning) : (
                                  String(score.reasoning).length > 25
                                    ? `${String(score.reasoning).slice(0, 25)}…`
                                    : displayScalar(score.reasoning)
                                )}
                              </button>
                            ) : '—'}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="empty-evidence" style={{ margin: '4px 0 0' }}>
                {mapObs?.reason ? mapObs.reason : 'No map scores emitted.'}
              </p>
            )}
          </div>

          {/* Latency & Operations */}
          <div className="jev-latency-block" style={{ marginTop: '12px' }}>
            <h5 className="jev-subheading">Timing & Operations</h5>
            <dl className="metric-grid">
              <div className="metric-field">
                <dt>Map Latency</dt>
                <dd>{mapLatency !== undefined && mapLatency !== null ? `${displayScalar(mapLatency)} ms` : '—'}</dd>
              </div>
              <div className="metric-field">
                <dt>Reduce Latency</dt>
                <dd>{reduceLatency !== undefined && reduceLatency !== null ? `${displayScalar(reduceLatency)} ms` : '—'}</dd>
              </div>
              <div className="metric-field">
                <dt>Total Latency</dt>
                <dd><strong>{totalLatency !== undefined && totalLatency !== null ? `${displayScalar(totalLatency)} ms` : '—'}</strong></dd>
              </div>
              <div className="metric-field">
                <dt>API Calls</dt>
                <dd>{apiCalls !== undefined && apiCalls !== null ? displayScalar(apiCalls) : '—'}</dd>
              </div>
            </dl>
            {tokenUsage && (
              <dl className="metric-grid compact" style={{ marginTop: '8px' }}>
                <div className="metric-field">
                  <dt>Input Tokens</dt>
                  <dd>{displayScalar(tokenUsage.input_tokens)}</dd>
                </div>
                <div className="metric-field">
                  <dt>Output Tokens</dt>
                  <dd>{displayScalar(tokenUsage.output_tokens)}</dd>
                </div>
              </dl>
            )}
            {digest && (
              <div className="routing-field" style={{ marginTop: '6px' }}>
                <dt>Card Digest</dt>
                <dd className="mono-break"><small>{digest}</small></dd>
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
