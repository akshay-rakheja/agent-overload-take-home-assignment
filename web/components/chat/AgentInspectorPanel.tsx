import clsx from 'clsx';
import { useState } from 'react';

export interface InspectorAgent {
  agent_id: string;
  name: string;
  purpose: string;
  status: string;
  use_count: number;
  created_at: string | null;
  last_used_at: string | null;
}

export interface InspectorCandidate {
  rank: number;
  agent_id: string;
  name: string;
  purpose: string;
  status: string;
  score: number | null;
  reasons: string[];
}

export interface LatestTurn {
  timestamp: string;
  routing_action: 'reuse' | 'create_new' | 'abstain' | 'pending' | string;
  recommended_action: string | null;
  recommended_agent_id: string | null;
  selected_agent_id: string | null;
  selected_agent_name: string | null;
  instructions: string | null;
  user_message?: string;
  response?: string;
}

export interface JevScoreDetail {
  agent_id: string;
  composite_score: number;
  affinity_score: number;
  continuity_score: number;
  risk_score: number;
  reasoning: string;
}

export interface JevDetails {
  action: string | null;
  confidence: number | null;
  winner_margin: number | null;
  rationale: string | null;
  map_latency_ms: number | null;
  reduce_latency_ms: number | null;
  total_latency_ms: number | null;
  api_calls_count?: number;
  total_tokens?: number;
  map_scores: JevScoreDetail[];
  shortlist: string[];
}

export interface InspectorData {
  ok: boolean;
  system: 'baseline' | 'enhanced' | 'enhanced_deterministic' | 'enhanced_jev' | string;
  roster_count: number;
  roster: InspectorAgent[];
  candidate_ids: string[];
  candidates: InspectorCandidate[];
  latest_turn: LatestTurn | null;
  jev_details?: JevDetails | null;
}

interface AgentInspectorPanelProps {
  system: 'baseline' | 'enhanced' | 'enhanced_deterministic' | 'enhanced_jev' | string;
  title: string;
  data: InspectorData | null;
  isLoading: boolean;
  error?: string | null;
}

export function AgentInspectorPanel({
  system,
  title,
  data,
  isLoading,
  error,
}: AgentInspectorPanelProps) {
  const [showAllRoster, setShowAllRoster] = useState(false);

  const isJev = system === 'enhanced_jev' || system === 'jev';
  const isEnhanced = system === 'enhanced' || system === 'enhanced_deterministic' || isJev;
  const latestTurn = data?.latest_turn;
  const candidates = data?.candidates || [];
  const roster = data?.roster || [];
  const selectedAgentId = latestTurn?.selected_agent_id;
  const selectedAgentName = latestTurn?.selected_agent_name;
  const jevDetails = data?.jev_details;

  const resolvedAgentName =
    selectedAgentName ||
    roster.find((r) => r.agent_id === selectedAgentId)?.name ||
    candidates.find((c) => c.agent_id === selectedAgentId)?.name ||
    (latestTurn?.routing_action === 'create_new' && roster.length > 0
      ? roster[roster.length - 1].name
      : null) ||
    (latestTurn?.routing_action === 'reuse' && latestTurn.recommended_agent_id
      ? roster.find((r) => r.agent_id === latestTurn.recommended_agent_id)?.name ||
        candidates.find((c) => c.agent_id === latestTurn.recommended_agent_id)?.name
      : null);

  return (
    <div
      className={clsx(
        'flex flex-col rounded-xl border p-4 shadow-sm transition-all',
        isJev
          ? 'border-purple-200 bg-purple-50/20'
          : isEnhanced
            ? 'border-emerald-200 bg-emerald-50/20'
            : 'border-blue-200 bg-blue-50/20',
      )}
      data-testid={`inspector-panel-${system}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between border-b pb-3">
        <div className="flex items-center space-x-2">
          <span
            className={clsx(
              'h-2.5 w-2.5 rounded-full',
              isJev ? 'bg-purple-600' : isEnhanced ? 'bg-emerald-500' : 'bg-blue-500',
            )}
          />
          <h3 className="text-sm font-semibold text-gray-800">{title}</h3>
        </div>
        <div className="flex items-center space-x-2 text-xs">
          <span className="rounded-full bg-gray-100 px-2.5 py-0.5 font-medium text-gray-600">
            Total Agents: <strong className="text-gray-900">{roster.length}</strong>
          </span>
          <span
            className={clsx(
              'rounded-full px-2.5 py-0.5 font-medium',
              isJev
                ? 'bg-purple-100 text-purple-800'
                : isEnhanced
                  ? 'bg-emerald-100 text-emerald-800'
                  : 'bg-amber-100 text-amber-800',
            )}
          >
            {isJev
              ? `Candidates: ${candidates.length} (Jev Shortlist)`
              : isEnhanced
                ? `Candidates: ${candidates.length} (Bounded)`
                : `Exposure: ${candidates.length || roster.length} (Full Roster)`}
          </span>
        </div>
      </div>

      {error && (
        <div className="mt-3 rounded-lg bg-red-50 p-2.5 text-xs text-red-700">
          Inspector error: {error}
        </div>
      )}

      {/* Jev Telemetry Banner if available */}
      {isJev && jevDetails && (
        <div className="mt-3 flex flex-wrap gap-1.5 rounded-lg border border-purple-200/80 bg-purple-100/50 p-2 text-[11px] text-purple-900">
          {jevDetails.confidence !== null && (
            <span className="rounded bg-purple-200/80 px-1.5 py-0.5 font-semibold">
              Confidence: {(jevDetails.confidence * 100).toFixed(0)}%
            </span>
          )}
          {jevDetails.winner_margin !== null && (
            <span className="rounded bg-purple-200/80 px-1.5 py-0.5 font-semibold">
              Winner Margin: +{(jevDetails.winner_margin).toFixed(2)}
            </span>
          )}
          {jevDetails.total_latency_ms !== null && (
            <span className="rounded bg-purple-200/80 px-1.5 py-0.5 font-medium">
              Latency: {jevDetails.total_latency_ms}ms (Map: {jevDetails.map_latency_ms}ms · Reduce: {jevDetails.reduce_latency_ms}ms)
            </span>
          )}
          {jevDetails.shortlist && jevDetails.shortlist.length > 0 && (
            <span className="rounded bg-purple-200/80 px-1.5 py-0.5 font-medium">
              Shortlist: {jevDetails.shortlist.length}
            </span>
          )}
        </div>
      )}

      {/* Latest Turn Action Banner */}
      <div className="mt-3">
        <div className="text-xs font-semibold uppercase tracking-wider text-gray-500">
          Latest Routing Action
        </div>
        <div className="mt-1">
          {latestTurn ? (
            <div
              className={clsx(
                'rounded-lg border p-3 transition-colors',
                latestTurn.routing_action === 'reuse' && (isJev ? 'border-purple-300 bg-purple-100/60' : 'border-emerald-300 bg-emerald-100/60'),
                latestTurn.routing_action === 'create_new' && 'border-sky-300 bg-sky-100/60',
                latestTurn.routing_action === 'abstain' && 'border-gray-200 bg-gray-50',
                latestTurn.routing_action === 'pending' && 'border-amber-200 bg-amber-50',
              )}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  {latestTurn.routing_action === 'reuse' && (
                    <span
                      className={clsx(
                        'inline-flex items-center rounded-md px-2 py-0.5 text-xs font-bold text-white shadow-sm',
                        isJev ? 'bg-purple-700' : 'bg-emerald-600',
                      )}
                    >
                      🟢 REUSED AGENT
                    </span>
                  )}
                  {latestTurn.routing_action === 'create_new' && (
                    <span className="inline-flex items-center rounded-md bg-sky-600 px-2 py-0.5 text-xs font-bold text-white shadow-sm">
                      🔵 CREATED NEW
                    </span>
                  )}
                  {latestTurn.routing_action === 'abstain' && (
                    <span className="inline-flex items-center rounded-md bg-gray-500 px-2 py-0.5 text-xs font-bold text-white shadow-sm">
                      ⚪ ABSTAINED
                    </span>
                  )}
                  {latestTurn.routing_action === 'pending' && (
                    <span className="inline-flex items-center rounded-md bg-amber-500 px-2 py-0.5 text-xs font-bold text-white shadow-sm">
                      🟡 DISPATCHING...
                    </span>
                  )}
                  <span className="font-semibold text-sm">
                    {resolvedAgentName ||
                      (selectedAgentId && `Agent ${selectedAgentId.slice(0, 8)}...`) ||
                      (latestTurn.routing_action === 'abstain'
                        ? 'No execution agent dispatched'
                        : 'Interaction Agent')}
                  </span>
                </div>
                {latestTurn.timestamp && (
                  <span className="text-[11px] text-gray-500">
                    {new Date(latestTurn.timestamp).toLocaleTimeString()}
                  </span>
                )}
              </div>

              {latestTurn.instructions && (
                <div className="mt-2 text-xs text-gray-700 bg-white/70 rounded p-1.5 border border-black/5">
                  <strong>Instructions:</strong> {latestTurn.instructions}
                </div>
              )}

              {latestTurn.routing_action === 'abstain' && (
                <p className="mt-1 text-xs text-gray-600">
                  The interaction agent answered directly using conversation context without invoking an execution agent.
                </p>
              )}
            </div>
          ) : (
            <div className="rounded-lg border border-dashed border-gray-200 bg-white p-3 text-center text-xs text-gray-400">
              {isLoading ? 'Loading inspector data...' : 'Awaiting prompt to observe routing decisions'}
            </div>
          )}
        </div>
      </div>

      {/* Candidate Set Section */}
      <div className="mt-4">
        <div className="flex items-center justify-between">
          <div className="text-xs font-semibold uppercase tracking-wider text-gray-500">
            {isEnhanced ? 'Candidates Evaluated by Interaction Agent' : 'Exposed Agents (Full Roster)'}
          </div>
          <span className="text-[11px] text-gray-400">
            {isEnhanced ? 'Top-k Bounded Working Set' : 'Unconstrained Prompt Exposure'}
          </span>
        </div>

        <div className="mt-1.5 space-y-2 max-h-48 overflow-y-auto pr-1">
          {candidates.length > 0 ? (
            candidates.map((cand) => {
              const isSelected =
                (selectedAgentId && cand.agent_id === selectedAgentId) ||
                (selectedAgentName && cand.name.toLowerCase() === selectedAgentName.toLowerCase());

              return (
                <div
                  key={cand.agent_id || cand.name}
                  className={clsx(
                    'rounded-lg border p-2 text-xs transition-all',
                    isSelected
                      ? 'border-emerald-500 bg-emerald-50/80 shadow-sm ring-1 ring-emerald-500'
                      : 'border-gray-200 bg-white hover:border-gray-300',
                  )}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-1.5">
                      <span className="flex h-4 w-4 items-center justify-center rounded-full bg-gray-100 text-[10px] font-bold text-gray-600">
                        {cand.rank}
                      </span>
                      <strong className="text-gray-900 font-semibold">{cand.name}</strong>
                      {isSelected && (
                        <span className="rounded bg-emerald-600 px-1.5 py-0.2 text-[10px] font-bold text-white">
                          SELECTED
                        </span>
                      )}
                    </div>
                    {typeof cand.score === 'number' && (
                      <span className="rounded bg-gray-100 px-1.5 py-0.5 text-[10px] text-gray-600">
                        Score: {cand.score.toFixed(2)}
                      </span>
                    )}
                  </div>
                  {cand.purpose && (
                    <p className="mt-1 text-[11px] text-gray-600 line-clamp-2">
                      {cand.purpose}
                    </p>
                  )}
                  {cand.reasons && cand.reasons.length > 0 && (
                    <div className="mt-1 text-[10px] text-gray-400">
                      Hints: {cand.reasons.join('; ')}
                    </div>
                  )}
                </div>
              );
            })
          ) : (
            <div className="rounded border border-dashed border-gray-200 bg-white/50 p-2.5 text-center text-xs text-gray-400">
              No candidate set evaluated yet
            </div>
          )}
        </div>
      </div>

      {/* Full Roster Dropdown / Summary */}
      <div className="mt-3 border-t pt-2.5">
        <button
          type="button"
          onClick={() => setShowAllRoster(!showAllRoster)}
          className="flex w-full items-center justify-between text-xs text-gray-600 hover:text-gray-900"
        >
          <span className="font-medium">
            Execution Agent Roster ({roster.length} registered)
          </span>
          <span className="text-xs font-bold text-gray-400">
            {showAllRoster ? '▲ Hide' : '▼ Show all'}
          </span>
        </button>

        {showAllRoster && (
          <div className="mt-2 space-y-1.5 max-h-40 overflow-y-auto pr-1">
            {roster.length > 0 ? (
              roster.map((agent) => {
                const isSelected =
                  (selectedAgentId && agent.agent_id === selectedAgentId) ||
                  (selectedAgentName && agent.name.toLowerCase() === selectedAgentName.toLowerCase());

                return (
                  <div
                    key={agent.agent_id}
                    className={clsx(
                      'flex items-center justify-between rounded border bg-white p-1.5 text-xs',
                      isSelected ? 'border-emerald-400 bg-emerald-50/50' : 'border-gray-200',
                    )}
                  >
                    <div>
                      <div className="font-semibold text-gray-800">{agent.name}</div>
                      <div className="text-[10px] text-gray-500 line-clamp-1">
                        {agent.purpose}
                      </div>
                    </div>
                    <div className="text-right">
                      <span className="rounded bg-gray-100 px-1.5 py-0.5 text-[10px] font-bold text-gray-700">
                        Uses: {agent.use_count}
                      </span>
                    </div>
                  </div>
                );
              })
            ) : (
              <div className="text-center text-xs text-gray-400 py-1">
                Roster is empty
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
