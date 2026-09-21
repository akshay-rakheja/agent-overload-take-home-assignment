'use client';

import { useState } from 'react';
import type { Scenario } from '@/lib/lab/schema';

export function ScenarioRunner({ scenarios, runnable, busy, onStart }: {
  scenarios: Scenario[]; runnable: boolean; busy: boolean; onStart: (scenarioId: string) => void;
}) {
  const [selectedId, setSelectedId] = useState('');
  const selected = scenarios.find((scenario) => scenario.scenario_id === selectedId) ?? scenarios[0];
  return <section className="lab-runner" aria-labelledby="runner-title">
    <div><h2 id="runner-title">Choose the comparison</h2><p>The server controls repetitions and the order of each paired run.</p></div>
    <form onSubmit={(event) => { event.preventDefault(); if (runnable && !busy && selected) onStart(selected.scenario_id); }}>
      <label htmlFor="lab-scenario">Scenario</label>
      <div className="lab-run-controls">
        <select id="lab-scenario" value={selected?.scenario_id ?? ''} disabled={busy || !selected} onChange={(event) => setSelectedId(event.target.value)}>
          {!selected && <option value="">Scenarios unavailable</option>}
          {scenarios.map((scenario) => <option key={scenario.scenario_id} value={scenario.scenario_id}>{scenario.title}</option>)}
        </select>
        <button type="submit" disabled={!runnable || busy || !selected}>Run scenario</button>
      </div>
      {selected && <div className="lab-scenario-details">
        <span>{selected.track === 'controlled' ? 'Controlled fixture' : 'Exploratory track'}</span><span>{selected.repetitions} repetitions</span><span>{selected.turn_count} {selected.turn_count === 1 ? 'turn' : 'turns'}</span>
        <span>{selected.reset_profile.roster_size} agents</span><span>{selected.reset_profile.history_entries} history entries</span>
        {selected.optional && <span>Optional</span>}{selected.budget_guarded && <span>Budget guarded</span>}
      </div>}
    </form>
  </section>;
}
