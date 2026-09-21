import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it } from 'vitest';
import backend from '../../tests/fixtures/backend.json';
import { ScenarioListSchema } from '../../lib/lab/schema';
import { ScenarioRunner } from './ScenarioRunner';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const scenarios = ScenarioListSchema.parse(backend.scenarios).scenarios;
it('renders numerical scenario evidence with tabular numerals', () => {
  const style = document.createElement('style');
  style.textContent = readFileSync(resolve(process.cwd(), 'app/lab/lab.css'), 'utf8');
  document.head.appendChild(style);
  try {
    render(<main className="lab-shell"><ScenarioRunner scenarios={scenarios} runnable busy={false} onStart={() => {}} /></main>);
    const evidence = screen.getByText('3 repetitions').parentElement!;
    expect(getComputedStyle(evidence).fontVariantNumeric).toBe('tabular-nums');
  } finally { style.remove(); }
});
it.each([false, true])('uses server runnable=%s, and shows the selected server scenario', (runnable) => {
  render(<ScenarioRunner scenarios={scenarios} runnable={runnable} busy={false} onStart={() => {}} />);
  expect(screen.getByRole('button', { name: 'Run scenario' })).toHaveProperty('disabled', !runnable);
  expect(screen.getByText('3 repetitions')).toBeVisible();
  expect(screen.getByText('Controlled fixture')).toBeVisible();
});
it('submits the selected ID and disables selection during an active run', () => {
  const selected: string[] = [];
  const { rerender } = render(<ScenarioRunner scenarios={scenarios} runnable busy={false} onStart={(id) => selected.push(id)} />);
  fireEvent.change(screen.getByLabelText('Scenario'), { target: { value: scenarios[1].scenario_id } });
  fireEvent.click(screen.getByRole('button', { name: 'Run scenario' }));
  expect(selected).toEqual([scenarios[1].scenario_id]);
  rerender(<ScenarioRunner scenarios={scenarios} runnable busy onStart={() => {}} />);
  expect(screen.getByLabelText('Scenario')).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Run scenario' })).toBeDisabled();
});
