import { expect, it } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { PreflightPanel } from './PreflightPanel';
import { LabPreflightSchema } from '../../lib/lab/schema';
import { preflight } from '../../tests/fixtures/preflight';

it('renders readiness and the server evidence without making a local gate decision', () => {
  render(<PreflightPanel preflight={LabPreflightSchema.parse(preflight)} />);
  expect(screen.getByText('Ready to run')).toBeVisible();
  expect(screen.getByText('Controlled fabricated fixtures only.')).toBeVisible();
  expect(screen.getByText('fixture-baseline')).toBeVisible();
  expect(screen.getByText('fixture-enhanced')).toBeVisible();
  expect(screen.getByText('$1.25 remaining')).toBeVisible();
});

it.each(['Backend down', 'Revision mismatch', 'Fixture mismatch', 'Gmail unsafe', 'Model mismatch', 'Budget exhausted'])('renders the server blocker: %s', (blocker) => {
  render(<PreflightPanel preflight={LabPreflightSchema.parse({ ...preflight, runnable: false, blockers: [blocker, 'Second blocker'] })} />);
  expect(screen.getByText('Run blocked')).toBeVisible();
  expect(screen.getByText(blocker)).toBeVisible();
  expect(screen.getByText('Second blocker')).toBeVisible();
});

it('offers a keyboard-operable Gmail Connect handoff when the server reports disconnected', () => {
  let connected = 0;
  render(<PreflightPanel preflight={LabPreflightSchema.parse({ ...preflight, runnable: false, gmail_safety: { connected: false, read_only: false, reason: 'Connect the fixture mailbox.' } })} onConnect={() => { connected += 1; }} />);
  const button = screen.getByRole('button', { name: 'Connect Gmail' });
  fireEvent.click(button);
  expect(connected).toBe(1);
});
