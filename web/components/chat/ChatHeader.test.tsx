import { render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { ChatHeader } from './ChatHeader';

it('provides non-destructive navigation from chat to the Evaluation Lab', () => {
  render(<ChatHeader onOpenSettings={vi.fn()} onClearHistory={vi.fn()} />);
  expect(screen.getByRole('link', { name: 'Evaluation Lab' })).toHaveAttribute('href', '/lab');
});
