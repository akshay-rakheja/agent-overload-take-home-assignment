import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { AgentInspectorPanel, InspectorData } from './AgentInspectorPanel';

const mockInspectorData: InspectorData = {
  ok: true,
  system: 'enhanced',
  roster_count: 2,
  roster: [
    {
      agent_id: 'agent-1',
      name: 'Email Security Scanner',
      purpose: 'Scan security emails',
      status: 'hot',
      use_count: 2,
      created_at: '2026-09-22T10:00:00Z',
      last_used_at: '2026-09-22T11:00:00Z',
    },
    {
      agent_id: 'agent-2',
      name: 'Calendar Manager',
      purpose: 'Manage schedule and meetings',
      status: 'hot',
      use_count: 1,
      created_at: '2026-09-22T10:00:00Z',
      last_used_at: '2026-09-22T11:00:00Z',
    },
  ],
  candidate_ids: ['agent-1'],
  candidates: [
    {
      rank: 1,
      agent_id: 'agent-1',
      name: 'Email Security Scanner',
      purpose: 'Scan security emails',
      status: 'hot',
      score: 0.94,
      reasons: ['Exact purpose match'],
    },
  ],
  latest_turn: {
    timestamp: '2026-09-22T11:00:00Z',
    routing_action: 'reuse',
    recommended_action: 'reuse',
    recommended_agent_id: 'agent-1',
    selected_agent_id: 'agent-1',
    selected_agent_name: 'Email Security Scanner',
    instructions: 'Search for security notices',
  },
};

describe('AgentInspectorPanel', () => {
  it('renders system title, roster counts, and reuse status', () => {
    render(
      <AgentInspectorPanel
        system="enhanced"
        title="Enhanced Inspector (Port 8002)"
        data={mockInspectorData}
        isLoading={false}
      />,
    );

    expect(screen.getByText('Enhanced Inspector (Port 8002)')).toBeInTheDocument();
    expect(screen.getByText('Total Agents:')).toBeInTheDocument();
    expect(screen.getByText('🟢 REUSED AGENT')).toBeInTheDocument();
    expect(screen.getAllByText('Email Security Scanner').length).toBeGreaterThan(0);
    expect(screen.getByText('Score: 0.94')).toBeInTheDocument();
    expect(screen.getByText('SELECTED')).toBeInTheDocument();
  });

  it('renders create_new state correctly', () => {
    const createData: InspectorData = {
      ...mockInspectorData,
      latest_turn: {
        timestamp: '2026-09-22T11:00:00Z',
        routing_action: 'create_new',
        recommended_action: 'create_new',
        recommended_agent_id: null,
        selected_agent_id: null,
        selected_agent_name: 'New Custom Worker',
        instructions: 'Do fresh task',
      },
    };

    render(
      <AgentInspectorPanel
        system="enhanced"
        title="Enhanced Inspector"
        data={createData}
        isLoading={false}
      />,
    );

    expect(screen.getByText('🔵 CREATED NEW')).toBeInTheDocument();
    expect(screen.getByText('New Custom Worker')).toBeInTheDocument();
  });

  it('renders abstain state correctly', () => {
    const abstainData: InspectorData = {
      ...mockInspectorData,
      latest_turn: {
        timestamp: '2026-09-22T11:00:00Z',
        routing_action: 'abstain',
        recommended_action: 'abstain',
        recommended_agent_id: null,
        selected_agent_id: null,
        selected_agent_name: null,
        instructions: null,
      },
    };

    render(
      <AgentInspectorPanel
        system="baseline"
        title="Baseline Inspector"
        data={abstainData}
        isLoading={false}
      />,
    );

    expect(screen.getByText('⚪ ABSTAINED')).toBeInTheDocument();
    expect(screen.getByText('No execution agent dispatched')).toBeInTheDocument();
  });

  it('toggles roster expansion', () => {
    render(
      <AgentInspectorPanel
        system="enhanced"
        title="Enhanced Inspector"
        data={mockInspectorData}
        isLoading={false}
      />,
    );

    const toggleButton = screen.getByText(/▼ Show all/);
    fireEvent.click(toggleButton);
    expect(screen.getByText(/▲ Hide/)).toBeInTheDocument();
    expect(screen.getByText('Calendar Manager')).toBeInTheDocument();
  });
});
