'use client';

import { useCallback, useEffect, useState } from 'react';
import SettingsModal, { useSettings } from '@/components/SettingsModal';
import { ChatHeader } from '@/components/chat/ChatHeader';
import { ChatInput } from '@/components/chat/ChatInput';
import { ChatMessages } from '@/components/chat/ChatMessages';
import { ErrorBanner } from '@/components/chat/ErrorBanner';
import { useAutoScroll } from '@/components/chat/useAutoScroll';
import { AgentInspectorPanel, type InspectorData } from '@/components/chat/AgentInspectorPanel';
import type { ChatBubble } from '@/components/chat/types';

const POLL_INTERVAL_MS = 1500;

const formatEscapeCharacters = (text: string): string => {
  return text
    .replace(/\\n/g, '\n')
    .replace(/\\t/g, '\t')
    .replace(/\\r/g, '\r')
    .replace(/\\\\/g, '\\');
};

const isRenderableMessage = (entry: any) =>
  typeof entry?.role === 'string' &&
  typeof entry?.content === 'string' &&
  entry.content.trim().length > 0;

const toBubbles = (payload: any, prefix = 'history'): ChatBubble[] => {
  if (!Array.isArray(payload?.messages)) return [];

  return payload.messages
    .filter(isRenderableMessage)
    .map((message: any, index: number) => ({
      id: `${prefix}-${index}`,
      role: message.role,
      text: formatEscapeCharacters(message.content),
    }));
};

export default function Page() {
  const { settings, setSettings } = useSettings();
  const [openSettingsModal, setOpenSettingsModal] = useState(false);
  const [input, setInput] = useState('');
  const [error, setError] = useState<string | null>(null);

  // 1. Baseline state (:8001)
  const [baselineMessages, setBaselineMessages] = useState<ChatBubble[]>([]);
  const [baselineWaiting, setBaselineWaiting] = useState(false);
  const [baselineInspector, setBaselineInspector] = useState<InspectorData | null>(null);
  const [baselineInspectorLoading, setBaselineInspectorLoading] = useState(false);
  const { scrollContainerRef: baselineScrollRef, handleScroll: handleBaselineScroll } =
    useAutoScroll({
      items: baselineMessages,
      isWaiting: baselineWaiting,
    });

  // 2. Enhanced Deterministic state (:8002)
  const [enhancedMessages, setEnhancedMessages] = useState<ChatBubble[]>([]);
  const [enhancedWaiting, setEnhancedWaiting] = useState(false);
  const [enhancedInspector, setEnhancedInspector] = useState<InspectorData | null>(null);
  const [enhancedInspectorLoading, setEnhancedInspectorLoading] = useState(false);
  const { scrollContainerRef: enhancedScrollRef, handleScroll: handleEnhancedScroll } =
    useAutoScroll({
      items: enhancedMessages,
      isWaiting: enhancedWaiting,
    });

  // 3. Enhanced Jev state (:8002)
  const [jevMessages, setJevMessages] = useState<ChatBubble[]>([]);
  const [jevWaiting, setJevWaiting] = useState(false);
  const [jevInspector, setJevInspector] = useState<InspectorData | null>(null);
  const [jevInspectorLoading, setJevInspectorLoading] = useState(false);
  const { scrollContainerRef: jevScrollRef, handleScroll: handleJevScroll } =
    useAutoScroll({
      items: jevMessages,
      isWaiting: jevWaiting,
    });

  const openSettings = useCallback(() => setOpenSettingsModal(true), []);
  const closeSettings = useCallback(() => setOpenSettingsModal(false), []);

  const loadHistories = useCallback(async () => {
    try {
      const [resBase, resDet, resJev] = await Promise.all([
        fetch('/api/chat/history?system=baseline', { cache: 'no-store' }),
        fetch('/api/chat/history?system=enhanced_deterministic', { cache: 'no-store' }),
        fetch('/api/chat/history?system=enhanced_jev', { cache: 'no-store' }),
      ]);

      if (resBase.ok) {
        const dataBase = await resBase.json();
        setBaselineMessages(toBubbles(dataBase, 'baseline'));
      }
      if (resDet.ok) {
        const dataDet = await resDet.json();
        setEnhancedMessages(toBubbles(dataDet, 'deterministic'));
      }
      if (resJev.ok) {
        const dataJev = await resJev.json();
        setJevMessages(toBubbles(dataJev, 'jev'));
      }
    } catch (err: any) {
      if (err?.name === 'AbortError') return;
      console.error('Failed to load chat histories', err);
    }
  }, []);

  const loadInspectors = useCallback(async () => {
    try {
      const [resBase, resDet, resJev] = await Promise.all([
        fetch('/api/agents/inspector?system=baseline', { cache: 'no-store' }),
        fetch('/api/agents/inspector?system=enhanced_deterministic', { cache: 'no-store' }),
        fetch('/api/agents/inspector?system=enhanced_jev', { cache: 'no-store' }),
      ]);

      if (resBase.ok) {
        const dataBase = await resBase.json();
        setBaselineInspector(dataBase);
      }
      if (resDet.ok) {
        const dataDet = await resDet.json();
        setEnhancedInspector(dataDet);
      }
      if (resJev.ok) {
        const dataJev = await resJev.json();
        setJevInspector(dataJev);
      }
    } catch (err: any) {
      console.error('Failed to load inspector data', err);
    }
  }, []);

  useEffect(() => {
    void loadHistories();
    void loadInspectors();
  }, [loadHistories, loadInspectors]);

  // Periodic polling
  useEffect(() => {
    const intervalId = window.setInterval(() => {
      void loadHistories();
      void loadInspectors();
    }, POLL_INTERVAL_MS);

    return () => window.clearInterval(intervalId);
  }, [loadHistories, loadInspectors]);

  // Timezone detection on first load
  useEffect(() => {
    const detectAndStoreTimezone = async () => {
      if (settings.timezone) return;
      try {
        const browserTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
        const response = await fetch('/api/timezone', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ timezone: browserTimezone }),
        });
        if (response.ok) {
          setSettings({ ...settings, timezone: browserTimezone });
        }
      } catch (err) {
        console.debug('Timezone detection failed:', err);
      }
    };

    void detectAndStoreTimezone();
  }, [settings, setSettings]);

  const canSubmit = input.trim().length > 0;
  const isAnyWaiting = baselineWaiting || enhancedWaiting || jevWaiting;

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed) return;

      setError(null);
      setBaselineWaiting(true);
      setEnhancedWaiting(true);
      setJevWaiting(true);

      const timestamp = Date.now();
      const userMessageBase: ChatBubble = {
        id: `user-baseline-${timestamp}`,
        role: 'user',
        text: formatEscapeCharacters(trimmed),
      };
      const userMessageDet: ChatBubble = {
        id: `user-deterministic-${timestamp}`,
        role: 'user',
        text: formatEscapeCharacters(trimmed),
      };
      const userMessageJev: ChatBubble = {
        id: `user-jev-${timestamp}`,
        role: 'user',
        text: formatEscapeCharacters(trimmed),
      };

      setBaselineMessages((prev) => [...prev, userMessageBase]);
      setEnhancedMessages((prev) => [...prev, userMessageDet]);
      setJevMessages((prev) => [...prev, userMessageJev]);

      try {
        // Send simultaneously to all three systems
        await Promise.all([
          fetch('/api/chat?system=baseline', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              messages: [{ role: 'user', content: trimmed }],
            }),
          }),
          fetch('/api/chat?system=enhanced_deterministic', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              messages: [{ role: 'user', content: trimmed }],
            }),
          }),
          fetch('/api/chat?system=enhanced_jev', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              messages: [{ role: 'user', content: trimmed }],
            }),
          }),
        ]);
      } catch (err: any) {
        console.error('Failed to dispatch message to servers', err);
        setError(err?.message || 'Failed to dispatch message');
      } finally {
        // Poll for responses
        let pollAttempts = 0;
        const maxPollAttempts = 35;

        const pollInterval = window.setInterval(async () => {
          pollAttempts++;
          try {
            await Promise.all([loadHistories(), loadInspectors()]);
          } catch {}

          if (pollAttempts >= maxPollAttempts) {
            window.clearInterval(pollInterval);
            setBaselineWaiting(false);
            setEnhancedWaiting(false);
            setJevWaiting(false);
          }
        }, 1500);

        // Turn off individual waiting indicators once a new assistant message arrives
        setTimeout(() => {
          setBaselineWaiting(false);
          setEnhancedWaiting(false);
          setJevWaiting(false);
        }, 25000);
      }
    },
    [loadHistories, loadInspectors],
  );

  const handleClearAll = useCallback(async () => {
    try {
      const res = await fetch('/api/chat/history', { method: 'DELETE' });
      if (!res.ok) {
        console.error('Failed to clear chat history', res.statusText);
        return;
      }
      setBaselineMessages([]);
      setEnhancedMessages([]);
      setJevMessages([]);
      setBaselineInspector(null);
      setEnhancedInspector(null);
      setJevInspector(null);
      await Promise.all([loadHistories(), loadInspectors()]);
    } catch (err) {
      console.error('Failed to clear chat history', err);
    }
  }, [loadHistories, loadInspectors]);

  const handleSubmit = useCallback(async () => {
    if (!canSubmit) return;
    const value = input;
    setInput('');
    try {
      await sendMessage(value);
    } catch {
      setInput(value);
    }
  }, [canSubmit, input, sendMessage]);

  return (
    <div className="mx-auto flex min-h-screen max-w-[1700px] flex-col bg-white p-4">
      {/* Top Header */}
      <ChatHeader onOpenSettings={openSettings} onClearHistory={handleClearAll} />

      {/* Sub-header Banner */}
      <div className="mb-4 flex flex-wrap items-center justify-between rounded-lg border border-indigo-100 bg-indigo-50/50 px-4 py-2.5 text-xs text-indigo-900 shadow-sm">
        <div className="flex items-center space-x-2.5">
          <span className="flex h-2.5 w-2.5 rounded-full bg-indigo-500 animate-pulse" />
          <span className="font-semibold text-sm">Interactive Tri-Chat: 3-Way Live Routing Evaluation</span>
        </div>
        <div className="text-indigo-700">
          Comparing <strong>Baseline (:8001)</strong> vs <strong>Enhanced Deterministic (:8002)</strong> vs <strong>Enhanced TypeSafe Jev (:8002)</strong>
        </div>
      </div>

      {error && (
        <div className="mb-4">
          <ErrorBanner message={error} onDismiss={() => setError(null)} />
        </div>
      )}

      {/* Tri-Chat & Inspector 3-Column Split View */}
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-3 pb-24">
        {/* Column 1: Baseline */}
        <div className="flex flex-col space-y-4">
          <div className="rounded-xl border border-blue-200 bg-blue-50/30 p-2 shadow-sm">
            <div className="flex items-center justify-between border-b border-blue-200/60 pb-2 px-2">
              <div className="flex items-center space-x-2">
                <span className="h-2.5 w-2.5 rounded-full bg-blue-600" />
                <h2 className="text-sm font-bold text-blue-900">Baseline OpenPoke</h2>
                <span className="rounded bg-blue-100 px-1.5 py-0.2 text-[10px] font-semibold text-blue-800">
                  Port 8001
                </span>
              </div>
              <span className="text-[11px] text-gray-500 font-medium">Historical Full-Roster Exposure</span>
            </div>

            <ChatMessages
              messages={baselineMessages}
              isWaitingForResponse={baselineWaiting}
              scrollContainerRef={baselineScrollRef}
              onScroll={handleBaselineScroll}
              className="h-[38vh]"
            />
          </div>

          <AgentInspectorPanel
            system="baseline"
            title="Baseline Execution Agents (Port 8001)"
            data={baselineInspector}
            isLoading={baselineInspectorLoading}
          />
        </div>

        {/* Column 2: Enhanced Deterministic */}
        <div className="flex flex-col space-y-4">
          <div className="rounded-xl border border-emerald-200 bg-emerald-50/30 p-2 shadow-sm">
            <div className="flex items-center justify-between border-b border-emerald-200/60 pb-2 px-2">
              <div className="flex items-center space-x-2">
                <span className="h-2.5 w-2.5 rounded-full bg-emerald-600" />
                <h2 className="text-sm font-bold text-emerald-900">Enhanced Deterministic</h2>
                <span className="rounded bg-emerald-100 px-1.5 py-0.2 text-[10px] font-semibold text-emerald-800">
                  Port 8002
                </span>
              </div>
              <span className="text-[11px] text-emerald-700 font-medium">Trigram / Lexical Token Router</span>
            </div>

            <ChatMessages
              messages={enhancedMessages}
              isWaitingForResponse={enhancedWaiting}
              scrollContainerRef={enhancedScrollRef}
              onScroll={handleEnhancedScroll}
              className="h-[38vh]"
            />
          </div>

          <AgentInspectorPanel
            system="enhanced_deterministic"
            title="Deterministic Router (Port 8002)"
            data={enhancedInspector}
            isLoading={enhancedInspectorLoading}
          />
        </div>

        {/* Column 3: Enhanced TypeSafe Jev */}
        <div className="flex flex-col space-y-4">
          <div className="rounded-xl border border-purple-200 bg-purple-50/30 p-2 shadow-sm">
            <div className="flex items-center justify-between border-b border-purple-200/60 pb-2 px-2">
              <div className="flex items-center space-x-2">
                <span className="h-2.5 w-2.5 rounded-full bg-purple-600" />
                <h2 className="text-sm font-bold text-purple-900">Enhanced TypeSafe Jev</h2>
                <span className="rounded bg-purple-100 px-1.5 py-0.2 text-[10px] font-semibold text-purple-800">
                  Port 8002
                </span>
              </div>
              <span className="text-[11px] text-purple-700 font-medium">Map/Reduce Activity Cards</span>
            </div>

            <ChatMessages
              messages={jevMessages}
              isWaitingForResponse={jevWaiting}
              scrollContainerRef={jevScrollRef}
              onScroll={handleJevScroll}
              className="h-[38vh]"
            />
          </div>

          <AgentInspectorPanel
            system="enhanced_jev"
            title="TypeSafe Jev Router (Port 8002)"
            data={jevInspector}
            isLoading={jevInspectorLoading}
          />
        </div>
      </div>

      {/* Sticky Bottom Input Bar */}
      <div className="sticky bottom-0 z-30 mt-6 border-t border-gray-200 bg-white/95 pt-3 pb-3 backdrop-blur shadow-md">
        <ChatInput
          value={input}
          onChange={setInput}
          onSubmit={handleSubmit}
          canSubmit={canSubmit}
          placeholder="Type a prompt to evaluate across Baseline, Deterministic & Jev simultaneously..."
        />
      </div>

      {/* Settings Modal */}
      <SettingsModal
        open={openSettingsModal}
        onClose={closeSettings}
        settings={settings}
        onSave={setSettings}
      />
    </div>
  );
}
