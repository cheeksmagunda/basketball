// ============================================================================
// BenTab — Container for the Ben/Lab chat interface
// ============================================================================

import { useEffect, useRef } from 'react';
import ChatMessages from './ChatMessages';
import ChatInput from './ChatInput';
import { useLabStore } from '../../store/labStore';
import { useLabBriefing } from '../../api/lab';
import { useKeyboardNavHide } from '../../hooks/useKeyboardNavHide';
import type { LabBriefing } from '../../types';
import styles from './BenTab.module.css';

export default function BenTab() {
  const { initialized, setInitialized, setSystem, addMessage } = useLabStore();
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const briefingQuery = useLabBriefing();

  useKeyboardNavHide(inputRef);

  // Remove .app bottom padding while Ben is mounted — Ben uses a fixed-height
  // flex layout that accounts for the nav, so the 140px clearance causes overflow.
  useEffect(() => {
    const app = document.querySelector('.app');
    if (app) app.classList.add('ben-active');
    return () => {
      const appEl = document.querySelector('.app');
      if (appEl) appEl.classList.remove('ben-active');
    };
  }, []);

  // When briefing data arrives and we haven't initialized, inject system context + visible greeting
  useEffect(() => {
    if (initialized || !briefingQuery.data) return;
    setInitialized(true);

    const briefingText = buildSystemPrompt(briefingQuery.data);
    setSystem(briefingText);
    addMessage({ role: 'assistant', content: briefingText, hidden: true });
    addMessage({ role: 'assistant', content: buildGreeting(briefingQuery.data) });
  }, [initialized, briefingQuery.data, setInitialized, setSystem, addMessage]);

  // Fallback: if briefing fails, still show a basic greeting
  useEffect(() => {
    if (initialized || !briefingQuery.isError) return;
    setInitialized(true);
    addMessage({
      role: 'assistant',
      content: 'What can I help with? Deep dives, config changes, backtests — just ask.',
    });
  }, [initialized, briefingQuery.isError, setInitialized, addMessage]);

  // Update system prompt when briefing data refreshes
  useEffect(() => {
    if (briefingQuery.data) {
      setSystem(buildSystemPrompt(briefingQuery.data));
    }
  }, [briefingQuery.data, setSystem]);

  return (
    <div className={styles.container}>
      <div className={styles.chatArea}>
        <ChatMessages />
      </div>
      <div className={styles.inputWrap}>
        <ChatInput inputRef={inputRef} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Build the system prompt from briefing data (matches vanilla JS buildLabSystemPrompt)
// ---------------------------------------------------------------------------

function buildSystemPrompt(briefing: object): string {
  const lines: string[] = [
    'You are Ben, the Basketball Oracle\'s AI assistant.',
    'You help analyze NBA prediction accuracy, tune model parameters, and run backtests.',
    '',
    '## Current Briefing',
    JSON.stringify(briefing, null, 2),
    '',
    'Answer concisely. Use data when available. You can suggest config changes, backtests, and analysis.',
  ];
  return lines.join('\n');
}

// ---------------------------------------------------------------------------
// Build a visible greeting from briefing data
// ---------------------------------------------------------------------------

function buildGreeting(briefing: LabBriefing): string {
  const parts: string[] = [];

  const ls = briefing.latest_slate;
  if (ls) {
    const mae = ls.mean_absolute_error?.toFixed(2) ?? '—';
    parts.push(`**Last slate (${ls.date}):** MAE ${mae} across ${ls.players_with_actuals} players`);

    if (ls.directional_accuracy != null) {
      parts[parts.length - 1] += ` · ${(ls.directional_accuracy * 100).toFixed(0)}% directional accuracy`;
    }

    if (ls.biggest_misses?.length > 0) {
      const m = ls.biggest_misses[0];
      parts.push(
        `Biggest miss: **${m.name}** (predicted ${m.predicted.toFixed(1)}, actual ${m.actual.toFixed(1)})`
      );
    }
  }

  const ra = briefing.rolling_accuracy;
  if (ra?.overall_mae != null) {
    parts.push(`**Rolling MAE:** ${ra.overall_mae.toFixed(2)} across ${ra.slates_with_data} slates`);
  }

  if (briefing.patterns?.length > 0) {
    parts.push(`*Pattern detected:* ${briefing.patterns[0].description}`);
  }

  if (parts.length === 0) {
    parts.push('No recent slate data yet.');
  }

  parts.push('');
  parts.push('What can I help with? Deep dives, config changes, backtests — just ask.');

  return parts.join('\n');
}
