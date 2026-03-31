// ============================================================================
// BenTab — Container for the Ben/Lab chat interface
// ============================================================================

import { useEffect, useRef } from 'react';
import ChatMessages from './ChatMessages';
import ChatInput from './ChatInput';
import { useLabStore } from '../../store/labStore';
import { useLabBriefing } from '../../api/lab';
import { useKeyboardNavHide } from '../../hooks/useKeyboardNavHide';
import styles from './BenTab.module.css';

export default function BenTab() {
  const { initialized, setInitialized, setSystem, addMessage } = useLabStore();
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const briefingQuery = useLabBriefing();

  useKeyboardNavHide(inputRef);

  // Initialize on first mount: inject briefing context as hidden system message
  useEffect(() => {
    if (initialized) return;
    setInitialized(true);

    if (briefingQuery.data) {
      const briefingText = buildSystemPrompt(briefingQuery.data);
      setSystem(briefingText);
      addMessage({
        role: 'assistant',
        content: briefingText,
        hidden: true,
      });
    }
  }, [initialized, setInitialized, briefingQuery.data, setSystem, addMessage]);

  // Update system prompt when briefing data loads (for late arrivals)
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
