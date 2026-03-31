// ============================================================================
// ChatMessages — Message list with auto-scroll
// ============================================================================

import { useEffect, useRef } from 'react';
import { useLabStore } from '../../store/labStore';
import ChatMessage from './ChatMessage';
import styles from './ChatMessages.module.css';

export default function ChatMessages() {
  const messages = useLabStore((s) => s.messages);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Filter out hidden messages for display
  const visibleMessages = messages.filter((m) => !m.hidden);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [visibleMessages.length]);

  return (
    <div
      className={`${styles.messages} ${visibleMessages.length > 0 ? styles.hasMessages : ''}`}
    >
      {/* Empty state — centered in scroll area */}
      {visibleMessages.length === 0 && (
        <div className={styles.emptyState}>
          <div className={styles.emptyIcon}>
            <span role="img" aria-label="crystal ball">&#128302;</span>
          </div>
          <span className={styles.emptyTitle}>Ask Ben anything</span>
          <span className={styles.emptySub}>
            Accuracy analysis, config changes, backtests, and more.
          </span>
        </div>
      )}

      {visibleMessages.map((msg, i) => (
        <ChatMessage key={i} message={msg} />
      ))}

      {/* Scroll anchor */}
      <div ref={bottomRef} />
    </div>
  );
}
