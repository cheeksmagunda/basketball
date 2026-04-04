// ============================================================================
// ChatMessage — Individual chat bubble (user/assistant/thinking)
// Renders basic markdown (bold, italic, code, newlines).
// ============================================================================

import type { LabMessage } from '../../types';
import styles from './ChatMessage.module.css';

interface ChatMessageProps {
  message: LabMessage;
}

// ---------------------------------------------------------------------------
// Minimal markdown renderer — converts **bold**, *italic*, `code`, and \n to <br>
// ---------------------------------------------------------------------------

function renderMarkdown(text: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  // Split by newlines first, then process inline formatting
  const lines = text.split('\n');

  lines.forEach((line, lineIdx) => {
    if (lineIdx > 0) {
      nodes.push(<br key={`br-${lineIdx}`} />);
    }

    // Process inline formatting: **bold**, *italic*, `code`
    const parts = line.split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g);

    parts.forEach((part, partIdx) => {
      const key = `${lineIdx}-${partIdx}`;
      if (part.startsWith('**') && part.endsWith('**')) {
        nodes.push(<strong key={key}>{part.slice(2, -2)}</strong>);
      } else if (part.startsWith('*') && part.endsWith('*') && part.length > 2) {
        nodes.push(<em key={key}>{part.slice(1, -1)}</em>);
      } else if (part.startsWith('`') && part.endsWith('`')) {
        nodes.push(
          <code key={key} className={styles.inlineCode}>{part.slice(1, -1)}</code>
        );
      } else {
        nodes.push(part);
      }
    });
  });

  return nodes;
}

export default function ChatMessage({ message }: ChatMessageProps) {
  // Thinking state: Ben typing indicator
  if (message.isStatus) {
    return (
      <div className={`${styles.msg} ${styles.assistant} ${styles.thinking}`}>
        <span className={styles.label}>BEN</span>
        <div className={styles.thinkingDots}>
          <span />
          <span />
          <span />
        </div>
        <div className={styles.thinkStatus}>
          {message.content || 'Thinking'}
        </div>
      </div>
    );
  }

  const isUser = message.role === 'user';

  return (
    <div className={`${styles.msg} ${isUser ? styles.user : styles.assistant}`}>
      {/* Ben label for assistant messages */}
      {!isUser && <span className={styles.label}>BEN</span>}

      {/* Image attachment */}
      {message.imageSrc && (
        <img
          src={message.imageSrc}
          alt="Uploaded screenshot"
          className={styles.image}
        />
      )}

      {/* Message content */}
      <div className={styles.content}>
        {renderMarkdown(message.content)}
      </div>
    </div>
  );
}
