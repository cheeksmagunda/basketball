import { useState, useEffect } from 'react';
import styles from './OracleLoader.module.css';

const ORACLE_MESSAGES = [
  'READING THE GAME',
  'CONSULTING THE ORACLE',
  'CALCULATING EDGE',
  'ANALYZING MATCHUPS',
  'PROJECTING VALUE',
  'SCANNING THE SLATE',
];

const MESSAGE_INTERVAL_MS = 1800;

interface OracleLoaderProps {
  visible: boolean;
}

export default function OracleLoader({ visible }: OracleLoaderProps) {
  const [messageIndex, setMessageIndex] = useState(0);

  useEffect(() => {
    if (!visible) return;

    const id = setInterval(() => {
      setMessageIndex((prev) => (prev + 1) % ORACLE_MESSAGES.length);
    }, MESSAGE_INTERVAL_MS);

    return () => clearInterval(id);
  }, [visible]);

  // Reset message index when loader becomes visible
  useEffect(() => {
    if (visible) {
      setMessageIndex(0);
    }
  }, [visible]);

  const className = [styles.oracleLoader, visible ? styles.visible : '']
    .filter(Boolean)
    .join(' ');

  return (
    <div className={className} aria-hidden={!visible} role="status">
      <div className={styles['magic8-ball']}>
        <div className={styles['magic8-num']}>8</div>
        <div className={styles['magic8-window']}>
          <span key={messageIndex} className={styles['magic8-text']}>
            {ORACLE_MESSAGES[messageIndex]}
          </span>
        </div>
      </div>
      <span className={styles['oracle-label']}>THE ORACLE</span>
    </div>
  );
}
