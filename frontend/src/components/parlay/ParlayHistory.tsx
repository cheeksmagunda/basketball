// ============================================================================
// ParlayHistory — Recent parlays section below the ticket
// ============================================================================

import { useState } from 'react';
import type { ParlayHistoryResponse, ParlayHistoryItem } from '../../types';
import { getStatLabel } from '../../utils/statLabels';
import SkeletonCard from '../shared/SkeletonCard';
import ParlayModal from './ParlayModal';
import styles from './ParlayHistory.module.css';

interface ParlayHistoryProps {
  data: ParlayHistoryResponse | null;
  isLoading: boolean;
}

function formatDate(dateStr: string): string {
  const d = new Date(dateStr + 'T12:00:00');
  return d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
}

function legsLabel(item: ParlayHistoryItem): string {
  return item.legs
    .map((l) => `${l.player_name.split(' ').pop()} ${l.direction.charAt(0).toUpperCase()} ${l.line} ${getStatLabel(l.stat_type)}`)
    .join(' / ');
}

export default function ParlayHistory({ data, isLoading }: ParlayHistoryProps) {
  const [selected, setSelected] = useState<ParlayHistoryItem | null>(null);

  return (
    <div className={styles.section}>
      <div className={styles.header}>
        <span className={styles.sectionLabel}>RECENT PARLAYS</span>
        {data && data.total > 0 && (
          <div className={styles.stats}>
            {data.hit_rate != null && (
              <span className={styles.statChip}>
                {data.hit_rate.toFixed(0)}% hit rate
              </span>
            )}
            {data.streak > 0 && data.streak_type && (
              <span
                className={`${styles.statChip} ${
                  data.streak_type === 'hit' ? styles.statHit : styles.statMiss
                }`}
              >
                {data.streak}{data.streak_type === 'hit' ? 'W' : 'L'}
              </span>
            )}
          </div>
        )}
      </div>

      {isLoading && <SkeletonCard count={3} />}

      {!isLoading && (!data || data.parlays.length === 0) && (
        <p className={styles.empty}>No parlay history yet.</p>
      )}

      {!isLoading && data && data.parlays.length > 0 && (
        <div className={styles.list}>
          {data.parlays.map((item) => (
            <button
              key={item.date}
              type="button"
              className={styles.row}
              onClick={() => setSelected(item)}
            >
              <div className={styles.rowLeft}>
                <span className={styles.rowDate}>{formatDate(item.date)}</span>
                <span className={styles.rowLegs}>{legsLabel(item)}</span>
              </div>
              <div className={styles.rowRight}>
                <span
                  className={`${styles.resultPill} ${
                    item.result === 'hit'
                      ? styles.resultHit
                      : item.result === 'miss'
                        ? styles.resultMiss
                        : styles.resultPending
                  }`}
                >
                  {item.result === 'pending' ? 'LIVE' : item.result.toUpperCase()}
                </span>
                <span className={styles.chevron}>{'\u203A'}</span>
              </div>
            </button>
          ))}
        </div>
      )}

      <ParlayModal parlay={selected} onClose={() => setSelected(null)} />
    </div>
  );
}
