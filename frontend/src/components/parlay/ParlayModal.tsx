// ============================================================================
// ParlayModal — Bottom-sheet modal showing full parlay detail + resolution
// ============================================================================

import type { ParlayHistoryItem } from '../../types';
import { getStatLabel } from '../../utils/statLabels';
import BottomSheet from '../shared/BottomSheet';
import styles from './ParlayModal.module.css';

interface ParlayModalProps {
  parlay: ParlayHistoryItem | null;
  onClose: () => void;
}

function formatDate(dateStr: string): string {
  const d = new Date(dateStr + 'T12:00:00');
  return d.toLocaleDateString('en-US', {
    weekday: 'long',
    month: 'long',
    day: 'numeric',
  });
}

export default function ParlayModal({ parlay, onClose }: ParlayModalProps) {
  if (!parlay) return <BottomSheet open={false} onClose={onClose}><div /></BottomSheet>;

  const pct = (parlay.combined_probability * 100).toFixed(1);
  const corrMult = parlay.correlation_multiplier?.toFixed(2) ?? '1.00';
  const allResolved = parlay.legs.every((l) => l.result && l.result !== 'pending');
  const hitsCount = parlay.legs.filter((l) => l.result === 'hit').length;

  return (
    <BottomSheet open={parlay !== null} onClose={onClose}>
      <div className={styles.modal}>
        {/* Header */}
        <div className={styles.header}>
          <div>
            <span className={styles.title}>3-LEG PARLAY</span>
            <span className={styles.date}>{formatDate(parlay.date)}</span>
          </div>
          <div className={styles.headerRight}>
            {allResolved && (
              <span
                className={`${styles.overallPill} ${
                  parlay.result === 'hit' ? styles.overallHit : styles.overallMiss
                }`}
              >
                {parlay.result === 'hit' ? `HIT (${hitsCount}/3)` : `MISS (${hitsCount}/3)`}
              </span>
            )}
            <span className={styles.prob}>{pct}%</span>
          </div>
        </div>

        {/* Divider */}
        <div className={styles.divider} />

        {/* Legs detail */}
        <div className={styles.legs}>
          {parlay.legs.map((leg, i) => {
            const isOver = leg.direction === 'over';
            const statLabel = getStatLabel(leg.stat_type);
            const resolved = leg.result && leg.result !== 'pending';

            return (
              <div key={i} className={styles.leg}>
                <div className={styles.legTop}>
                  <div className={styles.legInfo}>
                    <span className={styles.legNum}>{i + 1}</span>
                    <div className={styles.legNameBlock}>
                      <span className={styles.legName}>{leg.player_name}</span>
                      <span className={styles.legTeam}>
                        {leg.team} vs {leg.opponent}
                      </span>
                    </div>
                  </div>
                  <div className={styles.legPlayBlock}>
                    <span
                      className={`${styles.dirPill} ${isOver ? styles.dirOver : styles.dirUnder}`}
                    >
                      {leg.direction.toUpperCase()}
                    </span>
                    <span className={styles.legLineVal}>
                      {leg.line} {statLabel}
                    </span>
                  </div>
                </div>

                {/* Stats row */}
                <div className={styles.legStats}>
                  <div className={styles.legStatCol}>
                    <span className={styles.legStatLabel}>PROJ</span>
                    <span className={styles.legStatVal}>{leg.projection.toFixed(1)}</span>
                  </div>
                  <div className={styles.legStatCol}>
                    <span className={styles.legStatLabel}>AVG</span>
                    <span className={styles.legStatVal}>{leg.season_avg.toFixed(1)}</span>
                  </div>
                  <div className={styles.legStatCol}>
                    <span className={styles.legStatLabel}>CONF</span>
                    <span className={styles.legStatVal}>
                      {(leg.blended_confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  {resolved && (
                    <div className={styles.legStatCol}>
                      <span className={styles.legStatLabel}>ACTUAL</span>
                      <span
                        className={`${styles.legStatVal} ${
                          leg.result === 'hit' ? styles.valHit : styles.valMiss
                        }`}
                      >
                        {leg.actual_stat != null ? leg.actual_stat.toFixed(1) : '--'}
                      </span>
                    </div>
                  )}
                </div>

                {/* Result pill */}
                {resolved && (
                  <div className={styles.legResultRow}>
                    <span
                      className={`${styles.legResultPill} ${
                        leg.result === 'hit' ? styles.legResultHit : styles.legResultMiss
                      }`}
                    >
                      {leg.result?.toUpperCase()}
                    </span>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Correlation */}
        {parlay.correlation_multiplier != null && parlay.correlation_multiplier !== 1.0 && (
          <div className={styles.corrSection}>
            <span className={styles.corrLabel}>Correlation: {corrMult}x</span>
            {parlay.correlation_reasons && parlay.correlation_reasons.length > 0 && (
              <ul className={styles.corrList}>
                {parlay.correlation_reasons.map((r, i) => (
                  <li key={i} className={styles.corrItem}>{r}</li>
                ))}
              </ul>
            )}
          </div>
        )}

        {/* Narrative */}
        {parlay.narrative && (
          <div className={styles.narrativeWrap}>
            <p className={styles.narrative}>{parlay.narrative}</p>
          </div>
        )}
      </div>
    </BottomSheet>
  );
}
