// ============================================================================
// ParlayLeg — Individual leg card within the parlay ticket
// ============================================================================

import type { ParlayLeg as ParlayLegType } from '../../types';
import { getStatLabel } from '../../utils/statLabels';
import { TEAM_COLORS } from '../../utils/teamColors';
import { hexToRgba } from '../../utils/hexToRgba';
import styles from './ParlayLeg.module.css';

interface ParlayLegProps {
  leg: ParlayLegType;
  index: number;
}

export default function ParlayLeg({ leg, index }: ParlayLegProps) {
  const teamColor = TEAM_COLORS[leg.team] || '#14b8a6';
  const statLabel = getStatLabel(leg.stat_type);
  const conf = (leg.blended_confidence * 100).toFixed(0);
  const isOver = leg.direction === 'over';
  const resolved = leg.result && leg.result !== 'pending';

  return (
    <div
      className={styles.leg}
      style={{ '--leg-color': teamColor, '--leg-glow': hexToRgba(teamColor, 0.12) } as React.CSSProperties}
    >
      {/* Leg number indicator */}
      <div className={styles.legIndex} style={{ color: teamColor }}>
        {index + 1}
      </div>

      <div className={styles.legContent}>
        {/* Header: name + matchup */}
        <div className={styles.legHeader}>
          <div className={styles.legWho}>
            <span className={styles.legName}>{leg.player_name}</span>
            <span className={styles.legMatchup}>
              {leg.team} vs {leg.opponent}
            </span>
          </div>
          <div className={styles.legPlay}>
            <span className={`${styles.legPill} ${isOver ? styles.over : styles.under}`}>
              {leg.direction.toUpperCase()}
            </span>
            <span className={styles.legLine}>
              {leg.line} {statLabel}
            </span>
          </div>
        </div>

        {/* Meta: confidence + chips + L5 */}
        <div className={styles.legMeta}>
          <div className={styles.legChips}>
            <span className={`${styles.legChip} ${styles.accent}`}>{conf}%</span>
            {leg.american_odds != null && (
              <span className={styles.legChip}>
                {leg.american_odds > 0 ? '+' : ''}{leg.american_odds}
              </span>
            )}
            <span className={styles.legChip}>
              Proj {leg.projection.toFixed(1)}
            </span>
          </div>

          {/* L5 tiles */}
          {leg.recent_values.length > 0 && (
            <div className={styles.l5Col}>
              <span className={styles.l5Label}>L5</span>
              <div className={styles.l5}>
                {leg.recent_values.slice(-5).map((val, i) => {
                  const hit = isOver ? val > leg.line : val < leg.line;
                  return (
                    <span
                      key={i}
                      className={`${styles.tile} ${hit ? styles.tileHit : styles.tileMiss}`}
                    >
                      {Math.round(val)}
                    </span>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Result (when resolved) */}
        {resolved && (
          <div className={styles.resultRow}>
            <span className={styles.resultLabel}>ACTUAL</span>
            <span className={styles.resultActual}>
              {leg.actual_stat != null ? leg.actual_stat.toFixed(1) : '--'}
            </span>
            <span
              className={`${styles.resultPill} ${
                leg.result === 'hit' ? styles.resultHit : styles.resultMiss
              }`}
            >
              {leg.result?.toUpperCase()}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
