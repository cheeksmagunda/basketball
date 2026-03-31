// ============================================================================
// ParlayTicket — Stacked 3-leg ticket card with combined probability + narrative
// ============================================================================

import type { ParlayData } from '../../types';
import ParlayLeg from './ParlayLeg';
import styles from './ParlayTicket.module.css';

interface ParlayTicketProps {
  parlay: ParlayData;
}

export default function ParlayTicket({ parlay }: ParlayTicketProps) {
  const pct = (parlay.combined_probability * 100).toFixed(1);
  const corrMult = parlay.correlation_multiplier?.toFixed(2) ?? '1.00';

  return (
    <div className={styles.ticket}>
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <span className={styles.title}>3-LEG PARLAY</span>
          {parlay.projection_only && (
            <span className={styles.projBadge}>MODEL ONLY</span>
          )}
        </div>
        <div className={styles.headerRight}>
          <span className={styles.probLabel}>COMBINED</span>
          <span className={styles.probValue}>{pct}%</span>
        </div>
      </div>

      {/* Divider */}
      <div className={styles.divider} />

      {/* Legs */}
      <div className={styles.legs}>
        {parlay.legs.map((leg, i) => (
          <ParlayLeg key={`${leg.player_id}-${leg.stat_type}`} leg={leg} index={i} />
        ))}
      </div>

      {/* Correlation section */}
      {parlay.correlation_multiplier !== 1.0 && (
        <div className={styles.correlationSection}>
          <div className={styles.correlationHeader}>
            <span className={styles.correlationLabel}>CORRELATION</span>
            <span className={styles.correlationValue}>{corrMult}x</span>
          </div>
          {parlay.correlation_reasons.length > 0 && (
            <ul className={styles.correlationList}>
              {parlay.correlation_reasons.map((reason, i) => (
                <li key={i} className={styles.correlationReason}>{reason}</li>
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
  );
}
