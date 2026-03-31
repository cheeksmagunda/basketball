// ============================================================================
// ParlayTicketSkeleton — Shimmer placeholder while parlay data loads
// ============================================================================

import styles from './ParlayTicketSkeleton.module.css';

export default function ParlayTicketSkeleton() {
  return (
    <div className={styles.ticket}>
      {/* Header skeleton */}
      <div className={styles.header}>
        <div className={`${styles.block} ${styles.titleBlock}`} />
        <div className={`${styles.block} ${styles.probBlock}`} />
      </div>
      <div className={styles.divider} />
      {/* 3 leg skeletons */}
      {[1, 2, 3].map((i) => (
        <div key={i} className={styles.leg}>
          <div className={`${styles.block} ${styles.legName}`} />
          <div className={styles.legRow}>
            <div className={`${styles.block} ${styles.legStat}`} />
            <div className={`${styles.block} ${styles.legStat}`} />
            <div className={`${styles.block} ${styles.legStat}`} />
          </div>
        </div>
      ))}
    </div>
  );
}
