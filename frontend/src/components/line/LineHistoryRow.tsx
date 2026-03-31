import { getStatLabel } from '../../utils/statLabels';
import styles from './LineHistoryRow.module.css';
import type { LinePick } from '../../types';

interface LineHistoryRowProps {
  pick: LinePick;
  onClick?: () => void;
}

/**
 * Format a date string (YYYY-MM-DD) to a short display: "Mar 27".
 */
function fmtDateShort(dateStr: string): string {
  try {
    const [y, m, d] = dateStr.split('-').map(Number);
    const dt = new Date(y, m - 1, d);
    return dt.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  } catch {
    return dateStr;
  }
}

export default function LineHistoryRow({ pick, onClick }: LineHistoryRowProps) {
  const statLabel = getStatLabel(pick.stat_type);
  const isOver = pick.direction === 'over';
  const dirClass = isOver ? styles.over : styles.under;

  // Line display: "12.5 PTS"
  const lineDisplay = pick.line != null
    ? `${Number.isInteger(pick.line) ? pick.line : pick.line.toFixed(1)} ${statLabel}`
    : statLabel;

  // Sub-line: "12.5 PTS · Mar 27"
  const datePart = pick.date ? fmtDateShort(pick.date) : '';
  const subText = datePart ? `${lineDisplay}  ·  ${datePart}` : lineDisplay;

  return (
    <div
      className={styles.row}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={
        onClick
          ? (e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onClick();
              }
            }
          : undefined
      }
    >
      {/* Left side: direction pill + player info */}
      <div className={styles.left}>
        <span className={`${styles.dirPill} ${dirClass}`}>
          {isOver ? 'O' : 'U'}
        </span>
        <div className={styles.info}>
          <div className={styles.name}>{pick.player_name}</div>
          <div className={styles.sub}>{subText}</div>
        </div>
      </div>

      {/* Right side: result pill */}
      <ResultPill result={pick.result} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Result pill sub-component
// ---------------------------------------------------------------------------

function ResultPill({ result }: { result: string }) {
  const cls =
    result === 'hit'
      ? styles.resultHit
      : result === 'miss'
        ? styles.resultMiss
        : styles.resultPending;

  const label =
    result === 'hit' ? 'HIT' : result === 'miss' ? 'MISS' : 'PENDING';

  return <span className={`${styles.resultPill} ${cls}`}>{label}</span>;
}
