import { useSlate } from '../../api/slate';
import styles from './Header.module.css';

function fmtDateShort(dateStr: string): string {
  const d = new Date(dateStr + 'T12:00:00');
  return d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
}

function fmtTimeCT(isoStr: string): string {
  try {
    const utcStr = isoStr.endsWith('Z') ? isoStr : isoStr.replace(/([+-]\d{2}:\d{2})?$/, 'Z');
    return new Date(utcStr).toLocaleTimeString('en-US', {
      hour: 'numeric', minute: '2-digit', timeZone: 'America/Chicago',
    });
  } catch { return ''; }
}

export default function Header() {
  const { data: slate } = useSlate();

  let countBadge: string | null = null;
  let statusBadge: string | null = null;
  let statusLocked = false;

  if (slate && !slate.error) {
    const isLocked = slate.locked === true;
    const isWarmingUp = Boolean(slate.warming_up);
    const totalGames = (slate.games || []).length;
    const draftable = (slate.games || []).filter(g => !g.locked).length;
    const gameCount = isLocked ? totalGames : draftable;
    const dateStr = slate.date ? fmtDateShort(slate.date) : '';

    // During cold pipeline startup, show "Loading" instead of "0 Games"
    if (isWarmingUp && totalGames === 0) {
      countBadge = `Loading · ${dateStr}`;
    } else {
      countBadge = `${gameCount} Game${gameCount !== 1 ? 's' : ''} · ${dateStr}`;
    }

    if (isLocked && slate.lock_time) {
      const ltStr = fmtTimeCT(slate.lock_time);
      statusBadge = ltStr ? `Locked ${ltStr} CT` : 'Locked';
      statusLocked = true;
    } else if (isLocked) {
      statusBadge = 'Locked';
      statusLocked = true;
    } else if (isWarmingUp) {
      statusBadge = 'Generating\u2026';
    } else {
      const timeNow = new Date().toLocaleTimeString('en-US', {
        hour: 'numeric', minute: '2-digit', timeZone: 'America/Chicago',
      });
      statusBadge = `Updated ${timeNow} CT`;
    }
  }

  return (
    <header className={styles.header}>
      <div className={styles.logo}>
        <div className={styles.logoIcon}>🏀</div>
        <div>
          <h1 className={styles.title}>THE ORACLE</h1>
          <span className={styles.sub}>NBA Draft Optimizer</span>
        </div>
      </div>
      {countBadge && (
        <div className={styles.meta}>
          <div className={styles.badge}>{countBadge}</div>
          {statusBadge && (
            <div className={`${styles.badge} ${statusLocked ? styles.badgeLocked : ''}`}>
              {statusBadge}
            </div>
          )}
        </div>
      )}
    </header>
  );
}
