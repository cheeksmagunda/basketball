import { useEffect } from 'react';
import { useLineOfTheDay, useLineHistory } from '../../api/line';
import { useUiStore } from '../../store/uiStore';
import SlidingPillNav from '../shared/SlidingPillNav';
import LinePickCard from './LinePickCard';
import LineHistory from './LineHistory';
import EmptyState from '../shared/EmptyState';
import styles from './LineTab.module.css';
import type { LinePick } from '../../types';

const LINE_DIRS = [
  { key: 'over', label: 'Over' },
  { key: 'under', label: 'Under' },
];

export default function LineTab() {
  const lineDir = useUiStore((s) => s.lineDir);
  const setLineDir = useUiStore((s) => s.setLineDir);
  const { data, isLoading, error, refetch } = useLineOfTheDay();
  const historyQuery = useLineHistory();

  // Resolve the pick for each direction from the response.
  // The backend may return `over_pick` / `under_pick` directly, or a single
  // primary `pick` with a direction field.
  const overPick: LinePick | null =
    data?.over_pick ?? (data?.pick?.direction === 'over' ? data.pick : null) ?? null;
  const underPick: LinePick | null =
    data?.under_pick ?? (data?.pick?.direction === 'under' ? data.pick : null) ?? null;

  // Auto-correct direction if the selected direction has no pick but the other does.
  useEffect(() => {
    if (!overPick && underPick && lineDir === 'over') setLineDir('under');
    if (!underPick && overPick && lineDir === 'under') setLineDir('over');
  }, [overPick, underPick, lineDir, setLineDir]);

  const activePick = lineDir === 'over' ? overPick : underPick;
  const accentRgb = lineDir === 'over' ? '212,166,64' : '20,184,166';

  // Error state with no cached data
  if (error && !data) {
    return (
      <EmptyState
        icon="signal"
        message="Couldn't reach the server. Tap Retry."
        action={{ label: 'Retry', onClick: () => refetch() }}
      />
    );
  }

  return (
    <div className={styles.root}>
      <SlidingPillNav
        items={LINE_DIRS}
        activeKey={lineDir}
        onChange={(k) => setLineDir(k as 'over' | 'under')}
        accentRgb={accentRgb}
      />

      <div className={styles.pickSection}>
        {isLoading && !activePick ? (
          <LinePickCardSkeleton />
        ) : activePick ? (
          <LinePickCard pick={activePick} />
        ) : data?.next_slate_pending ? (
          <div className={styles.pending}>
            <div className={styles.pendingIcon}>&#127936;</div>
            <div className={styles.pendingText}>Tomorrow's picks are on their way.</div>
            <button className={styles.checkBtn} onClick={() => refetch()} type="button">
              Check for picks
            </button>
          </div>
        ) : !overPick && !underPick && !isLoading ? (
          <EmptyState icon="chart" message="No line pick available." />
        ) : (
          <div className={styles.noDir}>
            No {lineDir === 'over' ? 'OVER' : 'UNDER'} pick today.
          </div>
        )}
      </div>

      <LineHistory
        data={historyQuery.data ?? null}
        isLoading={historyQuery.isLoading}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Skeleton placeholder while the pick card is loading
// ---------------------------------------------------------------------------
function LinePickCardSkeleton() {
  return (
    <div className={styles.skeleton}>
      <div className={styles.skelRow}>
        <div className={`${styles.skelBlock} ${styles.skelWide}`} />
        <div className={`${styles.skelBlock} ${styles.skelNarrow}`} />
      </div>
      <div className={`${styles.skelBlock} ${styles.skelPill}`} />
      <div className={styles.skelDataRow}>
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className={`${styles.skelBlock} ${styles.skelCol}`} />
        ))}
      </div>
      <div className={`${styles.skelBlock} ${styles.skelConclusion}`} />
    </div>
  );
}
