import { useMemo } from 'react';
import { useUiStore } from '../../store/uiStore';
import SlidingPillNav from '../shared/SlidingPillNav';
import LineHistoryRow from './LineHistoryRow';
import styles from './LineHistory.module.css';
import type { LineHistoryResponse } from '../../types';

interface LineHistoryProps {
  data: LineHistoryResponse | null;
  isLoading: boolean;
}

const FILTER_TABS = [
  { key: 'all', label: 'All' },
  { key: 'over', label: 'Over' },
  { key: 'under', label: 'Under' },
];

export default function LineHistory({ data, isLoading }: LineHistoryProps) {
  const lineHistoryFilter = useUiStore((s) => s.lineHistoryFilter);
  const setLineHistoryFilter = useUiStore((s) => s.setLineHistoryFilter);

  // Filter picks by direction
  const filteredPicks = useMemo(() => {
    if (!data?.picks) return [];
    if (lineHistoryFilter === 'all') return data.picks;
    return data.picks.filter((p) => p.direction === lineHistoryFilter);
  }, [data?.picks, lineHistoryFilter]);

  // Compute hit rate + streak from the filtered subset (not global data)
  const { hitRate, streak, streakType } = useMemo(() => {
    const resolved = filteredPicks.filter(
      (p) => p.result === 'hit' || p.result === 'miss',
    );
    const hits = resolved.filter((p) => p.result === 'hit').length;
    const rate = resolved.length > 0 ? Math.round((hits / resolved.length) * 100) : null;

    let sk = 0;
    let skType: string | null = null;
    for (const p of filteredPicks) {
      if (p.result !== 'hit' && p.result !== 'miss') continue;
      if (skType === null) {
        skType = p.result;
        sk = 1;
      } else if (p.result === skType) {
        sk += 1;
      } else {
        break;
      }
    }
    return { hitRate: rate, streak: sk, streakType: skType };
  }, [filteredPicks]);

  // Accent color for the filter pill nav
  const filterAccent =
    lineHistoryFilter === 'over'
      ? '212,166,64'
      : lineHistoryFilter === 'under'
        ? '20,184,166'
        : '138,150,163'; // muted for "All"

  if (isLoading && !data) {
    return (
      <div className={styles.section}>
        <div className={styles.sectionTitle}>Recent Picks</div>
        <HistorySkeleton />
      </div>
    );
  }

  if (!data || !data.picks?.length) return null;

  return (
    <div className={styles.section}>
      {/* ── Stats row ── */}
      <div className={styles.sectionTitle}>Recent Picks</div>
      <div className={styles.statsRow}>
        {/* Hit rate — computed from filtered picks */}
        <div className={styles.statBox}>
          <div className={styles.statValue}>
            {hitRate != null ? `${hitRate}%` : '--'}
          </div>
          <div className={styles.statLabel}>Hit Rate</div>
        </div>

        {/* Streak — computed from filtered picks */}
        <div className={styles.statBox}>
          <div
            className={styles.statValue}
            style={
              streakType === 'hit'
                ? { color: 'var(--color-success)' }
                : streakType === 'miss'
                  ? { color: 'var(--color-danger)' }
                  : undefined
            }
          >
            {streak > 0
              ? `${streak}x ${streakType === 'hit' ? 'HIT' : 'MISS'}`
              : '--'}
          </div>
          <div className={styles.statLabel}>Streak</div>
        </div>
      </div>

      {/* ── Filter tabs ── */}
      <div className={styles.filterRow}>
        <SlidingPillNav
          items={FILTER_TABS}
          activeKey={lineHistoryFilter}
          onChange={(k) => setLineHistoryFilter(k as 'all' | 'over' | 'under')}
          accentRgb={filterAccent}
        />
      </div>

      {/* ── History list ── */}
      <div className={styles.list}>
        {filteredPicks.length === 0 ? (
          <div className={styles.emptyFilter}>
            No {lineHistoryFilter === 'all' ? '' : lineHistoryFilter.toUpperCase() + ' '}picks yet.
          </div>
        ) : (
          filteredPicks.map((pick, i) => (
            <LineHistoryRow
              key={`${pick.date}-${pick.player_name}-${pick.direction}-${i}`}
              pick={pick}
            />
          ))
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Skeleton placeholder for loading state
// ---------------------------------------------------------------------------
function HistorySkeleton() {
  return (
    <div className={styles.list}>
      {[1, 2, 3].map((i) => (
        <div key={i} className={styles.skelRow}>
          <div className={styles.skelLeft}>
            <div className={`${styles.skelBlock} ${styles.skelPill}`} />
            <div className={styles.skelLines}>
              <div className={`${styles.skelBlock} ${styles.skelName}`} />
              <div className={`${styles.skelBlock} ${styles.skelSub}`} />
            </div>
          </div>
          <div className={`${styles.skelBlock} ${styles.skelResult}`} />
        </div>
      ))}
    </div>
  );
}
