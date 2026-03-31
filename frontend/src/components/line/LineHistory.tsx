import { useMemo, useState } from 'react';
import { useUiStore } from '../../store/uiStore';
import SlidingPillNav from '../shared/SlidingPillNav';
import LineHistoryRow from './LineHistoryRow';
import styles from './LineHistory.module.css';
import type { LineHistoryResponse, LinePick } from '../../types';

interface LineHistoryProps {
  data: LineHistoryResponse | null;
  isLoading: boolean;
  lineDir: 'over' | 'under';
}

const FILTER_TABS = [
  { key: 'all', label: 'All' },
  { key: 'over', label: 'Over' },
  { key: 'under', label: 'Under' },
];

export default function LineHistory({ data, isLoading }: LineHistoryProps) {
  const lineHistoryFilter = useUiStore((s) => s.lineHistoryFilter);
  const setLineHistoryFilter = useUiStore((s) => s.setLineHistoryFilter);

  // Placeholder for a detail modal (tappable rows wire here)
  const [_selectedPick, setSelectedPick] = useState<LinePick | null>(null);

  // Filter picks by direction
  const filteredPicks = useMemo(() => {
    if (!data?.picks) return [];
    if (lineHistoryFilter === 'all') return data.picks;
    return data.picks.filter((p) => p.direction === lineHistoryFilter);
  }, [data?.picks, lineHistoryFilter]);

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
        {/* Hit rate */}
        <div className={styles.statBox}>
          <div className={styles.statValue}>
            {data.hit_rate != null ? `${Math.round(data.hit_rate)}%` : '--'}
          </div>
          <div className={styles.statLabel}>Hit Rate</div>
        </div>

        {/* Streak */}
        <div className={styles.statBox}>
          <div
            className={styles.statValue}
            style={
              data.streak_type === 'hit'
                ? { color: 'var(--color-success)' }
                : data.streak_type === 'miss'
                  ? { color: 'var(--color-danger)' }
                  : undefined
            }
          >
            {data.streak > 0
              ? `${data.streak}x ${data.streak_type === 'hit' ? 'HIT' : 'MISS'}`
              : '--'}
          </div>
          <div className={styles.statLabel}>Streak</div>
        </div>

        {/* Resolved count */}
        <div className={styles.statBox}>
          <div className={styles.statValue}>
            {data.resolved}/{data.total_picks}
          </div>
          <div className={styles.statLabel}>Resolved</div>
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
              onClick={() => setSelectedPick(pick)}
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
