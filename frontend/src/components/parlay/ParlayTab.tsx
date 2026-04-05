// ============================================================================
// ParlayTab — Container showing today's parlay ticket + recent history
// ============================================================================

import { useParlay, useParlayHistory } from '../../api/parlay';
import ParlayTicket from './ParlayTicket';
import ParlayHistory from './ParlayHistory';
import ParlayTicketSkeleton from './ParlayTicketSkeleton';
import EmptyState from '../shared/EmptyState';
import styles from './ParlayTab.module.css';

export default function ParlayTab() {
  const { data: parlay, isLoading, error, refetch } = useParlay();
  const historyQuery = useParlayHistory();

  // Inline skeleton while parlay data loads (prefetch started on app mount)
  if (isLoading && !parlay) {
    return (
      <div className={styles.root}>
        <ParlayTicketSkeleton />
        <ParlayHistory data={historyQuery.data || null} isLoading={historyQuery.isLoading} />
      </div>
    );
  }

  if (error || !parlay) {
    return (
      <div className={styles.root}>
        <EmptyState
          icon=""
          message="Couldn't load parlay."
          action={{ label: 'Retry', onClick: () => refetch() }}
        />
      </div>
    );
  }

  if (parlay.error || !parlay.legs?.length) {
    return (
      <div className={styles.root}>
        <EmptyState
          icon=""
          message={parlay.error || "No valid parlay found for today's slate."}
        />
        <ParlayHistory
          data={historyQuery.data || null}
          isLoading={historyQuery.isLoading}
        />
      </div>
    );
  }

  return (
    <div className={styles.root}>
      <ParlayTicket parlay={parlay} />
      <ParlayHistory
        data={historyQuery.data || null}
        isLoading={historyQuery.isLoading}
      />
    </div>
  );
}
