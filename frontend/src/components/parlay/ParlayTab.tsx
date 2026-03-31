// ============================================================================
// ParlayTab — Container showing today's parlay ticket + recent history
// ============================================================================

import { useParlay, useParlayHistory } from '../../api/parlay';
import ParlayTicket from './ParlayTicket';
import ParlayHistory from './ParlayHistory';
import OracleLoader from '../shared/OracleLoader';
import EmptyState from '../shared/EmptyState';

export default function ParlayTab() {
  const { data: parlay, isLoading, error, refetch } = useParlay();
  const historyQuery = useParlayHistory();

  if (isLoading) return <OracleLoader visible />;

  if (error || !parlay) {
    return (
      <EmptyState
        icon="&nbsp;"
        message="Couldn't load parlay."
        action={{ label: 'Retry', onClick: () => refetch() }}
      />
    );
  }

  if (parlay.error || !parlay.legs?.length) {
    return (
      <div>
        <EmptyState
          icon="&nbsp;"
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
    <div>
      <ParlayTicket parlay={parlay} />
      <ParlayHistory
        data={historyQuery.data || null}
        isLoading={historyQuery.isLoading}
      />
    </div>
  );
}
