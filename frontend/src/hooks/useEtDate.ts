// ============================================================================
// useEtDate
// Returns the current Eastern Time date as a YYYY-MM-DD string.
// Port of the vanilla JS _etToday() function from app.js.
// ============================================================================

import { useMemo } from 'react';

/**
 * React hook returning today's ET date as YYYY-MM-DD.
 *
 * The value is memoized per render (not reactive to clock ticks).
 * Components that need to detect midnight rollover should pair this
 * with a refetch or interval.
 */
export function useEtDate(): string {
  return useMemo(() => {
    const now = new Date();
    const etStr = now.toLocaleDateString('en-CA', { timeZone: 'America/New_York' });
    return etStr; // YYYY-MM-DD
  }, []);
}
