// ============================================================================
// Parlay API — React Query hooks
// Mirrors: /api/parlay, /api/parlay-history
// ============================================================================

import { useQuery } from '@tanstack/react-query';
import { fetchJson } from './client';
import type { ParlayData, ParlayHistoryResponse } from '../types';

// ---------------------------------------------------------------------------
// /api/parlay
// ---------------------------------------------------------------------------

/**
 * Safest 3-leg player prop parlay on today's slate.
 *
 * - 90s timeout (engine can be slow on cold starts, Z-score + Odds API)
 * - staleTime 30 min (parlay is cached 30 min server-side)
 * - refetchOnWindowFocus disabled to avoid slow re-fetches
 */
export function useParlay() {
  return useQuery<ParlayData>({
    queryKey: ['parlay'],
    queryFn: () => fetchJson<ParlayData>('/api/parlay', 90_000),
    staleTime: 30 * 60 * 1000, // 30 min
    refetchOnWindowFocus: false,
    retry: 2,
  });
}

// ---------------------------------------------------------------------------
// /api/parlay-history
// ---------------------------------------------------------------------------

/**
 * Recent parlays with lazy resolution (hit/miss per leg via ESPN box scores).
 *
 * - staleTime 10 min (matches backend cache TTL)
 */
export function useParlayHistory() {
  return useQuery<ParlayHistoryResponse>({
    queryKey: ['parlay-history'],
    queryFn: () => fetchJson<ParlayHistoryResponse>('/api/parlay-history', 15_000),
    staleTime: 10 * 60 * 1000, // 10 min
  });
}
