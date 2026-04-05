// ============================================================================
// Slate / Predict API — React Query hooks
// Mirrors: /api/slate, /api/picks, /api/games, /api/save-predictions
// ============================================================================

import { useQuery, useMutation } from '@tanstack/react-query';
import { fetchJson, fetchWithTimeout } from './client';
import type { SlateData, PicksData, GamesResponse } from '../types';

// ---------------------------------------------------------------------------
// /api/slate
// ---------------------------------------------------------------------------

/**
 * Full-slate predictions (Starting 5 + Moonshot).
 *
 * - 30s timeout (generation can be slow on cold starts)
 * - staleTime 5 min so repeated tab switches don't re-fetch
 * - When locked and not all_complete, refetchInterval 60s for game-final detection
 * - retry 3 with escalating delays [8s, 20s, 40s]
 * - refetchOnWindowFocus so returning from background picks up fresh data
 */
export function useSlate() {
  return useQuery<SlateData>({
    queryKey: ['slate'],
    queryFn: () => fetchJson<SlateData>('/api/slate', 30_000),
    staleTime: 60 * 1000, // 60s — matches backend _CACHE_TTLS["slate"]
    refetchOnWindowFocus: true,
    retry: 1,
    retryDelay: 8_000,
  });
}

// ---------------------------------------------------------------------------
// /api/games
// ---------------------------------------------------------------------------

/**
 * Today's games with lock status.
 *
 * Backend returns { data: GameInfo[], cache_status, cached_at }
 * so we unwrap .data here to keep consumers working with a plain array.
 */
export function useGames() {
  return useQuery<GamesResponse>({
    queryKey: ['games'],
    queryFn: async () => {
      const res = await fetchJson<{ data: GamesResponse }>('/api/games', 10_000);
      return res.data ?? [];
    },
    staleTime: 60 * 1000, // 60s — matches backend cache TTL so lock status stays fresh
  });
}

// ---------------------------------------------------------------------------
// /api/picks?gameId=X
// ---------------------------------------------------------------------------

/**
 * Per-game predictions (THE LINE UP).
 * Enabled only when a gameId is selected.
 */
export function usePicks(gameId: string | null) {
  return useQuery<PicksData>({
    queryKey: ['picks', gameId],
    queryFn: () => fetchJson<PicksData>(`/api/picks?gameId=${gameId}`, 15_000),
    enabled: !!gameId,
    staleTime: 5 * 60 * 1000, // 5 min
  });
}

// ---------------------------------------------------------------------------
// POST /api/save-predictions
// ---------------------------------------------------------------------------

/**
 * Mutation: persist locked predictions to GitHub CSV.
 * Deduped server-side (skips commit if unchanged).
 */
export function useSavePredictions() {
  return useMutation({
    mutationFn: async (data: { predictions: unknown }) => {
      const r = await fetchWithTimeout('/api/save-predictions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      }, 10_000);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    },
  });
}
