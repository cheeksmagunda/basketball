// ============================================================================
// Line of the Day API — React Query hooks
// Mirrors: /api/line-of-the-day, /api/line-history, /api/line-live-stat,
//          /api/refresh-line-odds
// ============================================================================

import { useQuery, useMutation } from '@tanstack/react-query';
import { fetchJson, fetchWithTimeout } from './client';
import type { LineOfTheDayResponse, LineHistoryResponse, LinePick } from '../types';

// ---------------------------------------------------------------------------
// /api/line-of-the-day
// ---------------------------------------------------------------------------

/**
 * Best player prop picks (over + under), slate-bound.
 *
 * - 90s timeout (generation can be slow on cold starts)
 * - staleTime 5 min to avoid redundant fetches on tab switches
 * - Pass nocache=true to bypass server cache
 */
export function useLineOfTheDay(nocache?: boolean) {
  return useQuery<LineOfTheDayResponse>({
    queryKey: ['line-of-the-day', nocache],
    queryFn: () => fetchJson<LineOfTheDayResponse>(
      nocache ? '/api/line-of-the-day?nocache=1' : '/api/line-of-the-day',
      90_000,
    ),
    staleTime: 5 * 60 * 1000, // 5 min
    refetchOnWindowFocus: false,
  });
}

// ---------------------------------------------------------------------------
// /api/line-history
// ---------------------------------------------------------------------------

/**
 * Recent line picks with streak and hit rate (resolved picks only).
 *
 * - 15s timeout
 * - staleTime 3 min
 */
export function useLineHistory() {
  return useQuery<LineHistoryResponse>({
    queryKey: ['line-history'],
    queryFn: () => fetchJson<LineHistoryResponse>('/api/line-history', 15_000),
    staleTime: 3 * 60 * 1000, // 3 min
  });
}

// ---------------------------------------------------------------------------
// /api/line-live-stat
// ---------------------------------------------------------------------------

/**
 * Fetch live in-game stat value for a pick.
 *
 * - Polls every 60s while enabled
 * - staleTime 30s so rapid re-renders don't trigger extra fetches
 * - Enabled only when a pick is provided
 */
export function useLineLiveStat(pick: LinePick | null) {
  return useQuery({
    queryKey: ['line-live-stat', pick?.player_id, pick?.stat_type],
    queryFn: async () => {
      if (!pick) return null;
      const p = new URLSearchParams({
        team: pick.team || '',
        stat_type: pick.stat_type || 'points',
        player_name: pick.player_name || '',
      });
      if (pick.player_id) p.set('player_id', pick.player_id);
      return fetchJson(`/api/line-live-stat?${p}`, 20_000);
    },
    enabled: !!pick,
    refetchInterval: 60_000, // poll every 60s
    staleTime: 30_000, // 30s
  });
}

// ---------------------------------------------------------------------------
// GET /api/refresh-line-odds (mutation — triggers side-effect)
// ---------------------------------------------------------------------------

/**
 * Mutation: refresh bookmaker odds for today's line picks.
 * Technically a GET that triggers a side-effect (odds update),
 * modeled as a mutation so callers get loading/error state.
 */
export function useRefreshLineOdds() {
  return useMutation({
    mutationFn: async () => {
      const r = await fetchWithTimeout('/api/refresh-line-odds', {}, 30_000);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    },
  });
}
