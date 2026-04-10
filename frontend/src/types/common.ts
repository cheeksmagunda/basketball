// ============================================================================
// common.ts — Shared types for the Basketball Oracle frontend
// ============================================================================

// ---------------------------------------------------------------------------
// Async data loading state (generic wrapper for API responses)
// ---------------------------------------------------------------------------

export type AsyncStatus = 'idle' | 'loading' | 'success' | 'error';

export interface AsyncState<T> {
  status: AsyncStatus;
  data: T | null;
  error: string | null;
  loadedAt: number | null; // Date.now() timestamp of last successful load
}

/** Factory: create an idle AsyncState */
export function idleState<T>(): AsyncState<T> {
  return { status: 'idle', data: null, error: null, loadedAt: null };
}

// ---------------------------------------------------------------------------
// Navigation types
// ---------------------------------------------------------------------------

/** Predict sub-nav: Slate-Wide vs Per-Game */
export type PredictSubNav = 'slate' | 'game';

/** Slate lineup mode: Starting 5 (chalk) vs Moonshot (upside) */
export type SlateMode = 'chalk' | 'upside';


// ---------------------------------------------------------------------------
// Team info (from ESPN via fetch_games)
// ---------------------------------------------------------------------------

export interface TeamInfo {
  id: string;
  name: string;
  abbr: string;
}

// ---------------------------------------------------------------------------
// Score bounds (returned on slate + per-game responses)
// ---------------------------------------------------------------------------

export interface ScoreBounds {
  total: number;
  lo: number;
  hi: number;
  in_range: boolean;
}

export type ScoreBoundsMap = Record<string, ScoreBounds>;

// ---------------------------------------------------------------------------
// API error shape (common across endpoints)
// ---------------------------------------------------------------------------

export interface ApiError {
  error: string;
  detail?: string;
}

// ---------------------------------------------------------------------------
// Cache metadata (appended by response cache layer)
// ---------------------------------------------------------------------------

export interface CacheMetadata {
  cache_status?: 'hit' | 'miss';
  _cached_at?: string;   // ISO 8601
  _cache_date?: string;  // YYYY-MM-DD
}
