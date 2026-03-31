// ============================================================================
// line.ts — Types for Line of the Day (over/under player prop picks)
// ============================================================================

import type { StatType, PickResult, LineDirection } from './common';

// ---------------------------------------------------------------------------
// LinePick — a single over/under player prop pick
// Matches _normalize_line_pick() contract in api/index.py
// ---------------------------------------------------------------------------

export interface LinePick {
  player_name: string;
  player_id: string;
  team: string;                       // 3-letter abbr
  opponent: string;
  direction: LineDirection;
  line: number;                       // sportsbook line (snapped to nearest 0.5)
  stat_type: StatType;
  projection: number;                 // model projection for the stat
  edge: number;                       // projection - line (over) or line - projection (under)
  confidence: number;                 // 0–100 integer score
  signals: Array<{ type: string; detail: string }>; // driver signal objects from backend
  result: PickResult;
  actual_stat: number | null;         // post-game actual (null when pending)
  date: string;                       // YYYY-MM-DD
  narrative?: string;                 // model reasoning paragraph

  // Odds / books data (populated by /api/refresh-line-odds or inline enrichment)
  line_updated_at: string | null;     // ISO 8601 or null
  odds_over: number | null;           // American odds e.g. -115
  odds_under: number | null;          // American odds e.g. -105
  books_consensus: number | null;     // consensus line from books

  // Player context (populated at generation time, stored in GitHub JSON)
  season_avg: number | null;
  proj_min: number | null;            // projected minutes
  avg_min: number | null;             // season average minutes
  game_time: string;                  // e.g. "7:30 PM ET" or ""
  game_start_iso: string;             // ISO 8601 or ""

  // Recent form (L5 game values)
  recent_form_bars: number[] | null;  // ratio vs season average (e.g. [0.8, 1.1, 0.9])
  recent_form_values: number[] | null; // raw stat values (e.g. [22, 28, 23])

  // Model-only flag (no real odds available)
  model_only?: boolean;
}

// ---------------------------------------------------------------------------
// Slate summary (returned when picks are freshly generated)
// ---------------------------------------------------------------------------

export interface LineSlateInfo {
  games_evaluated: number;
  props_scanned: number;
  edges_found: number;
  timestamp: string;
  model_only?: boolean;
}

// ---------------------------------------------------------------------------
// LineOfTheDayResponse — response from GET /api/line-of-the-day
// ---------------------------------------------------------------------------

export interface LineOfTheDayResponse {
  pick: LinePick | null;              // primary (highest confidence of over/under)
  over_pick: LinePick | null;
  under_pick: LinePick | null;

  // Transition state
  next_slate_pending?: boolean;       // true when resolved + next-slate not yet generated
  final_over?: LinePick | null;       // the pick to display (may be next-slate)
  final_under?: LinePick | null;

  // Meta
  summary?: string;
  slate_summary?: LineSlateInfo | null;
  error?: string;                     // "server_error", "no_projections", etc.
  source?: string;                    // "cache", "github_saved", "fresh"
  generated_at?: string;              // ISO 8601
  is_stale?: boolean;
  refreshing?: boolean;
  mock?: boolean;
}

// ---------------------------------------------------------------------------
// LineLiveStatResponse — response from GET /api/line-live-stat
// ---------------------------------------------------------------------------

export interface LineLiveStatResponse {
  status: 'live' | 'final' | 'pre' | 'unavailable';
  stat_current: number | null;
  stat_type: string;
  period: number;
  clock: string;
  pace?: number;
  game_id?: string;
}

// ---------------------------------------------------------------------------
// LineHistoryResponse — response from GET /api/line-history
// ---------------------------------------------------------------------------

export interface LineHistoryResponse {
  picks: LinePick[];
  hit_rate: number | null;            // percentage (e.g. 66.7) or null if none resolved
  total_picks: number;
  resolved: number;
  streak: number;
  streak_type: 'hit' | 'miss' | null;
}
