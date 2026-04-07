// ============================================================================
// slate.ts — Types for /api/slate, /api/picks, /api/games
// ============================================================================

import type { TeamInfo, ScoreBoundsMap, CacheMetadata } from './common';

// ---------------------------------------------------------------------------
// GameInfo — one NBA game on today's slate (from fetch_games / /api/games)
// ---------------------------------------------------------------------------

export interface GameInfo {
  gameId: string;
  label: string;              // e.g. "BOS @ LAL"
  home: TeamInfo;
  away: TeamInfo;
  spread: number | null;      // ESPN or Odds API spread
  total: number | null;       // ESPN or Odds API over/under
  startTime: string;          // ISO 8601 UTC
  home_b2b?: boolean;
  away_b2b?: boolean;
  /** Added client-side by populateGameSelector */
  locked?: boolean;
  /** False once past the lock window — permanent (no 6h ceiling). Use this for UI lock icons. */
  draftable?: boolean;
  _odds_source?: string;      // "espn" | "odds_api"
}

// ---------------------------------------------------------------------------
// PlayerCard — a projected player in a lineup
// Matches _normalize_player() contract in api/index.py
// ---------------------------------------------------------------------------

export interface PlayerCard {
  id: string;
  name: string;
  pos: string;                // e.g. "PG", "SF", "C"
  team: string;               // 3-letter abbr
  rating: number;             // projected Real Score
  predMin: number;            // projected minutes
  pts: number;
  reb: number;
  ast: number;
  stl: number;
  blk: number;
  est_mult: number;           // card boost (0.0–3.0; zeroed for per-game)
  slot: string;               // "2.0x", "1.8x", "1.6x", "1.4x", "1.2x"
  draft_ev: number;           // RS x (2.0 + boost)
  chalk_ev: number;
  moonshot_ev: number;
  injury_status: string;      // "", "GTD", "DTD", "DOUBT"
  _decline: number;

  // Optional extras from model / enrichment (pass-through fields)
  season_pts?: number;
  season_reb?: number;
  season_ast?: number;
  season_stl?: number;
  season_blk?: number;
  season_min?: number;
  avg_min?: number;
  recent_pts?: number;
  opp?: string;               // opponent team abbr
  _hot_streak?: boolean;
  _odds_adjusted?: boolean;
  _context_adj?: number;
  _is_value_anchor?: boolean;
  _favored_team?: boolean;
  _pre_boost_rs?: number;
  variance?: number;
  cascade_bonus?: number;
  roto_status?: string;
}

// ---------------------------------------------------------------------------
// WatchlistPlayer — near-bubble player sensitive to late-breaking news
// ---------------------------------------------------------------------------

export interface WatchlistPlayer {
  name: string;
  team: string;
  rating: number;
  est_mult: number;
  reason: string;
}

// ---------------------------------------------------------------------------
// SlateData — response from GET /api/slate
// ---------------------------------------------------------------------------

export interface SlateLineups {
  chalk: PlayerCard[];
  upside: PlayerCard[];
}

export interface SlateData extends CacheMetadata {
  date: string;                       // YYYY-MM-DD
  locked: boolean;
  all_complete: boolean;
  lock_time: string | null;           // ISO 8601 or null
  error?: string;                     // "slate_failed" on pipeline error
  warming_up?: boolean;               // true while cold pipeline is running — poll and retry
  no_games?: boolean;
  next_slate_date?: string | null;    // YYYY-MM-DD when no games today
  games: GameInfo[];
  lineups: SlateLineups;
  draftable_count: number;
  watchlist?: WatchlistPlayer[];
  score_bounds?: ScoreBoundsMap;
  deploy_sha?: string;
  pass?: number;                      // pipeline pass number
}

// ---------------------------------------------------------------------------
// Per-Game Strategy (from _per_game_strategy in api/index.py)
// ---------------------------------------------------------------------------

export type PerGameStrategyType =
  | 'balanced'
  | 'neutral'
  | 'blowout_lean'
  | 'shootout'
  | 'defensive_grind';

export interface PerGameStrategy {
  type: PerGameStrategyType;
  label: string;              // e.g. "Balanced Build", "Blowout Lean"
  description: string;
  total_mult?: number;
  spread?: number;
  total?: number;
  favored_team?: string;
  underdog_team?: string;
}

// ---------------------------------------------------------------------------
// PicksData — response from GET /api/picks?gameId=X
// ---------------------------------------------------------------------------

export interface PicksLineups {
  the_lineup: PlayerCard[];
}

export interface PicksData {
  date: string;
  locked: boolean;
  gameScript: string | null;          // "balanced", "shootout", etc.
  strategy: PerGameStrategy | null;
  lineups: PicksLineups;
  game: GameInfo;
  injuries?: Array<{ name: string; team: string; status: string }>;
  score_bounds?: ScoreBoundsMap;
  mock?: boolean;
}

// ---------------------------------------------------------------------------
// GamesResponse — response from GET /api/games
// ---------------------------------------------------------------------------

export type GamesResponse = GameInfo[];
