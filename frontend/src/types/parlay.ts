// ============================================================================
// parlay.ts — Types for the 3-leg parlay engine + history
// ============================================================================

import type { StatType, PickResult, LineDirection } from './common';

// ---------------------------------------------------------------------------
// ParlayLeg — one leg of a 3-leg parlay (from run_parlay_engine candidates)
// ---------------------------------------------------------------------------

export interface ParlayLeg {
  player_name: string;
  player_id: string;
  team: string;
  opponent: string;
  gameId: string;
  stat_type: StatType;
  direction: LineDirection;
  line: number;
  projection: number;
  edge: number;
  std_dev: number;
  z_score: number;
  model_hit_prob: number;             // raw model probability
  vegas_implied_prob: number | null;  // from American odds conversion
  blended_confidence: number;         // 55% model + 45% Vegas blend
  american_odds: number | null;       // e.g. -140, +150
  minutes_cv: number;                 // coefficient of variation for minutes
  season_avg: number;
  recent_values: number[];            // last 5 game stat values
  avg_min: number;
  season_pts: number;
  season_ast: number;
  season_reb: number;
  position: string;                   // e.g. "PG", "C"
  game_spread: number | null;
  game_total: number | null;
  game_time: string;
  home_team: string;
  away_team: string;
  is_b2b: boolean;
  opp_b2b: boolean;

  // Resolution fields (added post-game by parlay-history resolution)
  result?: PickResult;
  actual_stat?: number | null;
}

// ---------------------------------------------------------------------------
// ParlayData — response from GET /api/parlay
// ---------------------------------------------------------------------------

export interface ParlayData {
  legs: ParlayLeg[];
  combined_probability: number;       // product of blended confidences
  correlation_multiplier: number;     // correlation * structure bonus
  correlation_reasons: string[];
  parlay_score: number;               // combined_prob * corr_mult
  narrative: string;
  projection_only: boolean;           // true when Odds API unavailable

  // Meta
  locked?: boolean;
  date?: string;
  odds_available?: boolean;
  candidates_evaluated?: number;
  combinations_scored?: number;
  error?: string;                     // "no_valid_parlay", "odds_required", etc.
  debug?: Record<string, unknown>;
  market_matches?: ParlayMarketMatch[];
}

// ---------------------------------------------------------------------------
// ParlayMarketMatch — Vegas-aligned legs shown as additional context
// ---------------------------------------------------------------------------

export interface ParlayMarketMatch {
  player_name: string;
  stat_type: StatType;
  direction: LineDirection;
  line: number;
  blended_confidence: number;
  american_odds: number | null;
}

// ---------------------------------------------------------------------------
// ParlayLivePayload — SSE tick from /api/parlay-live-stream
// ---------------------------------------------------------------------------

export interface ParlayLiveLeg {
  leg_index: number;
  player_name: string;
  line: number | null;
  direction: LineDirection;
  stat_type?: string;
  status: 'live' | 'final' | 'pre' | 'unavailable';
  stat_current: number | null;
  period: number;
  clock: string;
  pace?: number;
  game_id?: string;
  leg_result_preview: PickResult | null;
  progress?: number;                  // 0.0–1.0 bar fill
  hit_threshold_met?: boolean;
}

export interface ParlayLivePayload {
  date: string;
  legs: ParlayLiveLeg[];
  all_games_final: boolean;
  no_ticket: boolean;
  error?: string;
}

// ---------------------------------------------------------------------------
// ParlayHistoryItem — one historical parlay from /api/parlay-history
// ---------------------------------------------------------------------------

export interface ParlayHistoryItem {
  date: string;
  result: PickResult;
  legs: ParlayLeg[];
  combined_probability: number;
  correlation_multiplier?: number;
  correlation_reasons?: string[];
  parlay_score?: number;
  narrative?: string;
  projection_only?: boolean;
  generated_at?: string;
}

// ---------------------------------------------------------------------------
// ParlayHistoryResponse — response from GET /api/parlay-history
// ---------------------------------------------------------------------------

export interface ParlayHistoryResponse {
  parlays: ParlayHistoryItem[];
  hit_rate: number | null;            // percentage or null
  total: number;
  resolved: number;
  streak: number;
  streak_type: 'hit' | 'miss' | null;
  error?: string;
  narrative?: string;
}
