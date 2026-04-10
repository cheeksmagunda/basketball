// ============================================================================
// types/index.ts — Barrel re-export of all shared types
// ============================================================================

// Common / shared primitives
export type {
  AsyncStatus,
  AsyncState,
  PredictSubNav,
  SlateMode,
  TeamInfo,
  ScoreBounds,
  ScoreBoundsMap,
  ApiError,
  CacheMetadata,
} from './common';

export { idleState } from './common';

// Navigation aliases (used by stores and hooks)
export type PredictSub = 'slate' | 'game';

// Slate / Predict
export type {
  GameInfo,
  PlayerCard,
  WatchlistPlayer,
  SlateLineups,
  SlateData,
  PerGameStrategyType,
  PerGameStrategy,
  PicksLineups,
  PicksData,
  GamesResponse,
} from './slate';
