// ============================================================================
// types/index.ts — Barrel re-export of all shared types
// ============================================================================

// Common / shared primitives
export type {
  AsyncStatus,
  AsyncState,
  Tab,
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
// Note: Tab in common.ts uses 'predict' but the vanilla JS app uses 'predictions'.
// TabName matches the actual DOM tab IDs (tab-predictions, tab-lab)
export type TabName = 'predictions' | 'lab';
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


// Lab / Ben
export type {
  LabMessage,
  LabStatus,
  LabBriefingMiss,
  LabBriefingSlate,
  LabBriefingPattern,
  LabBriefing,
  LabConfigChangelogEntry,
  LabConfigHistory,
  LabUpdateConfigPayload,
  LabUpdateConfigResponse,
  LabBacktestPayload,
  LabBacktestResult,
  LabChatPayload,
  LabChatHistory,
} from './lab';

// Convenience aliases (used by API hooks)
export type { LabStatus as LabStatusResponse } from './lab';
export type { LabBriefing as LabBriefingResponse } from './lab';
export type { LabConfigHistory as LabConfigHistoryResponse } from './lab';

// ---------------------------------------------------------------------------
// Pending image (Lab camera upload) — defined here since it's UI-only state
// ---------------------------------------------------------------------------

export interface PendingImage {
  base64: string;
  mediaType: string;
  dataUrl: string;
}
