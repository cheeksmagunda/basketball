// ============================================================================
// lab.ts — Types for the Ben (Lab) chat interface and related endpoints
// ============================================================================

// ---------------------------------------------------------------------------
// LabMessage — a single message in the Ben chat thread
// ---------------------------------------------------------------------------

export interface LabMessage {
  role: 'user' | 'assistant';
  content: string;
  hidden?: boolean;         // hidden system messages (briefing context, etc.)
  isStatus?: boolean;       // status indicator message (e.g. "Thinking...")
  imageSrc?: string;        // base64 or URL for image messages (screenshot uploads)
}

// ---------------------------------------------------------------------------
// LabStatus — response from GET /api/lab/status
// ---------------------------------------------------------------------------

export interface LabStatus {
  locked: boolean;
  reason: string;
  current_config_version?: number;
  games_remaining?: number;
  games_final?: number;
  next_lock_time?: string | null;    // ISO 8601 or null
  estimated_unlock?: string | null;  // ISO 8601 (only when locked)
}

// ---------------------------------------------------------------------------
// LabBriefing — response from GET /api/lab/briefing
// ---------------------------------------------------------------------------

export interface LabBriefingMiss {
  name: string;
  predicted: number;
  actual: number;
  error: number;
}

export interface LabBriefingSlate {
  date: string;
  players_with_actuals: number;
  mean_absolute_error: number;
  directional_accuracy: number | null;
  over_projected: number;
  under_projected: number;
  biggest_misses: LabBriefingMiss[];
  simulated_draft_score?: number | null;
}

export interface LabBriefingPattern {
  type: string;                      // "high_error", "systematic_over_projection"
  description: string;
  slates_observed: number;
}

export interface LabBriefing {
  latest_slate: LabBriefingSlate | null;
  rolling_accuracy: {
    slates_with_data: number;
    overall_mae: number | null;
  };
  patterns: LabBriefingPattern[];
  current_config: {
    version: number;
    last_change: string;
    last_change_date: string;
  };
  pending_upload_date: string | null;
  pending_historical_date: string | null;
  ownership_calibration_available: boolean;
  ownership_dates: string[];
}

// ---------------------------------------------------------------------------
// LabConfigHistory — response from GET /api/lab/config-history
// ---------------------------------------------------------------------------

export interface LabConfigChangelogEntry {
  version?: number;
  date?: string;
  change?: string;
  description?: string;
  changes?: Record<string, unknown>;
}

export interface LabConfigHistory {
  version: number;
  config: Record<string, unknown>;   // full model-config.json
  changelog: LabConfigChangelogEntry[];
}

// ---------------------------------------------------------------------------
// LabUpdateConfigPayload — request body for POST /api/lab/update-config
// ---------------------------------------------------------------------------

export interface LabUpdateConfigPayload {
  changes: Record<string, unknown>;   // dot-notation keys => values
  change_description?: string;
}

export interface LabUpdateConfigResponse {
  status: string;
  version: number;
  changes_applied: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// LabBacktestPayload — request body for POST /api/lab/backtest
// ---------------------------------------------------------------------------

export interface LabBacktestPayload {
  changes: Record<string, unknown>;
  dates?: string[];                   // specific dates to backtest
}

export interface LabBacktestResult {
  current_mae: number;
  proposed_mae: number;
  improvement_pct: number;
  dates_tested: number;
  per_date: Array<{
    date: string;
    current_mae: number;
    proposed_mae: number;
  }>;
}

// ---------------------------------------------------------------------------
// LabChatPayload — request body for POST /api/lab/chat
// ---------------------------------------------------------------------------

export interface LabChatPayload {
  messages: LabMessage[];
  system?: string;
}

// ---------------------------------------------------------------------------
// LabChatHistory — response from GET /api/lab/chat-history
// ---------------------------------------------------------------------------

export type LabChatHistory = LabMessage[];
