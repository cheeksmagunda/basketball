// ============================================================================
// Lab / Ben API — React Query hooks
// Mirrors: /api/lab/status, /api/lab/briefing, /api/lab/config-history
// ============================================================================

import { useQuery } from '@tanstack/react-query';
import { fetchJson } from './client';
import type {
  LabStatusResponse,
  LabBriefingResponse,
  LabConfigHistoryResponse,
} from '../types';

// ---------------------------------------------------------------------------
// /api/lab/status
// ---------------------------------------------------------------------------

/**
 * Lab lock status (locked during slate, unlocked after all games final).
 *
 * Used internally for conditional data loading (briefing, etc.) and
 * for legacy lock-poll paths. Ben chat itself is always available.
 *
 * - staleTime 2 min (matches vanilla JS poll interval)
 */
export function useLabStatus() {
  return useQuery<LabStatusResponse>({
    queryKey: ['lab-status'],
    queryFn: () => fetchJson<LabStatusResponse>('/api/lab/status', 10_000),
    staleTime: 2 * 60 * 1000, // 2 min
  });
}

// ---------------------------------------------------------------------------
// /api/lab/briefing
// ---------------------------------------------------------------------------

/**
 * Prediction accuracy analysis (MAE, biggest misses, patterns).
 *
 * - 30s timeout (briefing can involve GitHub reads)
 * - Only enabled when not locked (no point loading stale briefing)
 */
export function useLabBriefing(enabled: boolean = true) {
  return useQuery<LabBriefingResponse>({
    queryKey: ['lab-briefing'],
    queryFn: () => fetchJson<LabBriefingResponse>('/api/lab/briefing', 30_000),
    enabled,
    staleTime: 5 * 60 * 1000, // 5 min
  });
}

// ---------------------------------------------------------------------------
// /api/lab/config-history
// ---------------------------------------------------------------------------

/**
 * Full config + changelog for Lab context.
 */
export function useLabConfigHistory() {
  return useQuery<LabConfigHistoryResponse>({
    queryKey: ['lab-config-history'],
    queryFn: () => fetchJson<LabConfigHistoryResponse>('/api/lab/config-history', 10_000),
    staleTime: 5 * 60 * 1000, // 5 min
  });
}
