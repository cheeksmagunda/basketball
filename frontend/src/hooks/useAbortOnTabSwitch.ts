// ============================================================================
// useAbortOnTabSwitch
// Provides an AbortSignal that fires when the user navigates away from
// the specified tab. Mirrors the vanilla JS _abortTab() pattern.
// ============================================================================

import { useRef, useEffect, useCallback } from 'react';
import { useUiStore } from '../store/uiStore';
import type { TabName } from '../types';

/**
 * Creates an AbortController scoped to a tab. When activeTab changes away
 * from `tabName`, the controller is aborted so in-flight fetches are
 * cancelled (preventing stale response processing on the wrong tab).
 *
 * Usage:
 *   const { getSignal } = useAbortOnTabSwitch('line');
 *   // Pass getSignal() as the externalSignal arg to fetchWithTimeout
 */
export function useAbortOnTabSwitch(tabName: TabName) {
  const controllerRef = useRef<AbortController | null>(null);
  const activeTab = useUiStore((s) => s.activeTab);

  // When the active tab moves away from our tab, abort any pending controller.
  useEffect(() => {
    if (activeTab !== tabName && controllerRef.current) {
      controllerRef.current.abort();
      controllerRef.current = null;
    }
  }, [activeTab, tabName]);

  /**
   * Returns the current AbortSignal for this tab.
   * Creates a fresh controller if the previous one was aborted or absent.
   */
  const getSignal = useCallback((): AbortSignal => {
    if (!controllerRef.current || controllerRef.current.signal.aborted) {
      controllerRef.current = new AbortController();
    }
    return controllerRef.current.signal;
  }, []);

  return { getSignal } as const;
}
