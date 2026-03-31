// ============================================================================
// UI Store — Zustand
// Global navigation and view state for the Oracle app.
// Mirrors the vanilla JS globals: activeTab, predictSub, slateMode,
// LINE_DIR, lineHistoryFilter.
// ============================================================================

import { create } from 'zustand';
import type {
  TabName,
  PredictSub,
  SlateMode,
  LineDirection,
  LineHistoryFilter,
} from '../types';

interface UiState {
  /** Active top-level tab */
  activeTab: TabName;
  /** Predict sub-tab: slate-wide or per-game */
  predictSub: PredictSub;
  /** Slate-wide lineup mode: chalk (Starting 5) or upside (Moonshot) */
  slateMode: SlateMode;
  /** Line direction filter: over or under */
  lineDir: LineDirection;
  /** Line history section filter: all, over, or under */
  lineHistoryFilter: LineHistoryFilter;

  // Setters
  setActiveTab: (tab: TabName) => void;
  setPredictSub: (sub: PredictSub) => void;
  setSlateMode: (mode: SlateMode) => void;
  setLineDir: (dir: LineDirection) => void;
  setLineHistoryFilter: (filter: LineHistoryFilter) => void;
}

export const useUiStore = create<UiState>((set) => ({
  activeTab: 'predictions',
  predictSub: 'slate',
  slateMode: 'chalk',
  lineDir: 'under',
  lineHistoryFilter: 'all',

  setActiveTab: (tab) => set({ activeTab: tab }),
  setPredictSub: (sub) => set({ predictSub: sub }),
  setSlateMode: (mode) => set({ slateMode: mode }),
  setLineDir: (dir) =>
    set({
      lineDir: dir,
      lineHistoryFilter: dir === 'over' || dir === 'under' ? dir : 'all',
    }),
  setLineHistoryFilter: (filter) => set({ lineHistoryFilter: filter }),
}));
