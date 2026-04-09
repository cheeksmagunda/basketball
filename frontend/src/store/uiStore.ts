// ============================================================================
// UI Store — Zustand
// Global navigation and view state for the Oracle app.
// ============================================================================

import { create } from 'zustand';
import type {
  TabName,
  PredictSub,
  SlateMode,
} from '../types';

interface UiState {
  /** Active top-level tab */
  activeTab: TabName;
  /** Predict sub-tab: slate-wide or per-game */
  predictSub: PredictSub;
  /** Slate-wide lineup mode: chalk (Starting 5) or upside (Moonshot) */
  slateMode: SlateMode;

  // Setters
  setActiveTab: (tab: TabName) => void;
  setPredictSub: (sub: PredictSub) => void;
  setSlateMode: (mode: SlateMode) => void;
}

export const useUiStore = create<UiState>((set) => ({
  activeTab: 'predictions',
  predictSub: 'slate',
  slateMode: 'chalk',

  setActiveTab: (tab) => set({ activeTab: tab }),
  setPredictSub: (sub) => set({ predictSub: sub }),
  setSlateMode: (mode) => set({ slateMode: mode }),
}));
