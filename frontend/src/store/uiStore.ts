// ============================================================================
// UI Store — Zustand
// Global navigation and view state for the Oracle app.
// ============================================================================

import { create } from 'zustand';
import type {
  PredictSub,
  SlateMode,
} from '../types';

interface UiState {
  /** Predict sub-tab: slate-wide or per-game */
  predictSub: PredictSub;
  /** Slate-wide lineup mode: chalk (Starting 5) or upside (Moonshot) */
  slateMode: SlateMode;

  // Setters
  setPredictSub: (sub: PredictSub) => void;
  setSlateMode: (mode: SlateMode) => void;
}

export const useUiStore = create<UiState>((set) => ({
  predictSub: 'slate',
  slateMode: 'chalk',

  setPredictSub: (sub) => set({ predictSub: sub }),
  setSlateMode: (mode) => set({ slateMode: mode }),
}));
