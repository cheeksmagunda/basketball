// ============================================================================
// Lab (Ben) Store — Zustand
// Chat messages, system prompt, pending image, and session date.
// Mirrors the vanilla JS LAB global object.
// ============================================================================

import { create } from 'zustand';
import type { LabMessage, PendingImage } from '../types';

interface LabState {
  /** Chat message history (user + assistant + hidden system messages) */
  messages: LabMessage[];
  /** System prompt built from briefing + config context */
  system: string;
  /** Camera/image upload pending attachment */
  pendingImage: PendingImage | null;
  /** ET date string when Lab was last initialized (clears messages on rollover) */
  initDate: string;
  /** Whether Lab has been initialized this session */
  initialized: boolean;

  // Actions
  addMessage: (msg: LabMessage) => void;
  clearMessages: () => void;
  setPendingImage: (img: PendingImage | null) => void;
  setSystem: (prompt: string) => void;
  setInitDate: (date: string) => void;
  setInitialized: (val: boolean) => void;
}

export const useLabStore = create<LabState>((set) => ({
  messages: [],
  system: '',
  pendingImage: null,
  initDate: '',
  initialized: false,

  addMessage: (msg) =>
    set((state) => ({ messages: [...state.messages, msg] })),

  clearMessages: () => set({ messages: [] }),

  setPendingImage: (img) => set({ pendingImage: img }),

  setSystem: (prompt) => set({ system: prompt }),

  setInitDate: (date) => set({ initDate: date }),

  setInitialized: (val) => set({ initialized: val }),
}));
