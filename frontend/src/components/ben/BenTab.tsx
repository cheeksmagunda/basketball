// ============================================================================
// BenTab — Container for the Ben/Lab chat interface
// ============================================================================

import { useEffect, useRef } from 'react';
import ChatMessages from './ChatMessages';
import ChatInput from './ChatInput';
import { useLabStore } from '../../store/labStore';
import { useLabBriefing } from '../../api/lab';
import { useSlate } from '../../api/slate';
import { useKeyboardNavHide } from '../../hooks/useKeyboardNavHide';
import type { LabBriefing, SlateData, PlayerCard } from '../../types';
import styles from './BenTab.module.css';

export default function BenTab() {
  const { initialized, setInitialized, setSystem, addMessage } = useLabStore();
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const briefingQuery = useLabBriefing();
  const slateQuery = useSlate();

  useKeyboardNavHide(inputRef);

  // Remove .app bottom padding while Ben is mounted — Ben uses a fixed-height
  // flex layout that accounts for the nav, so the 140px clearance causes overflow.
  useEffect(() => {
    const app = document.querySelector('.app');
    if (app) app.classList.add('ben-active');
    return () => {
      const appEl = document.querySelector('.app');
      if (appEl) appEl.classList.remove('ben-active');
    };
  }, []);

  // When briefing data arrives and we haven't initialized, inject system context + visible greeting
  useEffect(() => {
    if (initialized || !briefingQuery.data) return;
    setInitialized(true);

    const briefingText = buildSystemPrompt(briefingQuery.data, slateQuery.data);
    setSystem(briefingText);
    addMessage({ role: 'assistant', content: briefingText, hidden: true });
  }, [initialized, briefingQuery.data, slateQuery.data, setInitialized, setSystem, addMessage]);

  // Fallback: if briefing fails, still mark initialized so empty state shows
  useEffect(() => {
    if (initialized || !briefingQuery.isError) return;
    setInitialized(true);
  }, [initialized, briefingQuery.isError, setInitialized]);

  // Update system prompt when briefing or slate data refreshes
  useEffect(() => {
    if (briefingQuery.data) {
      setSystem(buildSystemPrompt(briefingQuery.data, slateQuery.data));
    }
  }, [briefingQuery.data, slateQuery.data, setSystem]);

  return (
    <div className={styles.container}>
      <div className={styles.chatArea}>
        <ChatMessages />
      </div>
      <div className={styles.inputWrap}>
        <ChatInput inputRef={inputRef} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Build the system prompt from briefing data (matches vanilla JS buildLabSystemPrompt)
// ---------------------------------------------------------------------------

function formatPlayer(p: PlayerCard, i: number): string {
  const boost = p.est_mult > 0 ? ` +${p.est_mult}x card` : '';
  const cascade = p.cascade_bonus && p.cascade_bonus > 0 ? ` (cascade +${p.cascade_bonus} min)` : '';
  return `  ${i + 1}. ${p.name} (${p.team} ${p.pos}) — RS ${p.rating}, ${p.slot} slot, ${p.predMin} min proj (${p.season_min ?? p.avg_min ?? '?'} avg)${boost}${cascade} | ${p.pts}/${p.reb}/${p.ast}/${p.stl}/${p.blk} PTS/REB/AST/STL/BLK`;
}

function buildSystemPrompt(briefing: LabBriefing, slate?: SlateData): string {
  const lines: string[] = [
    'You are Ben, the Basketball Oracle\'s AI assistant.',
    'You help analyze NBA prediction accuracy, tune model parameters, and run backtests.',
    '',
    '## Current Briefing',
    JSON.stringify(briefing, null, 2),
  ];

  if (slate && !slate.error && slate.lineups) {
    lines.push('');
    lines.push(`## Tonight's Slate (${slate.date}) — ${slate.games?.length ?? 0} games${slate.locked ? ' [LOCKED]' : ''}`);

    if (slate.lineups.chalk?.length) {
      lines.push('');
      lines.push('### Starting 5 (Safe lineup)');
      slate.lineups.chalk.forEach((p, i) => lines.push(formatPlayer(p, i)));
    }

    if (slate.lineups.upside?.length) {
      lines.push('');
      lines.push('### Moonshot (Upside lineup)');
      slate.lineups.upside.forEach((p, i) => lines.push(formatPlayer(p, i)));
    }

    if (slate.games?.length) {
      lines.push('');
      lines.push('### Games');
      slate.games.forEach(g => {
        const spread = g.spread != null ? ` (spread ${g.spread})` : '';
        const total = g.total != null ? ` (total ${g.total})` : '';
        lines.push(`  - ${g.label}${spread}${total}`);
      });
    }
  }

  lines.push('');
  lines.push('Answer concisely. Use data when available. You can suggest config changes, backtests, and analysis.');
  lines.push('When asked about tonight\'s players or picks, reference the slate data above. You KNOW the current Starting 5 and Moonshot lineups.');
  return lines.join('\n');
}
