// ============================================================================
// GameView.tsx — Per-game analysis view
// 2-column game card grid → header row + THE LINE UP (5-player optimal lineup).
// ============================================================================

import { useState, useCallback, useEffect } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { useGames, usePicks } from '../../api/slate';
import { fetchJson } from '../../api/client';
import type { PicksData } from '../../types';
import PlayerCard from '../shared/PlayerCard';
import SkeletonCard from '../shared/SkeletonCard';
import EmptyState from '../shared/EmptyState';
import StrategyInsight from './StrategyInsight';
import { TEAM_COLORS } from '../../utils/teamColors';
import styles from './GameView.module.css';

/** Parse "BOS @ LAL" or "BOS vs LAL" into [away, home] */
function parseLabel(label: string): [string, string] {
  const parts = label.split(/ vs | @ /);
  return [parts[0] || '?', parts[1] || '?'];
}

/** Format ISO time string to local time (e.g. "7:30 PM") */
function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
  });
}

export default function GameView() {
  const { data: games, isLoading: gamesLoading } = useGames();
  const [selectedGame, setSelectedGame] = useState<string | null>(null);
  const {
    data: picks,
    isLoading: picksLoading,
    error: picksError,
    refetch,
  } = usePicks(selectedGame);

  const handleSelectGame = useCallback((gameId: string) => {
    setSelectedGame(gameId);
  }, []);

  const handleBack = useCallback(() => {
    setSelectedGame(null);
  }, []);

  // Prefetch all game picks as soon as the games list loads so card clicks are instant.
  const queryClient = useQueryClient();
  useEffect(() => {
    if (!games || games.length === 0) return;
    for (const g of games) {
      queryClient.prefetchQuery({
        queryKey: ['picks', g.gameId],
        queryFn: () => fetchJson<PicksData>(`/api/picks?gameId=${g.gameId}`, 15_000),
        staleTime: 5 * 60 * 1000,
      });
    }
  }, [games, queryClient]);

  // Loading games list
  if (gamesLoading) return <SkeletonCard count={1} />;

  // ── VIEW A: Game card grid (no game selected) ──
  if (!selectedGame) {
    if (!games || games.length === 0) {
      return (
        <EmptyState
          icon={'\uD83C\uDFC0'}
          message="No games available today."
        />
      );
    }

    return (
      <div className={styles.grid}>
        {games.map((g) => {
          const [away, home] = parseLabel(g.label);
          const timeStr = g.startTime ? formatTime(g.startTime) : '';
          const totalStr = g.total ? `O/U ${g.total}` : '';
          const meta = [timeStr, totalStr].filter(Boolean).join(' \u00b7 ');

          return (
            <div
              key={g.gameId}
              className={styles.gameCard}
              role="button"
              tabIndex={0}
              onClick={() => handleSelectGame(g.gameId)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  handleSelectGame(g.gameId);
                }
              }}
            >
              <div className={styles.gameCardTeams}>
                {away} <span className={styles.gameCardVs}>vs</span> {home}
              </div>
              <div className={styles.gameCardMeta}>
                {meta || '\u00a0'}
                {g.locked ? ' \uD83D\uDD12' : ''}
              </div>
            </div>
          );
        })}
      </div>
    );
  }

  // ── VIEW B: Analysis view (game selected) ──
  const lineup = picks?.lineups?.the_lineup || [];
  const strategy = picks?.strategy;

  // Build badge text for header
  let badgeText: string | null = null;
  let badgeBorderColor = 'var(--border)';
  if (strategy?.label && !picksLoading) {
    const parts: string[] = [];
    if (picks?.game?.total) parts.push(`O/U ${picks.game.total}`);
    if (picks?.game?.spread != null)
      parts.push(`Spread ${Math.abs(picks.game.spread)}`);
    badgeText = parts.length
      ? `${strategy.label} \u00b7 ${parts.join(' \u00b7 ')}`
      : strategy.label;
    if (strategy.type === 'balanced') badgeBorderColor = 'var(--color-success)';
    else if (strategy.type === 'blowout_lean')
      badgeBorderColor = 'var(--color-danger)';
  }

  return (
    <div>
      {/* Header row: [← Back] [🎯 THE LINE UP] [Strategy badge] */}
      <div className={styles.headerRow}>
        <button type="button" className={styles.backBtn} onClick={handleBack}>
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polyline points="15 18 9 12 15 6" />
          </svg>
          Back
        </button>

        <div className={styles.title}>{'\uD83C\uDFAF'} The Line Up</div>

        {badgeText && (
          <div
            className={styles.headerBadge}
            style={{ borderColor: badgeBorderColor }}
          >
            {badgeText}
          </div>
        )}
      </div>

      {/* Loading picks */}
      {picksLoading && <SkeletonCard count={5} />}

      {/* Error loading picks */}
      {!picksLoading && picksError && (
        <EmptyState
          icon={'\uD83D\uDCE1'}
          message="Could not load analysis."
          action={{ label: 'Retry', onClick: () => refetch() }}
        />
      )}

      {/* Picks loaded successfully */}
      {!picksLoading && picks && (
        <div>
          {strategy && <StrategyInsight strategy={strategy} />}

          {lineup.length === 0 ? (
            <EmptyState
              icon={'\uD83D\uDCCA'}
              message="No lineup available for this game."
            />
          ) : (
            <div style={{ marginTop: 12 }}>
              {lineup.map((p, i) => (
                <PlayerCard
                  key={p.id || p.name}
                  player={p}
                  index={i}
                  showBoost={false}
                  tcolor={TEAM_COLORS[p.team] || '#14b8a6'}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
