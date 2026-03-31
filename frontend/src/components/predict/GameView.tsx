// ============================================================================
// GameView.tsx — Per-game analysis view
// Game selector dropdown + THE LINE UP (5-player optimal lineup).
// ============================================================================

import { useState, useCallback } from 'react';
import { useGames, usePicks } from '../../api/slate';
import PlayerCard from '../shared/PlayerCard';
import SkeletonCard from '../shared/SkeletonCard';
import OracleLoader from '../shared/OracleLoader';
import EmptyState from '../shared/EmptyState';
import StrategyInsight from './StrategyInsight';
import { TEAM_COLORS } from '../../utils/teamColors';

export default function GameView() {
  const { data: games, isLoading: gamesLoading } = useGames();
  const [selectedGame, setSelectedGame] = useState<string | null>(null);
  const {
    data: picks,
    isLoading: picksLoading,
    error: picksError,
    refetch,
  } = usePicks(selectedGame);

  const handleGameChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      setSelectedGame(e.target.value || null);
    },
    [],
  );

  // Loading games list
  if (gamesLoading) return <SkeletonCard count={1} />;

  const lineup = picks?.lineups?.the_lineup || [];

  return (
    <div>
      {/* Game selector dropdown -- uses global .game-select-wrap styles */}
      <div className="game-select-wrap">
        <select value={selectedGame || ''} onChange={handleGameChange}>
          <option value="">Select a game...</option>
          {(games || []).map((g) => (
            <option key={g.gameId} value={g.gameId} disabled={g.locked}>
              {g.label}
              {g.locked ? ' (locked)' : ''}
            </option>
          ))}
        </select>
      </div>

      {/* No game selected */}
      {!selectedGame && (
        <EmptyState
          icon={'\uD83C\uDFC0'}
          message="Select a game above to see the optimal lineup."
        />
      )}

      {/* Loading picks for selected game */}
      {selectedGame && picksLoading && <OracleLoader visible />}

      {/* Error loading picks */}
      {selectedGame && !picksLoading && picksError && (
        <EmptyState
          icon={'\uD83D\uDCE1'}
          message="Could not load analysis."
          action={{ label: 'Retry', onClick: () => refetch() }}
        />
      )}

      {/* Picks loaded successfully */}
      {selectedGame && !picksLoading && picks && (
        <div>
          {/* Strategy insight bar (if strategy metadata is present) */}
          {picks.strategy && <StrategyInsight strategy={picks.strategy} />}

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
