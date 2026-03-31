// ============================================================================
// SlateView.tsx — Slate-Wide predictions (Starting 5 + Moonshot)
// Shows the chalk/upside lineup toggle and PlayerCard list.
// ============================================================================

import { useSlate } from '../../api/slate';
import { useUiStore } from '../../store/uiStore';
import SlidingPillNav from '../shared/SlidingPillNav';
import PlayerCard from '../shared/PlayerCard';
import OracleLoader from '../shared/OracleLoader';
import EmptyState from '../shared/EmptyState';
import LateDraftBanner from './LateDraftBanner';
import { TEAM_COLORS } from '../../utils/teamColors';

const SLATE_MODES = [
  { key: 'chalk', label: 'Starting 5' },
  { key: 'upside', label: 'Moonshot' },
] as const;

export default function SlateView() {
  const { data: slate, isLoading, error, refetch } = useSlate();
  const slateMode = useUiStore((s) => s.slateMode);
  const setSlateMode = useUiStore((s) => s.setSlateMode);

  // Loading state -- show Oracle 8-ball
  if (isLoading) return <OracleLoader visible />;

  // Error or failed slate
  if (error || !slate || slate.error) {
    return (
      <EmptyState
        icon="&#128225;"
        message={
          slate?.error === 'slate_failed'
            ? 'Slate temporarily unavailable.'
            : "Could not load predictions."
        }
        action={{ label: 'Retry', onClick: () => refetch() }}
      />
    );
  }

  // No games today
  if (slate.no_games) {
    return (
      <EmptyState
        icon="&#127936;"
        message={
          slate.next_slate_date
            ? `No NBA games today. Next slate: ${slate.next_slate_date}`
            : 'No NBA games today.'
        }
      />
    );
  }

  const lineups = slate.lineups;
  const players =
    slateMode === 'chalk'
      ? lineups?.chalk || []
      : lineups?.upside || [];

  // Determine if late-draft banner should show
  // Slate is locked but not all games are complete -- remaining games exist
  const showLateDraft = slate.locked && !slate.all_complete;

  return (
    <div>
      <SlidingPillNav
        items={[...SLATE_MODES]}
        activeKey={slateMode}
        onChange={(k) => setSlateMode(k as 'chalk' | 'upside')}
        accentRgb="20,184,166"
      />

      {showLateDraft && <LateDraftBanner onRegenerated={() => refetch()} />}

      {players.length === 0 ? (
        <EmptyState
          icon="&#128202;"
          message="No players in this lineup."
        />
      ) : (
        <div style={{ marginTop: 12 }}>
          {players.map((p, i) => (
            <PlayerCard
              key={p.id || p.name}
              player={p}
              index={i}
              showBoost
              tcolor={TEAM_COLORS[p.team] || '#14b8a6'}
            />
          ))}
        </div>
      )}
    </div>
  );
}
