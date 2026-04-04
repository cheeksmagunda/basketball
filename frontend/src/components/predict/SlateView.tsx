// ============================================================================
// SlateView.tsx — Slate-Wide predictions (Starting 5 + Moonshot)
// Shows the chalk/upside lineup toggle and PlayerCard list.
// ============================================================================

import { useSlate } from '../../api/slate';
import { useUiStore } from '../../store/uiStore';
import SlidingPillNav from '../shared/SlidingPillNav';
import PlayerCard from '../shared/PlayerCard';
import EmptyState from '../shared/EmptyState';
import LateDraftBanner from './LateDraftBanner';
import { TEAM_COLORS } from '../../utils/teamColors';

const SLOT_LABELS = ['2.0x', '1.8x', '1.6x', '1.4x', '1.2x'] as const;

const SLATE_MODES = [
  { key: 'chalk', label: 'Starting 5' },
  { key: 'upside', label: 'Moonshot' },
] as const;

export default function SlateView() {
  const { data: slate, isLoading, error, refetch } = useSlate();
  const slateMode = useUiStore((s) => s.slateMode);
  const setSlateMode = useUiStore((s) => s.setSlateMode);

  // Initial load handled by App-level OracleLoader; nothing to render yet
  if (isLoading && !slate) return null;

  // Error or failed slate
  if (error || !slate || slate.error) {
    return (
      <EmptyState
        icon={'\uD83D\uDCE1'}
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
        icon={'\uD83C\uDFC0'}
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

  // Determine if late-draft banner should show.
  // Slate is locked, games aren't all done, AND at least one game hasn't tipped yet
  // (startTime more than 5 min in the future). Mirrors the old app.js showLateDraftBanner check.
  const now = Date.now();
  const hasRemainingGames = (slate.games ?? []).some(
    (g) => g.startTime && new Date(g.startTime).getTime() - now > 5 * 60 * 1000,
  );
  const showLateDraft = slate.locked && !slate.all_complete && hasRemainingGames;

  return (
    <div>
      <SlidingPillNav
        items={[...SLATE_MODES]}
        activeKey={slateMode}
        onChange={(k) => setSlateMode(k as 'chalk' | 'upside')}
        accentRgb="20,184,166"
      />

      {players.length === 0 ? (
        <EmptyState
          icon={'\uD83D\uDCCA'}
          message="No players in this lineup."
        />
      ) : (
        <div>
          {players.map((p, i) => (
            <PlayerCard
              key={p.id || p.name}
              player={{ ...p, slot: SLOT_LABELS[i] ?? p.slot }}
              index={i}
              showBoost
              tcolor={TEAM_COLORS[p.team] || '#14b8a6'}
            />
          ))}
        </div>
      )}

      {showLateDraft && <LateDraftBanner onRegenerated={() => refetch()} />}
    </div>
  );
}
