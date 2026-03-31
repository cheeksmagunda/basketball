import type { PlayerCard as Player } from '../../types/index';
import { TEAM_COLORS } from '../../utils/teamColors';
import { hexToRgba } from '../../utils/hexToRgba';
import styles from './PlayerCard.module.css';

interface PlayerCardProps {
  player: Player;
  index: number;
  showBoost?: boolean;
  tcolor?: string;
  animDelay?: number;
}

const STAT_KEYS = ['pts', 'reb', 'ast', 'stl', 'blk'] as const;
const STAT_LABELS = ['PTS', 'REB', 'AST', 'STL', 'BLK'] as const;

/** Format a numeric value for display. Returns "0" for falsy/NaN. */
function fmt(val: number | undefined | null): string {
  if (val == null || Number.isNaN(val)) return '0';
  return val % 1 === 0 ? String(val) : val.toFixed(1);
}

/** Semantic accent bar color based on player value tier. */
function getAccentColor(player: Player, showBoost: boolean): string {
  if (player.injury_status && player.injury_status !== '') {
    return 'var(--color-danger)';
  }
  if (showBoost) {
    if (player.draft_ev >= 15.0) return 'var(--color-success)';
    if (player.draft_ev >= 8.0) return 'var(--color-warning)';
    return 'var(--color-text-muted)';
  }
  if (player.rating >= 5.0) return 'var(--color-success)';
  if (player.rating >= 3.5) return 'var(--color-warning)';
  return 'var(--color-text-muted)';
}

export default function PlayerCard({
  player,
  index,
  showBoost = true,
  tcolor,
  animDelay,
}: PlayerCardProps) {
  const teamHex = tcolor || TEAM_COLORS[player.team] || '#14b8a6';
  const teamAlpha = hexToRgba(teamHex, 0.04);
  const delay = animDelay ?? index * 0.06;
  const scoreColor = player.rating >= 5.0 ? teamHex : undefined;
  const accentColor = getAccentColor(player, showBoost);
  const avgMin = player.avg_min ?? player.season_min ?? 0;

  return (
    <div
      className={styles['player-card']}
      role="article"
      aria-label={`${player.name}, ${player.team}, projected RS ${player.rating.toFixed(1)}${
        showBoost && player.est_mult > 0 ? `, ${player.est_mult.toFixed(1)}x boost` : ''
      }`}
      style={
        {
          '--tcolor': teamHex,
          '--tcolor-alpha': teamAlpha,
          '--score-color': scoreColor,
          '--accent-color': accentColor,
          animationDelay: `${delay}s`,
        } as React.CSSProperties
      }
    >
      {/* Rank badge */}
      <div className={styles['rank-badge']}>{index + 1}</div>

      {/* Player info column */}
      <div className={styles['player-info']}>
        {/* === HEADER ZONE === */}
        <div className={styles['card-header']}>
          <div className={styles['player-identity']}>
            <span className={styles['player-name']}>{player.name}</span>
            {player.opp ? (
              <span className={styles['player-opp']}>
                {player.team} vs {player.opp}
              </span>
            ) : (
              <span className={styles['player-opp']}>{player.team}</span>
            )}
          </div>
          <div className={styles['badge-row']}>
            {player._hot_streak && (
              <span className={styles['hot-pill']}>HOT</span>
            )}
            <span className={styles['pos-pill']}>{player.pos}</span>
            {player.injury_status && player.injury_status !== '' && (
              <span className={styles['injury-badge']}>
                {player.injury_status}
              </span>
            )}
          </div>
        </div>

        {/* === BODY ZONE === */}
        <div className={styles['card-body']}>
          {/* Boost + Odds row (slate-wide only) */}
          {showBoost && (player.est_mult > 0 || player._odds_adjusted) && (
            <div className={styles['boost-row']}>
              {player.est_mult > 0 && (
                <span
                  className={styles['boost-pill']}
                  style={{
                    color: teamHex,
                    borderColor: hexToRgba(teamHex, 0.25),
                  }}
                >
                  +{player.est_mult.toFixed(1)}x card
                </span>
              )}
              {player._odds_adjusted && (
                <span className={styles['odds-pill']}>ODDS</span>
              )}
            </div>
          )}

          {/* Minutes progress bar */}
          {player.predMin > 0 && (
            <div className={styles['minutes-bar-wrap']}>
              <span className={styles['minutes-label']}>MIN</span>
              <div className={styles['minutes-track']}>
                <div
                  className={styles['minutes-fill']}
                  style={{
                    width: `${Math.min((player.predMin / 48) * 100, 100)}%`,
                  }}
                />
                {avgMin > 0 && (
                  <div
                    className={styles['minutes-avg-marker']}
                    style={{
                      left: `${Math.min((avgMin / 48) * 100, 100)}%`,
                    }}
                  />
                )}
              </div>
              <span className={styles['minutes-values']}>
                {avgMin > 0 && <>{fmt(avgMin)} / </>}
                <span className={styles.proj}>{fmt(player.predMin)}</span>
              </span>
            </div>
          )}
        </div>

        {/* === FOOTER ZONE === */}
        <div className={styles['card-footer']}>
          <div className={styles['stat-grid-row']}>
            {STAT_KEYS.map((key, i) => {
              const seasonKey = `season_${key}` as keyof Player;
              const seasonVal = player[seasonKey] as number | undefined;
              return (
                <div key={key} className={styles['stat-col']}>
                  <span className={styles['stat-val-proj']}>
                    {fmt(player[key] as number)}
                  </span>
                  {seasonVal != null && seasonVal > 0 && (
                    <span className={styles['stat-val-avg']}>
                      {fmt(seasonVal)}
                    </span>
                  )}
                  <span className={styles['stat-lbl']}>{STAT_LABELS[i]}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Score zone */}
      <div className={styles['score-zone']}>
        <div className={styles['score-num']}>{player.rating.toFixed(1)}</div>
        <span className={styles['score-label']}>RS</span>
        {player.slot && (
          <span
            className={`${styles['slot-badge']}${
              player.slot === '2.0x' || player.slot === '1.8x'
                ? ` ${styles['slot-badge-high']}`
                : ''
            }`}
          >
            {player.slot}
          </span>
        )}
      </div>
    </div>
  );
}
