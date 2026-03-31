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

function fmt(val: number | undefined | null): string {
  if (val == null || Number.isNaN(val)) return '0';
  return val % 1 === 0 ? String(val) : val.toFixed(1);
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
          animationDelay: `${delay}s`,
        } as React.CSSProperties
      }
    >
      {/* Rank badge */}
      <div className={styles['rank-badge']}>{index + 1}</div>

      {/* Player info column */}
      <div className={styles['player-info']}>

        {/* Row 1: Name only — full width, no competing badges */}
        <div className={styles['card-header']}>
          <span className={styles['player-name']}>{player.name}</span>
        </div>

        {/* Row 2: Subtitle — boost + HOT + team/pos/matchup left | slot badge right */}
        <div className={styles['subtitle-row']}>
          <div className={styles['subtitle-left']}>
            {showBoost && player.est_mult > 0 && (
              <span
                className={styles['boost-pill']}
                style={{ color: teamHex, borderColor: hexToRgba(teamHex, 0.25) }}
              >
                +{player.est_mult.toFixed(1)}x card
              </span>
            )}
            {player._odds_adjusted && (
              <span className={styles['odds-pill']}>ODDS</span>
            )}
            {player._hot_streak && (
              <span className={styles['hot-pill']}>HOT</span>
            )}
            <span
              className={styles['team-badge']}
              style={{ color: teamHex, borderColor: hexToRgba(teamHex, 0.3) }}
            >
              {player.team}
            </span>
            <span className={styles['pos-pill']}>{player.pos}</span>
            {player.injury_status && player.injury_status !== '' && (
              <span className={styles['injury-badge']}>{player.injury_status}</span>
            )}
            {player.opp && (
              <span className={styles['matchup-text']}>vs {player.opp}</span>
            )}
          </div>
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

        {/* Row 3: Stat grid */}
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

        {/* Row 4: Minutes bar */}
        {player.predMin > 0 && (
          <div className={styles['minutes-bar-wrap']}>
            <span className={styles['minutes-label']}>MIN</span>
            <div className={styles['minutes-track']}>
              <div
                className={styles['minutes-fill']}
                style={{ width: `${Math.min((player.predMin / 48) * 100, 100)}%` }}
              />
              {avgMin > 0 && (
                <div
                  className={styles['minutes-avg-marker']}
                  style={{ left: `${Math.min((avgMin / 48) * 100, 100)}%` }}
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

      {/* Score zone — RS number + label only */}
      <div className={styles['score-zone']}>
        <div className={styles['score-num']}>{player.rating.toFixed(1)}</div>
        <span className={styles['score-label']}>RS</span>
      </div>
    </div>
  );
}
