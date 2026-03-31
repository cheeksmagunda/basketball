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

/**
 * Format a numeric value to one decimal place for display.
 * Returns "0" for falsy / NaN values.
 */
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

  // Score color: above 5.0 RS use team color, below use muted
  const scoreColor = player.rating >= 5.0 ? teamHex : undefined;

  return (
    <div
      className={styles['player-card']}
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
        {/* Meta row: team + position + injury */}
        <div className={styles['player-meta']}>
          <span>{player.team}</span>
          <span className={styles['pos-pill']}>{player.pos}</span>
          {player.injury_status && player.injury_status !== '' && (
            <span className={styles['injury-badge']}>
              {player.injury_status}
            </span>
          )}
        </div>

        {/* Player name */}
        <div className={styles['player-name']}>{player.name}</div>

        {/* Stat pills row: context pills + stat grid */}
        <div className={styles['stat-pills']}>
          {/* Context pills row */}
          <div className={styles['stat-context-row']}>
            {showBoost && player.est_mult > 0 && (
              <span
                className={styles['stat-context-pill']}
                style={{ color: teamHex, borderColor: hexToRgba(teamHex, 0.25) }}
              >
                +{player.est_mult.toFixed(1)}x card
              </span>
            )}
            {/* Hot streak indicator (from pass-through fields) */}
            {player._hot_streak && (
              <span
                className={`${styles['stat-context-pill']} ${styles['overperform-hot']}`}
              >
                HOT
              </span>
            )}
            {/* Odds adjusted indicator */}
            {player._odds_adjusted && (
              <span
                className={`${styles['stat-context-pill']} ${styles['overperform-odds']}`}
              >
                ODDS
              </span>
            )}
          </div>

          {/* Stat grid (5 columns) */}
          <div className={styles['stat-grid-row']}>
            {STAT_KEYS.map((key, i) => (
              <div key={key} className={styles['stat-col']}>
                <span className={styles['stat-col-val']}>
                  {fmt(player[key] as number)}
                </span>
                <span className={styles['stat-col-lbl']}>{STAT_LABELS[i]}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Score column */}
      <div className={styles['score-col']}>
        <div className={styles['score-num']}>{player.rating.toFixed(1)}</div>
        <span className={styles['score-label']}>RS</span>
        {player.slot && (
          <span className={styles['slot-badge']}>{player.slot}</span>
        )}
      </div>
    </div>
  );
}
