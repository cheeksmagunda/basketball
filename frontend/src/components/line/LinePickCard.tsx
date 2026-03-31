import { TEAM_COLORS } from '../../utils/teamColors';
import { hexToRgba } from '../../utils/hexToRgba';
import { getStatLabel } from '../../utils/statLabels';
import styles from './LinePickCard.module.css';
import type { LinePick } from '../../types';

interface LinePickCardProps {
  pick: LinePick;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Format a number to one decimal, or return "--" for null/NaN. */
function fmt(val: number | null | undefined, decimals = 1): string {
  if (val == null || Number.isNaN(val)) return '--';
  return Number.isInteger(val) ? String(val) : val.toFixed(decimals);
}

/** Build the "7:30 PM ET" or "CLE vs BOS" matchup string. */
function buildSubheader(pick: LinePick): string {
  const matchup = `${pick.team} vs ${pick.opponent}`;
  if (pick.game_time) return `${matchup}  ·  ${pick.game_time}`;
  return matchup;
}

/** Format odds timestamp to "Odds · 3:15 PM CT" style. */
function fmtOddsTimestamp(iso: string | null): string | null {
  if (!iso) return null;
  try {
    const d = new Date(iso);
    const time = d.toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      timeZone: 'America/Chicago',
      hour12: true,
    });
    return `Odds · ${time} CT`;
  } catch {
    return null;
  }
}

/**
 * Render L5 recent form values.
 * Uses `recent_form_values` (raw stat values) when available,
 * otherwise falls back to ratio-derived values from `recent_form_bars`.
 */
function getL5Values(pick: LinePick): number[] | null {
  if (pick.recent_form_values?.length) return pick.recent_form_values;
  if (pick.recent_form_bars?.length && pick.season_avg != null && pick.season_avg > 0) {
    return pick.recent_form_bars.map((ratio) => Math.round(ratio * pick.season_avg! * 10) / 10);
  }
  return null;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function LinePickCard({ pick }: LinePickCardProps) {
  const teamHex = TEAM_COLORS[pick.team] || '#d4a640';
  const teamBorder = hexToRgba(teamHex, 0.3);
  const teamGlow = hexToRgba(teamHex, 0.15);
  const statLabel = getStatLabel(pick.stat_type);
  const isOver = pick.direction === 'over';
  const dirClass = isOver ? styles.over : styles.under;
  const edgePlus = pick.edge >= 0;
  const oddsTs = fmtOddsTimestamp(pick.line_updated_at);
  const l5 = getL5Values(pick);
  const isResolved = pick.result === 'hit' || pick.result === 'miss';

  // Determine the odds string to show (e.g. "-115")
  const oddsVal = isOver ? pick.odds_over : pick.odds_under;
  const oddsDisplay = oddsVal != null
    ? (oddsVal > 0 ? `+${oddsVal}` : String(oddsVal))
    : null;

  return (
    <div
      className={styles.card}
      style={
        {
          '--tcolor': teamHex,
          '--tcolor-border': teamBorder,
          '--tcolor-glow': teamGlow,
        } as React.CSSProperties
      }
    >
      {/* ── Zone 1: Header ── */}
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <div className={styles.playerName}>{pick.player_name}</div>
          <div className={styles.teamLine}>{buildSubheader(pick)}</div>
        </div>
        <div className={styles.headerMeta}>
          {oddsDisplay && (
            <div className={styles.oddsNum}>{oddsDisplay}</div>
          )}
          {pick.model_only && !oddsDisplay && (
            <div className={styles.oddsLabel}>MODEL</div>
          )}
          {oddsTs && <div className={styles.oddsTs}>{oddsTs}</div>}
        </div>
      </div>

      {/* ── Zone 2: The Play ── */}
      <div className={styles.playRow}>
        <span className={`${styles.dirPill} ${dirClass}`}>
          {pick.direction.toUpperCase()}
        </span>
        <span className={styles.statVal}>
          {fmt(pick.line, 1)} {statLabel.toLowerCase()}
        </span>
      </div>

      {/* ── Zone 3: Data row (5 columns) ── */}
      <div className={styles.dataRow}>
        {/* Col 1: Baseline */}
        <div className={styles.dataCol}>
          <span className={styles.dataVal}>{fmt(pick.books_consensus ?? pick.line, 1)}</span>
          <span className={styles.dataLabel}>{statLabel}</span>
          <span className={styles.dataSub}>Baseline</span>
        </div>

        {/* Col 2: Edge */}
        <div className={styles.dataCol}>
          <span
            className={`${styles.dataVal} ${edgePlus ? styles.edgePlus : styles.edgeMinus}`}
          >
            {edgePlus ? '+' : ''}{fmt(pick.edge, 1)}
          </span>
          <span className={styles.dataLabel}>Edge</span>
        </div>

        {/* Col 3: Projection / Season avg */}
        <div className={styles.dataCol}>
          <span className={styles.dataVal}>{fmt(pick.projection, 1)}</span>
          <span className={styles.dataSub}>{fmt(pick.season_avg, 1)} avg</span>
          <span className={styles.dataLabel}>{statLabel}</span>
        </div>

        {/* Col 4: Minutes */}
        <div className={styles.dataCol}>
          <span className={styles.dataVal}>{fmt(pick.proj_min, 0)}</span>
          <span className={styles.dataSub}>{fmt(pick.avg_min, 0)} avg</span>
          <span className={styles.dataLabel}>MIN</span>
        </div>

        {/* Col 5: L5 recent form */}
        <div className={styles.dataCol}>
          {l5 ? (
            <div className={styles.l5Wrap}>
              {l5.slice(0, 5).map((v, i) => {
                const baseline = pick.books_consensus ?? pick.line;
                const hitBaseline = isOver ? v >= baseline : v <= baseline;
                return (
                  <span
                    key={i}
                    className={`${styles.l5Val} ${hitBaseline ? styles.l5Hit : styles.l5Miss}`}
                  >
                    {Math.round(v)}
                  </span>
                );
              })}
            </div>
          ) : (
            <span className={styles.dataVal}>--</span>
          )}
          <span className={styles.dataLabel}>L5</span>
        </div>
      </div>

      {/* ── Zone 4: Result (when resolved) ── */}
      {isResolved && (
        <div className={styles.resultRow}>
          <span className={styles.resultLabel}>Result</span>
          <div className={styles.resultRight}>
            {pick.actual_stat != null && (
              <span className={styles.resultActual}>
                {fmt(pick.actual_stat, 1)} {statLabel}
              </span>
            )}
            <span
              className={`${styles.resultPill} ${
                pick.result === 'hit' ? styles.hit : styles.miss
              }`}
            >
              {pick.result === 'hit' ? 'HIT' : 'MISS'}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
