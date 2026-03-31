// ============================================================================
// StrategyInsight.tsx — Per-game strategy badge + description
// Shown above THE LINE UP cards when a per-game analysis has strategy metadata.
// ============================================================================

import type { PerGameStrategy, PerGameStrategyType } from '../../types';
import styles from './StrategyInsight.module.css';

interface Props {
  strategy: PerGameStrategy;
}

const STRATEGY_COLORS: Record<PerGameStrategyType | string, string> = {
  balanced: '#14b8a6',
  neutral: '#94a3b8',
  blowout_lean: '#f59e0b',
  shootout: '#ef4444',
  defensive_grind: '#6366f1',
};

export default function StrategyInsight({ strategy }: Props) {
  const color = STRATEGY_COLORS[strategy.type] || '#14b8a6';

  return (
    <div className={styles.wrap}>
      <span
        className={styles.badge}
        style={{
          background: `${color}22`,
          color,
          borderColor: `${color}44`,
        }}
      >
        {strategy.label}
      </span>
      <p className={styles.desc}>{strategy.description}</p>
    </div>
  );
}
