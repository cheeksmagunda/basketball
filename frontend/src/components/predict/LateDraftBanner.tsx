// ============================================================================
// LateDraftBanner.tsx — Late draft banner
// Shown when the slate is locked but remaining (unlocked) games exist.
// Triggers /api/force-regenerate?scope=remaining to generate picks for
// games that haven't tipped off yet.
// ============================================================================

import { useState } from 'react';
import { fetchWithTimeout } from '../../api/client';
import styles from './LateDraftBanner.module.css';

interface Props {
  onRegenerated: () => void;
}

export default function LateDraftBanner({ onRegenerated }: Props) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);
  const [triggered, setTriggered] = useState(false);

  async function handleGenerate() {
    setLoading(true);
    setError(false);
    try {
      const r = await fetchWithTimeout(
        '/api/force-regenerate?scope=remaining',
        {},
        60_000,
      );
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      if (data.status === 'no_remaining_games') {
        setError(true);
        return;
      }
      setTriggered(true);
      onRegenerated();
    } catch (e) {
      console.warn('[late-draft] regeneration failed:', e);
      setError(true);
    } finally {
      setLoading(false);
    }
  }

  if (triggered) {
    return (
      <div className={styles.banner}>
        <div className={styles.text}>
          <strong>Picks updated.</strong>{' '}
          Lineups regenerated for remaining games.
        </div>
      </div>
    );
  }

  return (
    <div className={styles.banner}>
      <div className={styles.text}>
        <strong>Late to the party?</strong>{' '}
        {error
          ? 'No remaining games to generate picks for.'
          : 'Generate picks for remaining games.'}
      </div>
      <button
        className={styles.btn}
        onClick={handleGenerate}
        disabled={loading}
        type="button"
      >
        {loading ? 'Generating...' : 'Generate Late Draft'}
      </button>
    </div>
  );
}
