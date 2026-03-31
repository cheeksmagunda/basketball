// ============================================================================
// PredictTab.tsx — Container for the Predict tab
// Sub-views: Slate-Wide (full slate) and Game (per-game analysis)
// ============================================================================

import { useUiStore } from '../../store/uiStore';
import SlidingPillNav from '../shared/SlidingPillNav';
import SlateView from './SlateView';
import GameView from './GameView';

const PREDICT_SUBS = [
  { key: 'slate', label: 'Slate-Wide' },
  { key: 'game', label: 'Game' },
] as const;

export default function PredictTab() {
  const predictSub = useUiStore((s) => s.predictSub);
  const setPredictSub = useUiStore((s) => s.setPredictSub);

  return (
    <div className="tab-page active">
      <SlidingPillNav
        items={[...PREDICT_SUBS]}
        activeKey={predictSub}
        onChange={(k) => setPredictSub(k as 'slate' | 'game')}
        accentRgb="20,184,166"
      />
      {predictSub === 'slate' ? <SlateView /> : <GameView />}
    </div>
  );
}
