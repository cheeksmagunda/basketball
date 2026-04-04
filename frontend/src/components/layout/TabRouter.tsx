import { lazy, Suspense } from 'react';
import { useUiStore } from '../../store/uiStore';
import PredictTab from '../predict/PredictTab';

// Code-split non-default tabs — only loaded when first visited
const LineTab = lazy(() => import('../line/LineTab'));
const ParlayTab = lazy(() => import('../parlay/ParlayTab'));
const BenTab = lazy(() => import('../ben/BenTab'));

export default function TabRouter() {
  const activeTab = useUiStore((s) => s.activeTab);

  return (
    <Suspense fallback={null}>
      {activeTab === 'predictions' && <PredictTab />}
      {activeTab === 'line' && <LineTab />}
      {activeTab === 'parlay' && <ParlayTab />}
      {activeTab === 'lab' && <BenTab />}
    </Suspense>
  );
}
