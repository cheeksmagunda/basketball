import { lazy, Suspense } from 'react';
import { useUiStore } from '../../store/uiStore';
import OracleLoader from '../shared/OracleLoader';

const PredictTab = lazy(() => import('../predict/PredictTab'));
const LineTab = lazy(() => import('../line/LineTab'));
const ParlayTab = lazy(() => import('../parlay/ParlayTab'));
const BenTab = lazy(() => import('../ben/BenTab'));

export default function TabRouter() {
  const activeTab = useUiStore((s) => s.activeTab);

  return (
    <Suspense fallback={<OracleLoader visible />}>
      {activeTab === 'predictions' && <PredictTab />}
      {activeTab === 'line' && <LineTab />}
      {activeTab === 'parlay' && <ParlayTab />}
      {activeTab === 'lab' && <BenTab />}
    </Suspense>
  );
}
