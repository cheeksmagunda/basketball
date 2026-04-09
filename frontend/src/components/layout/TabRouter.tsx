import { lazy, Suspense } from 'react';
import { useUiStore } from '../../store/uiStore';
import PredictTab from '../predict/PredictTab';

// Code-split non-default tabs — only loaded when first visited
const BenTab = lazy(() => import('../ben/BenTab'));

// Shown only during lazy JS chunk download (~50-200ms). After the chunk loads,
// each tab renders its own skeleton/loading state via React Query.
function TabLoadingFallback() {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      height: '200px', opacity: 0.5,
    }}>
      <div style={{
        width: '24px', height: '24px', border: '3px solid rgba(212,166,64,0.3)',
        borderTopColor: 'var(--line)', borderRadius: '50%',
        animation: 'spin 0.8s linear infinite',
      }} />
    </div>
  );
}

export default function TabRouter() {
  const activeTab = useUiStore((s) => s.activeTab);

  return (
    <>
      {activeTab === 'predictions' && <PredictTab />}
      {activeTab === 'lab' && (
        <Suspense fallback={<TabLoadingFallback />}><BenTab /></Suspense>
      )}
    </>
  );
}
