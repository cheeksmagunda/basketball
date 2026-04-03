import { useEffect, useState } from 'react';
import { useSlate } from './api/slate';
import ErrorBoundary from './components/shared/ErrorBoundary';
import Header from './components/layout/Header';
import BottomNav from './components/layout/BottomNav';
import TabRouter from './components/layout/TabRouter';
import OracleLoader from './components/shared/OracleLoader';

// After this many ms of loading with no data, show a "taking too long" retry UI
// instead of an infinite spinner. With retry:1 + 30s timeout + 8s delay the
// worst-case backend wait is ~68s, so we surface the retry option at 45s.
const LOAD_TIMEOUT_MS = 45_000;

function AppInner() {
  const { data: slate, isLoading: slateLoading, refetch } = useSlate();
  const [loadTimedOut, setLoadTimedOut] = useState(false);

  // Remove the static HTML preloader once React has mounted and taken over.
  useEffect(() => {
    const el = document.getElementById('static-preloader');
    if (el) el.remove();
  }, []);

  // If the slate is still loading after LOAD_TIMEOUT_MS, surface a retry UI
  // so the user isn't staring at a blank Oracle loader for 3+ minutes.
  useEffect(() => {
    if (!slateLoading || slate) {
      setLoadTimedOut(false);
      return;
    }
    const id = setTimeout(() => setLoadTimedOut(true), LOAD_TIMEOUT_MS);
    return () => clearTimeout(id);
  }, [slateLoading, slate]);

  // Show the React OracleLoader while the critical first query (slate) is loading.
  // The static HTML preloader covers the gap before React mounts; this covers
  // the gap while the API call is in flight.
  const showLoader = slateLoading && !slate && !loadTimedOut;

  if (loadTimedOut) {
    return (
      <div style={{
        display: 'flex', flexDirection: 'column', alignItems: 'center',
        justifyContent: 'center', height: '100dvh', gap: '16px',
        padding: '24px', textAlign: 'center',
        fontFamily: "'Barlow Condensed', sans-serif",
        background: '#060a0f', color: '#e8edf5',
      }}>
        <span style={{ fontSize: '2rem' }}>📡</span>
        <p style={{ fontSize: '1rem', fontWeight: 700, letterSpacing: '0.05em' }}>
          Taking longer than expected.
        </p>
        <p style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
          The backend may be starting up. Try again in a moment.
        </p>
        <button
          onClick={() => { setLoadTimedOut(false); refetch(); }}
          style={{
            padding: '12px 28px', borderRadius: '9999px',
            background: '#14b8a6', color: '#060a0f',
            fontFamily: "'Barlow Condensed', sans-serif",
            fontSize: '0.95rem', fontWeight: 900,
            letterSpacing: '0.08em', textTransform: 'uppercase',
            border: 'none', cursor: 'pointer',
          }}
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="app">
      <OracleLoader visible={showLoader} />
      <Header />
      <div className="divider" style={{ margin: '8px 0 0' }} />
      <TabRouter />
      <BottomNav />
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <AppInner />
    </ErrorBoundary>
  );
}
