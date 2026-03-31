import { useEffect } from 'react';
import { useSlate } from './api/slate';
import Header from './components/layout/Header';
import BottomNav from './components/layout/BottomNav';
import TabRouter from './components/layout/TabRouter';
import OracleLoader from './components/shared/OracleLoader';

export default function App() {
  const { data: slate, isLoading: slateLoading } = useSlate();

  // Remove the static HTML preloader once React has mounted and taken over.
  useEffect(() => {
    const el = document.getElementById('static-preloader');
    if (el) el.remove();
  }, []);

  // Show the React OracleLoader while the critical first query (slate) is loading.
  // The static HTML preloader covers the gap before React mounts; this covers
  // the gap while the API call is in flight.
  const showLoader = slateLoading && !slate;

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
