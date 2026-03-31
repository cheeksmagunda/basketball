import Header from './components/layout/Header';
import BottomNav from './components/layout/BottomNav';
import TabRouter from './components/layout/TabRouter';

export default function App() {
  return (
    <div className="app">
      <Header />
      <div className="divider" style={{ margin: '8px 0 0' }} />
      <TabRouter />
      <BottomNav />
    </div>
  );
}
