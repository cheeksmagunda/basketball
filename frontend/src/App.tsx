import Header from './components/layout/Header';
import BottomNav from './components/layout/BottomNav';
import TabRouter from './components/layout/TabRouter';

export default function App() {
  return (
    <div className="app">
      <Header />
      <TabRouter />
      <BottomNav />
    </div>
  );
}
