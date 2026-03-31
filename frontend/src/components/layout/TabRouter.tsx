import { useUiStore } from '../../store/uiStore';
import PredictTab from '../predict/PredictTab';
import LineTab from '../line/LineTab';
import ParlayTab from '../parlay/ParlayTab';
import BenTab from '../ben/BenTab';

export default function TabRouter() {
  const activeTab = useUiStore((s) => s.activeTab);

  return (
    <>
      {activeTab === 'predictions' && <PredictTab />}
      {activeTab === 'line' && <LineTab />}
      {activeTab === 'parlay' && <ParlayTab />}
      {activeTab === 'lab' && <BenTab />}
    </>
  );
}
