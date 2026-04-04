import { useRef, useEffect, useCallback } from 'react';
import { Activity, TrendingUp, Layers, MessageCircle } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { fetchJson } from '../../api/client';
import { useUiStore } from '../../store/uiStore';
import type { TabName } from '../../types';
import styles from './BottomNav.module.css';

const TABS: { id: TabName; label: string; Icon: typeof Activity }[] = [
  { id: 'predictions', label: 'Predict', Icon: Activity },
  { id: 'line', label: 'Line', Icon: TrendingUp },
  { id: 'parlay', label: 'Parlay', Icon: Layers },
  { id: 'lab', label: 'Ben', Icon: MessageCircle },
];

/** Accent color per tab — pill background tint (gold for all) */
const TAB_ACCENT: Record<TabName, string> = {
  predictions: 'rgba(212,166,64,0.14)',
  line: 'rgba(212,166,64,0.14)',
  parlay: 'rgba(212,166,64,0.14)',
  lab: 'rgba(212,166,64,0.14)',
};

const TAB_ACCENT_BORDER: Record<TabName, string> = {
  predictions: 'rgba(212,166,64,0.10)',
  line: 'rgba(212,166,64,0.10)',
  parlay: 'rgba(212,166,64,0.10)',
  lab: 'rgba(212,166,64,0.10)',
};

const TAB_ACCENT_GLOW: Record<TabName, string> = {
  predictions: 'rgba(212,166,64,0.10)',
  line: 'rgba(212,166,64,0.10)',
  parlay: 'rgba(212,166,64,0.10)',
  lab: 'rgba(212,166,64,0.10)',
};

const TAB_ACTIVE_COLOR: Record<TabName, string> = {
  predictions: 'var(--line)',
  line: 'var(--line)',
  parlay: 'var(--line)',
  lab: 'var(--line)',
};

export default function BottomNav() {
  const activeTab = useUiStore((s) => s.activeTab);
  const setActiveTab = useUiStore((s) => s.setActiveTab);
  const pillRef = useRef<HTMLDivElement>(null);
  const rowRef = useRef<HTMLDivElement>(null);
  const qc = useQueryClient();

  // Prefetch tab data on hover/touch-start (~100-300ms head start)
  const prefetchTab = useCallback((tab: TabName) => {
    if (tab === activeTab) return;
    if (tab === 'line') {
      qc.prefetchQuery({ queryKey: ['line-of-the-day', false], queryFn: () => fetchJson('/api/line-of-the-day', 90_000) });
    } else if (tab === 'parlay') {
      qc.prefetchQuery({ queryKey: ['parlay'], queryFn: () => fetchJson('/api/parlay', 90_000) });
    }
  }, [activeTab, qc]);

  const movePill = useCallback(() => {
    const pill = pillRef.current;
    const row = rowRef.current;
    if (!pill || !row) return;

    const idx = TABS.findIndex((t) => t.id === activeTab);
    const btn = row.children[idx + 1] as HTMLElement | undefined; // +1 because pill is child[0]
    if (!btn) return;

    pill.style.transform = `translateX(${btn.offsetLeft}px)`;
    pill.style.width = `${btn.offsetWidth}px`;
    pill.style.background = `linear-gradient(135deg, ${TAB_ACCENT[activeTab]} 0%, ${TAB_ACCENT[activeTab].replace('0.14', '0.06')} 100%)`;
    pill.style.borderColor = TAB_ACCENT_BORDER[activeTab];
    pill.style.boxShadow = `0 0 10px ${TAB_ACCENT_GLOW[activeTab]}, inset 0 1px 1px ${TAB_ACCENT_GLOW[activeTab].replace('0.10', '0.08')}`;
  }, [activeTab]);

  useEffect(() => {
    movePill();
  }, [movePill]);

  // Recalculate pill on resize
  useEffect(() => {
    window.addEventListener('resize', movePill);
    return () => window.removeEventListener('resize', movePill);
  }, [movePill]);

  return (
    <nav className={styles['bottom-nav']}>
      <div className={styles['bnav-icon-row']} ref={rowRef}>
        <div className={styles['bnav-pill']} ref={pillRef} />
        {TABS.map(({ id, label, Icon }) => (
          <button
            key={id}
            className={`${styles['bnav-icon-btn']} ${activeTab === id ? styles.active : ''}`}
            style={activeTab === id ? { color: TAB_ACTIVE_COLOR[id] } : undefined}
            onPointerEnter={() => prefetchTab(id)}
            onClick={() => setActiveTab(id)}
            aria-label={label}
            aria-current={activeTab === id ? 'page' : undefined}
          >
            <span className={styles['bnav-icon']}>
              <Icon />
            </span>
            <span className={styles['bnav-label']}>{label}</span>
          </button>
        ))}
      </div>
    </nav>
  );
}
