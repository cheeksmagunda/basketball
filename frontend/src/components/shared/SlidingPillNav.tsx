import { useRef, useEffect, useState, useCallback } from 'react';
import styles from './SlidingPillNav.module.css';

interface SlidingPillNavItem {
  key: string;
  label: string;
}

interface SlidingPillNavProps {
  items: SlidingPillNavItem[];
  activeKey: string;
  onChange: (key: string) => void;
  accentRgb?: string; // e.g. "212,166,64" for gold
}

export default function SlidingPillNav({
  items,
  activeKey,
  onChange,
  accentRgb,
}: SlidingPillNavProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const btnRefs = useRef<Map<string, HTMLButtonElement>>(new Map());
  const [pillStyle, setPillStyle] = useState<React.CSSProperties>({});

  const measurePill = useCallback(() => {
    const wrap = wrapRef.current;
    const btn = btnRefs.current.get(activeKey);
    if (!wrap || !btn) return;

    const wrapRect = wrap.getBoundingClientRect();
    const btnRect = btn.getBoundingClientRect();

    setPillStyle({
      width: btnRect.width,
      transform: `translateX(${btnRect.left - wrapRect.left - 3}px)`,
    });
  }, [activeKey]);

  useEffect(() => {
    measurePill();
  }, [measurePill]);

  // Re-measure on resize
  useEffect(() => {
    const handler = () => measurePill();
    window.addEventListener('resize', handler);
    return () => window.removeEventListener('resize', handler);
  }, [measurePill]);

  // Apply accent color CSS custom properties if provided
  const wrapStyle = accentRgb
    ? ({
        '--accent-rgb': accentRgb,
      } as React.CSSProperties)
    : undefined;

  return (
    <div
      ref={wrapRef}
      className={`${styles['slide-pill-wrap']} ${accentRgb ? styles['custom-accent'] : ''}`}
      style={wrapStyle}
    >
      {/* Sliding pill indicator */}
      <div className={styles['slide-pill']} style={pillStyle} />

      {/* Buttons */}
      {items.map((item) => (
        <button
          key={item.key}
          ref={(el) => {
            if (el) {
              btnRefs.current.set(item.key, el);
            } else {
              btnRefs.current.delete(item.key);
            }
          }}
          className={`${styles['slide-pill-btn']} ${activeKey === item.key ? styles.active : ''}`}
          onClick={() => onChange(item.key)}
          type="button"
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}
