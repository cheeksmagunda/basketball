// ============================================================================
// useMediaQuery — CSS media query hook for responsive layout decisions.
// Uses window.matchMedia for efficient, event-driven screen size detection.
// No resize polling — the browser notifies us when the query result changes.
// ============================================================================

import { useState, useEffect } from 'react';

/**
 * Subscribe to a CSS media query and return whether it currently matches.
 *
 * @param query - A valid CSS media query string, e.g. '(min-width: 768px)'
 * @returns `true` when the query matches, `false` otherwise.
 *
 * @example
 * const isDesktop = useMediaQuery('(min-width: 768px)');
 * const isWide    = useMediaQuery('(min-width: 1024px)');
 */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => {
    if (typeof window === 'undefined') return false;
    return window.matchMedia(query).matches;
  });

  useEffect(() => {
    const mql = window.matchMedia(query);
    setMatches(mql.matches);

    const handler = (e: MediaQueryListEvent) => setMatches(e.matches);
    mql.addEventListener('change', handler);
    return () => mql.removeEventListener('change', handler);
  }, [query]);

  return matches;
}
