import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fetchJson } from './api/client';
import './styles/global.css';
import App from './App';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60 * 1000,           // 60 seconds (more aggressive revalidation paired with prefetch)
      gcTime: 30 * 60 * 1000,         // Keep cached data for 30 minutes (prevents unmount loss)
      retry: 1,
      refetchOnWindowFocus: false,     // Global disable (stop alt-tab thrashing)
    },
  },
});

// ---------------------------------------------------------------------------
// Prefetch critical first-tab data in parallel.
// Slate + Games are both needed by the Predict tab (default tab).
// Line, Parlay, and Lab Briefing are prefetched in the background while the user
// reads the Predict tab. This enables instant tab switches (zero skeleton loaders).
// ---------------------------------------------------------------------------
queryClient.prefetchQuery({ queryKey: ['slate'], queryFn: () => fetchJson('/api/slate', 30_000) });
queryClient.prefetchQuery({
  queryKey: ['games'],
  queryFn: async () => {
    const res = await fetchJson<{ data: unknown[] }>('/api/games', 15_000);
    return res.data ?? [];
  },
});

// ── Global prefetch: background load of adjacent tabs ──
// These complete silently while the user reads the Predict tab (10-20 seconds),
// enabling instant tab switches with no skeleton loaders.

// Prefetch Line of the Day (high priority, reasonable size)
queryClient.prefetchQuery({
  queryKey: ['line'],
  queryFn: () => fetchJson('/api/line-of-the-day', 90_000), // 90s timeout (matches useLineOfTheDay)
  staleTime: 5 * 60 * 1000, // 5 minutes
});

// Prefetch Parlay (heavy computation, but cacheable)
queryClient.prefetchQuery({
  queryKey: ['parlay'],
  queryFn: () => fetchJson('/api/parlay', 90_000), // 90s timeout (matches useParlay)
  staleTime: 30 * 60 * 1000, // 30 minutes (already has long staleTime in hook)
});

// Prefetch Lab Briefing (required for historical accuracy context)
queryClient.prefetchQuery({
  queryKey: ['lab-briefing'],
  queryFn: () => fetchJson('/api/lab/briefing', 30_000), // 30s timeout (context load)
  staleTime: 5 * 60 * 1000, // 5 minutes
});

createRoot(document.getElementById('root')!).render(
  <QueryClientProvider client={queryClient}>
    <App />
  </QueryClientProvider>,
);
