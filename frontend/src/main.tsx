import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fetchJson } from './api/client';
import './styles/global.css';
import App from './App';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

// ---------------------------------------------------------------------------
// Prefetch ALL tab data immediately — before React even renders.
// When components mount and call hooks with matching queryKeys, React Query
// joins the in-flight request instead of firing a new one.
// ---------------------------------------------------------------------------
queryClient.prefetchQuery({ queryKey: ['slate'], queryFn: () => fetchJson('/api/slate', 30_000) });
queryClient.prefetchQuery({ queryKey: ['line-of-the-day', undefined], queryFn: () => fetchJson('/api/line-of-the-day', 90_000) });
queryClient.prefetchQuery({ queryKey: ['line-history'], queryFn: () => fetchJson('/api/line-history', 15_000) });
queryClient.prefetchQuery({ queryKey: ['parlay'], queryFn: () => fetchJson('/api/parlay', 90_000) });
queryClient.prefetchQuery({ queryKey: ['parlay-history'], queryFn: () => fetchJson('/api/parlay-history', 15_000) });
queryClient.prefetchQuery({ queryKey: ['lab-briefing'], queryFn: () => fetchJson('/api/lab/briefing', 30_000) });

createRoot(document.getElementById('root')!).render(
  <QueryClientProvider client={queryClient}>
    <App />
  </QueryClientProvider>,
);
