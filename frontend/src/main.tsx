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
// Prefetch critical data in parallel.
// Slate + Games are both needed by the Predict tab.
// ---------------------------------------------------------------------------
queryClient.prefetchQuery({ queryKey: ['slate'], queryFn: () => fetchJson('/api/slate', 30_000) });
queryClient.prefetchQuery({
  queryKey: ['games'],
  queryFn: async () => {
    const res = await fetchJson<{ data: unknown[] }>('/api/games', 15_000);
    return res.data ?? [];
  },
});

createRoot(document.getElementById('root')!).render(
  <QueryClientProvider client={queryClient}>
    <App />
  </QueryClientProvider>,
);
