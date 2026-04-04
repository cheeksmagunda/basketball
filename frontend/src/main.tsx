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
// Prefetch critical first-tab data in parallel.
// Slate + Games are both needed by the Predict tab (default tab).
// Other tabs (Line, Parlay, Ben) load lazily on first visit — their endpoints
// can take 30-90s and hammering them all on cold start blocks the slate
// request and makes the app appear frozen for 1-3 minutes.
// ---------------------------------------------------------------------------
queryClient.prefetchQuery({ queryKey: ['slate'], queryFn: () => fetchJson('/api/slate', 30_000) });
queryClient.prefetchQuery({
  queryKey: ['games'],
  queryFn: async () => {
    const res = await fetchJson<{ data: unknown[] }>('/api/games', 10_000);
    return res.data ?? [];
  },
});

createRoot(document.getElementById('root')!).render(
  <QueryClientProvider client={queryClient}>
    <App />
  </QueryClientProvider>,
);
