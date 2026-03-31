// ============================================================================
// useEventSource
// Generic hook for Server-Sent Events (SSE).
// Used by the Parlay live stream (/api/parlay-live-stream) and any
// future SSE endpoints.
// ============================================================================

import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Opens an EventSource when `url` is truthy. Closes on cleanup or url change.
 * Parses JSON from event.data and calls onMessage.
 *
 * @param url       - SSE endpoint URL, or null to stay closed
 * @param onMessage - callback receiving the parsed JSON payload
 * @param opts      - optional configuration (retryInterval)
 * @returns { connected } - whether the EventSource is currently connected
 */
export function useEventSource<T>(
  url: string | null,
  onMessage: (data: T) => void,
  _opts?: { retryInterval?: number },
) {
  const [connected, setConnected] = useState(false);
  const esRef = useRef<EventSource | null>(null);

  // Keep latest callback in ref so the effect doesn't re-run on every render.
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  useEffect(() => {
    if (!url) return;

    const es = new EventSource(url);
    esRef.current = es;

    es.onopen = () => setConnected(true);

    es.onmessage = (e) => {
      try {
        onMessageRef.current(JSON.parse(e.data));
      } catch {
        // Non-JSON message (e.g. keepalive comment) -- ignore silently
      }
    };

    es.onerror = () => {
      setConnected(false);
    };

    return () => {
      es.close();
      esRef.current = null;
      setConnected(false);
    };
  }, [url]);

  /** Manually close the SSE connection. */
  const stop = useCallback(() => {
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
      setConnected(false);
    }
  }, []);

  return { connected, stop };
}
