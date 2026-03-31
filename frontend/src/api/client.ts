// ============================================================================
// API Client — fetchWithTimeout + fetchJson
// Port of the vanilla JS fetchWithTimeout from app.js (lines 34-66).
// ============================================================================

const DEFAULT_TIMEOUT_MS = 10_000;

/**
 * Fetch with a hard timeout via AbortController.
 *
 * - On timeout: throws Error("Request timed out -- tap Retry")
 * - On externalSignal abort (tab switch): throws DOMException("Tab switched", "AbortError")
 * - Composes signals via AbortSignal.any when available, with a manual fallback
 *   for older browsers that lack AbortSignal.any.
 */
export async function fetchWithTimeout(
  url: string,
  options: RequestInit = {},
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
  externalSignal?: AbortSignal,
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  // Compose timeout signal + external signal (tab switch abort)
  let signal: AbortSignal = controller.signal;

  if (externalSignal) {
    if (typeof AbortSignal.any === 'function') {
      signal = AbortSignal.any([controller.signal, externalSignal]);
    } else {
      // Fallback for older browsers: forward external abort to our controller
      if (externalSignal.aborted) {
        controller.abort();
      } else {
        externalSignal.addEventListener(
          'abort',
          () => controller.abort(),
          { once: true },
        );
      }
    }
  }

  try {
    const response = await fetch(url, { ...options, signal });
    clearTimeout(timeoutId);
    return response;
  } catch (error: unknown) {
    clearTimeout(timeoutId);

    if (error instanceof DOMException && error.name === 'AbortError') {
      // Distinguish tab-switch abort (silent) from timeout (user-facing)
      if (externalSignal?.aborted) {
        throw new DOMException('Tab switched', 'AbortError');
      }
      throw new Error('Request timed out \u2014 tap Retry');
    }

    throw error;
  }
}

/**
 * Convenience wrapper: fetchWithTimeout + .ok check + .json() parse.
 * Throws on non-OK HTTP status or timeout.
 */
export async function fetchJson<T>(
  url: string,
  timeout: number = DEFAULT_TIMEOUT_MS,
): Promise<T> {
  const response = await fetchWithTimeout(url, {}, timeout);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}
