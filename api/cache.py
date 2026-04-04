"""
Redis-backed cache layer for the Oracle API.

Drop-in replacement for the /tmp file-based _cg/_cs helpers.
Redis is primary; /tmp files are the automatic fallback when Redis is
unavailable (cold start, network blip, local dev without Redis).

Environment:
    REDIS_URL — full redis:// connection string (set by Railway Redis plugin).
                When absent the module is inert and all calls fall through to
                the file-based helpers unchanged.

Usage in api/index.py:
    from api.cache import rcg, rcs, rflush, redis_ok

    # Read (returns parsed JSON or None)
    data = rcg(key, date_str)

    # Write (stores JSON with optional TTL in seconds)
    rcs(key, value, date_str, ttl=300)

    # Flush all keys in the cache namespace
    rflush()

    # Health probe
    redis_ok()  -> bool
"""

import json
import os
import hashlib
import time
import threading

_REDIS_URL = os.getenv("REDIS_URL", "")
_PREFIX = "oracle:"  # namespace prefix to avoid key collisions

_r = None  # redis.Redis client (lazy-init)
_r_available = None  # tri-state: None = not tried, True/False = last probe result
_r_last_check = 0.0
_RECONNECT_INTERVAL = 10  # seconds between reconnect attempts after failure
_r_lock = threading.Lock()  # guards reconnect cooldown check + state mutation


def _redis_client():
    """Lazy-initialise and return the Redis client, or None if unavailable."""
    global _r, _r_available, _r_last_check

    if not _REDIS_URL:
        _r_available = False
        return None

    # Fast path: healthy client already exists (no lock needed for read)
    if _r is not None and _r_available:
        return _r

    with _r_lock:
        # Re-check inside the lock to avoid redundant reconnects from concurrent threads
        if _r is not None and _r_available:
            return _r

        # Rate-limit reconnect attempts
        now = time.time()
        if _r_available is False and (now - _r_last_check) < _RECONNECT_INTERVAL:
            return None

        try:
            import redis as _redis_mod
            _r = _redis_mod.from_url(
                _REDIS_URL,
                decode_responses=True,
                max_connections=100,
                socket_connect_timeout=3,
                socket_timeout=3,
                socket_keepalive=True,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            _r.ping()
            _r_available = True
            _r_last_check = now
            return _r
        except Exception as e:
            print(f"[cache] Redis unavailable: {e}")
            _r_available = False
            _r_last_check = now
            return None


def _make_key(k: str, date_str: str | None = None) -> str:
    """Build a namespaced Redis key that mirrors the file-based MD5 scheme."""
    # Keep the same hash so that key semantics match between Redis and /tmp
    d = date_str or ""  # caller should pass _today_str(); empty string is safe fallback
    raw = f"{d}:{k}"
    hashed = hashlib.md5(raw.encode()).hexdigest()
    return f"{_PREFIX}{hashed}"


# ── Public API ────────────────────────────────────────────────────────────────

def rcg(key: str, date_str: str | None = None):
    """Redis Cache GET — returns parsed JSON value or None."""
    client = _redis_client()
    if client is None:
        return None
    try:
        raw = client.get(_make_key(key, date_str))
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as e:
        print(f"[cache] Redis GET error for {key}: {e}")
        return None


def rcs(key: str, value, date_str: str | None = None, ttl: int | None = None):
    """Redis Cache SET — stores JSON-serialised value with optional TTL (seconds)."""
    client = _redis_client()
    if client is None:
        return
    try:
        rkey = _make_key(key, date_str)
        payload = json.dumps(value)
        if ttl and ttl > 0:
            client.setex(rkey, ttl, payload)
        else:
            # Default 24h expiry to prevent unbounded growth
            client.setex(rkey, 86400, payload)
    except Exception as e:
        print(f"[cache] Redis SET error for {key}: {e}")


def rcd(key: str, date_str: str | None = None):
    """Redis Cache DELETE — remove a single key. Safe no-op if Redis is down."""
    client = _redis_client()
    if client is None:
        return False
    try:
        return bool(client.delete(_make_key(key, date_str)))
    except Exception as e:
        print(f"[cache] Redis DELETE error for {key}: {e}")
        return False


def rflush():
    """Flush all keys under our namespace prefix. Safe no-op if Redis is down."""
    client = _redis_client()
    if client is None:
        return 0
    try:
        cursor, keys = 0, []
        while True:
            cursor, batch = client.scan(cursor, match=f"{_PREFIX}*", count=200)
            keys.extend(batch)
            if cursor == 0:
                break
        if keys:
            client.delete(*keys)
        return len(keys)
    except Exception as e:
        print(f"[cache] Redis FLUSH error: {e}")
        return 0


def redis_ok() -> bool:
    """Health probe — True if Redis is reachable and responding."""
    client = _redis_client()
    if client is None:
        return False
    try:
        return client.ping()
    except Exception:
        return False
