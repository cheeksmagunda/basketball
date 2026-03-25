#!/usr/bin/env python3
"""
Lightweight smoke harness for parlay SSE reconnect/state behavior.

Usage:
  python3 scripts/parlay_sse_smoke.py --base-url https://the-oracle.up.railway.app
  python3 scripts/parlay_sse_smoke.py --base-url http://127.0.0.1:8000 --cycles 5 --ticks 3
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from typing import Dict, Optional


def _get_json(url: str, timeout_s: float = 15.0) -> Dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _read_one_sse_payload(url: str, timeout_s: float = 20.0) -> Optional[Dict]:
    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        start = time.time()
        data_lines = []
        while True:
            raw = resp.readline()
            if not raw:
                return None
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
            elif line == "":
                if data_lines:
                    joined = "\n".join(data_lines)
                    try:
                        return json.loads(joined)
                    except json.JSONDecodeError:
                        return None
                    finally:
                        data_lines = []
            if time.time() - start > timeout_s:
                return None


def _ticket_signature(payload: Dict) -> str:
    legs = payload.get("legs") or []
    if not legs:
        return "no-legs"
    parts = []
    for leg in legs:
        parts.append(
            f"{leg.get('player_name','?')}|{leg.get('stat_type','?')}|"
            f"{leg.get('direction','?')}|{leg.get('line','?')}"
        )
    return " || ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Parlay SSE reconnect smoke test")
    parser.add_argument("--base-url", required=True, help="API base URL, e.g. https://the-oracle.up.railway.app")
    parser.add_argument("--cycles", type=int, default=3, help="Reconnect cycles to run")
    parser.add_argument("--ticks", type=int, default=2, help="SSE events to read per cycle")
    parser.add_argument("--sleep-ms", type=int, default=500, help="Pause between reconnect cycles")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    parlay_url = f"{base}/api/parlay"
    sse_url = f"{base}/api/parlay-live-stream"

    print(f"[smoke] fetching baseline parlay: {parlay_url}")
    baseline = _get_json(parlay_url)
    base_sig = _ticket_signature(baseline)
    print(f"[smoke] baseline signature: {base_sig}")

    failures = 0
    for cycle in range(1, args.cycles + 1):
        print(f"[smoke] cycle {cycle}/{args.cycles}: connect SSE")
        for tick_idx in range(1, args.ticks + 1):
            payload = _read_one_sse_payload(sse_url)
            if not payload:
                failures += 1
                print(f"[smoke]   tick {tick_idx}: no parseable payload")
                continue
            sig = _ticket_signature(payload)
            if sig != "no-legs" and base_sig != "no-legs" and sig != base_sig:
                failures += 1
                print(f"[smoke]   tick {tick_idx}: signature mismatch")
                print(f"[smoke]     baseline: {base_sig}")
                print(f"[smoke]     sse:      {sig}")
            else:
                print(f"[smoke]   tick {tick_idx}: ok ({payload.get('date', '?')})")
        time.sleep(max(args.sleep_ms, 0) / 1000.0)

    if failures:
        print(f"[smoke] FAIL: {failures} issue(s) detected")
        return 1
    print("[smoke] PASS: no mismatches detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
