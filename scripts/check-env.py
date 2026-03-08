#!/usr/bin/env python3
"""Verify required and optional env vars for local or deployed runs. Exit 1 if any required var is missing."""
import os
import sys
from pathlib import Path

# Load .env from repo root when running locally
_root = Path(__file__).resolve().parent.parent
_env = _root / ".env"
if _env.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env)
    except ImportError:
        pass

REQUIRED = ["GITHUB_TOKEN", "GITHUB_REPO", "ANTHROPIC_API_KEY"]
OPTIONAL = ["ODDS_API_KEY", "CRON_SECRET", "DOCS_SECRET", "VERCEL_GIT_COMMIT_SHA"]

def main():
    missing = []
    for name in REQUIRED:
        val = os.getenv(name)
        if not (val and str(val).strip()):
            missing.append(name)
            print(f"  ✗ {name} (required) — missing")
        else:
            print(f"  ✓ {name} (required)")

    for name in OPTIONAL:
        val = os.getenv(name)
        if not (val and str(val).strip()):
            print(f"  ○ {name} (optional) — not set")
        else:
            print(f"  ✓ {name} (optional)")

    if missing:
        print(f"\nMissing required: {', '.join(missing)}. Set them in .env or the environment.")
        sys.exit(1)
    print("\nAll required env vars present.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
