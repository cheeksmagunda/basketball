"""
Weekly MAE drift check helper.

This script triggers the backend-only endpoint `/api/mae-drift-check` and prints
the returned payload. Railway can call it directly (or call the endpoint),
while the backend remains the source of truth for computation + flag writing.
"""

import os
import requests


def main() -> None:
    port = os.environ.get("PORT", "8080")
    url = f"http://localhost:{port}/api/mae-drift-check"

    headers = {}
    cron_secret = os.environ.get("CRON_SECRET")
    if cron_secret:
        headers["Authorization"] = f"Bearer {cron_secret}"

    r = requests.get(url, headers=headers, timeout=45)
    try:
        data = r.json()
        print(f"status={r.status_code} payload={data}")
    except Exception:
        print(f"status={r.status_code} body={r.text[:500]}")


if __name__ == "__main__":
    main()

