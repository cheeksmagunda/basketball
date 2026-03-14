"""
Sync _CONFIG_DEFAULTS from api/index.py into the live data/model-config.json.

Run by the sync-model-config.yml GitHub Action on every push to main.
- Deep-merges missing keys from code defaults into live config (never overwrites Ben's tuned values)
- Stamps code_sha and code_pushed_at
- Bumps version + appends changelog entry if anything changed
- Writes data/model-config.json back to disk (workflow commits if diff exists)
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "data" / "model-config.json"


def extract_defaults():
    """Import _CONFIG_DEFAULTS from api/index.py via subprocess to avoid import side-effects."""
    result = subprocess.run(
        [
            sys.executable, "-c",
            "import sys; sys.path.insert(0, '.'); "
            "from api.index import _CONFIG_DEFAULTS; "
            "import json; print(json.dumps(_CONFIG_DEFAULTS))",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        print(f"ERROR extracting _CONFIG_DEFAULTS:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return json.loads(result.stdout)


def deep_merge_missing(defaults, live):
    """Add keys from defaults that are missing in live. Never overwrites existing values.

    Returns list of dot-notation keys that were added.
    """
    added = []
    for key, val in defaults.items():
        if key not in live:
            live[key] = val
            added.append(key)
        elif isinstance(val, dict) and isinstance(live.get(key), dict):
            sub_added = deep_merge_missing(val, live[key])
            added.extend(f"{key}.{k}" for k in sub_added)
    return added


def main():
    # Load current live config
    if not CONFIG_PATH.exists():
        print(f"ERROR: {CONFIG_PATH} not found — cannot sync.", file=sys.stderr)
        sys.exit(1)

    with open(CONFIG_PATH) as f:
        live = json.load(f)

    # Extract defaults from code
    defaults = extract_defaults()

    # Deep-merge missing keys
    added = deep_merge_missing(defaults, live)

    # Stamp code_sha and code_pushed_at from env (set by workflow)
    commit_sha = os.environ.get("COMMIT_SHA", "")
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    sha_changed = live.get("code_sha") != commit_sha
    live["code_sha"] = commit_sha
    live["code_pushed_at"] = now_iso

    if not added and not sha_changed:
        print("No new config keys to add. SHA stamps updated. No version bump needed.")
        # Still write back to update timestamps
        with open(CONFIG_PATH, "w") as f:
            json.dump(live, f, indent=2)
            f.write("\n")
        return

    if added:
        # Bump version
        current_version = live.get("version", 1)
        new_version = current_version + 1
        live["version"] = new_version

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        changelog_entry = {
            "version": new_version,
            "date": date_str,
            "change": f"Auto-sync from code push [{commit_sha[:8]}]: added missing keys: {', '.join(added)}",
        }
        if "changelog" not in live:
            live["changelog"] = []
        live["changelog"].append(changelog_entry)
        print(f"Added {len(added)} missing key(s): {added}")
        print(f"Bumped config version {current_version} → {new_version}")
    else:
        print("No new config keys. SHA stamps updated.")

    with open(CONFIG_PATH, "w") as f:
        json.dump(live, f, indent=2)
        f.write("\n")

    print(f"Wrote {CONFIG_PATH}")


if __name__ == "__main__":
    main()
