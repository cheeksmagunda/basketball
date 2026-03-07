"""
Stamp model_retrained_at in data/model-config.json after a successful retrain.

Run by retrain-model.yml only when lgbm_model.pkl changed.
Bumps version + appends changelog entry so the retrain event is traceable.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "data" / "model-config.json"


def main():
    if not CONFIG_PATH.exists():
        print(f"ERROR: {CONFIG_PATH} not found.", flush=True)
        raise SystemExit(1)

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    current_version = config.get("version", 1)
    new_version = current_version + 1

    config["version"] = new_version
    config["model_retrained_at"] = now_iso
    config["updated_at"] = now_iso
    config["updated_by"] = "retrain-model-action"

    changelog_entry = {
        "version": new_version,
        "date": date_str,
        "change": f"Auto-bump after model retrain: lgbm_model.pkl updated on {date_str}",
    }
    if "changelog" not in config:
        config["changelog"] = []
    config["changelog"].append(changelog_entry)

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print(f"Config bumped v{current_version} → v{new_version}, model_retrained_at={now_iso}")


if __name__ == "__main__":
    main()
