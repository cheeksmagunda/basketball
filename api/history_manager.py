import json
from pathlib import Path
from datetime import datetime

HISTORY_FILE = Path(__file__).parent.parent / "data" / "lineup_history.json"
HISTORY_FILE.parent.mkdir(exist_ok=True)

def save_lineup_to_history(lineup_data, mode="slate"):
    """Saves generated lineups with their predicted ratings."""
    history = []
    if HISTORY_FILE.exists():
        try:
            history = json.loads(HISTORY_FILE.read_text())
        except: history = []

    entry = {
        "timestamp": datetime.now().isoformat(),
        "date": datetime.now().date().isoformat(),
        "mode": mode,
        "lineup": lineup_data
    }
    history.append(entry)
    # Keep last 50 entries
    HISTORY_FILE.write_text(json.dumps(history[-50:], indent=2))

def get_history():
    if not HISTORY_FILE.exists(): return []
    return json.loads(HISTORY_FILE.read_text())
