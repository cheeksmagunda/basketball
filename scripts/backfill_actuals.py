#!/usr/bin/env python3
"""One-time backfill: write historical 'Highest Value' leaderboard data
into data/actuals/ CSVs + generate data/audit/ JSONs for Mar 5–17, 2026.

Run from repo root:  python scripts/backfill_actuals.py
"""

import csv
import io
import json
import os
import re
import unicodedata
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTUALS_DIR = os.path.join(REPO_ROOT, "data", "actuals")
AUDIT_DIR = os.path.join(REPO_ROOT, "data", "audit")
PREDICTIONS_DIR = os.path.join(REPO_ROOT, "data", "predictions")
SKIPPED_PATH = os.path.join(REPO_ROOT, "data", "skipped-uploads.json")

ACT_HEADER = "player_name,actual_rs,actual_card_boost,drafts,avg_finish,total_value,source"
PRED_FIELDS = ["scope", "lineup_type", "slot", "player_name", "player_id",
               "team", "pos", "predicted_rs", "est_card_boost", "pred_min",
               "pts", "reb", "ast", "stl", "blk"]

# ---------- raw data from user ----------

RAW_DATA = """\
3.17,Josh Hart,7.6,+1.1x,1st,47,23.5e
3.17,Bub Carrington,4.7,+2.4x,1st,6,20.9
3.17,Jose Alvarado,3.8,+2.8x,1st,30,18.4
3.17,Bones Hyland,3.4,+3.0x,1st,7,17.2
3.17,Jalen Duren,6.1,+0.7x,1st,232,16.4
3.17,Oso Ighodaro,3.3,+2.9x,1st,14,16.2
3.17,OG Anunoby,5.3,+1.0x,1st,69,16
3.17,Daniss Jenkins,3.4,+2.8x,2nd,4,15.8
3.17,Will Riley,2.9,+3.0x,1st,40,14.6
3.17,Christian Braun,3.7,+1.8x,1st,4,14.1
3.17,Kevin Porter Jr.,5.1,+0.6x,1st,26,13.2
3.17,Jaden McDaniels,3.9,+1.3x,1st,52,12.8
3.17,Pete Nance,3,+3.0x,5th,1,12.6
3.16,Nickeil Alexander-Walker,6.8,+0.9x,1st,293,19.6
3.16,Jordan Miller,3.8,+2.5x,1st,14,17.3
3.16,Matas Buzelis,5,+1.3x,1st,172,16.4
3.16,Naji Marshall,4.8,+1.3x,1st,41,16.2
3.16,Gui Santos,3.4,+2.5x,1st,243,15.5
3.16,De'Anthony Melton,4.1,+1.7x,1st,9,15.3
3.16,Kristaps Porzingis,5,+1.0x,1st,28,14.9
3.16,Gary Payton II,2.9,+3.0x,1st,60,14.7
3.16,Oso Ighodaro,3,+2.9x,1st,5,14.7
3.16,Will Riley,2.9,+3.0x,1st,13,14.4
3.16,Karlo Matković,2.8,+3.0x,1st,1,14.2
3.16,Devin Booker,5.5,+0.5x,1st,552,13.7
3.16,Leonard Miller,2.7,+3.0x,1st,30,13.6
3.15,Cody Williams,6,+3.0x,1st,78,30.1
3.15,Gary Payton II,4.3,+3.0x,1st,11,21.5
3.15,DeMar DeRozan,6.8,+0.9x,1st,627,20.4
3.15,Jakob Poeltl,5.4,+1.6x,1st,1,19.3
3.15,Bobby Portis,5.2,+1.6x,1st,21,18.7
3.15,Aaron Nesmith,4.8,+1.9x,1st,2,18.6
3.15,Killian Hayes,3.3,+3.0x,1st,7,16.5
3.15,Quinten Post,3.3,+3.0x,1st,4,16.4
3.15,Quentin Grimes,4.2,+1.7x,1st,470,15.5
3.15,Jarace Walker,3.5,+2.3x,1st,24,15
3.15,P.J. Washington,4.5,+1.5x,2nd,4,14.9
3.15,Naji Marshall,4.4,+1.4x,1st,20,14.9
3.15,Alex Caruso,2.9,+3.0x,1st,2,14.4
3.15,Julius Randle,5.5,+0.6x,1st,162,14.2
3.14,Precious Achiuwa,5.1,+2.2x,1st,104,21.6
3.14,Marcus Smart,4.2,+2.5x,1st,3,18.9
3.14,Neemias Queta,4.9,+1.5x,1st,56,17.1
3.14,Justin Edwards,3.4,+3.0x,1st,24,16.8
3.14,Daeqwon Plowden,3.3,+3.0x,1st,2,16.4
3.14,Tristan Vukcevic,3.2,+3.0x,1st,1,16.1
3.14,Maxime Raynaud,3.8,+2.1x,1st,146,15.5
3.14,Victor Wembanyama,7,+0.2x,1st,448,15.4
3.14,Quentin Grimes,3.8,+1.7x,1st,59,14.1
3.14,Jamir Watkins,2.7,+3.0x,1st,2,13.5
3.14,Jayson Tatum,4.6,+0.9x,1st,275,13.2
3.14,Ryan Rollins,4.5,+0.9x,1st,189,13.2
3.14,Austin Reaves,5.3,+0.5x,1st,177,13.2
3.13,Jalen Green,5.6,+1.6x,1st,728,21.3
3.13,Dejounte Murray,6.7,+1.1x,1st,231,20.7
3.13,Brice Sensabaugh,4.5,+2.2x,1st,136,18.9
3.13,Donovan Clingan,5.4,+1.0x,1st,186,16.2
3.13,Jalen Duren,5.7,+0.7x,1st,128,15.3
3.13,Anthony Edwards,6.5,+0.3x,1st,2000,15
3.13,Gui Santos,3.2,+2.6x,1st,146,14.8
3.13,Mitchell Robinson,3.5,+2.1x,1st,227,14.5
3.13,Javon Small,3.5,+2.0x,1st,119,14.3
3.13,Brandon Ingram,5.3,+0.7x,1st,382,14.3
3.13,Jrue Holiday,4.8,+0.9x,1st,32,13.9
3.13,Marcus Sasser,2.7,+3.0x,1st,2,13.6
3.13,Brandin Podziemski,3.9,+1.4x,1st,207,13.4
3.12,Khris Middleton,5.8,+2.3x,1st,2,24.8
3.12,Tristan da Silva,4.6,+2.6x,1st,36,21.1
3.12,Luka Dončić,9.3,,1st,2900,18.7
3.12,Jalen Green,4.9,+1.8x,1st,863,18.5
3.12,Pelle Larsson,4.6,+2.0x,1st,12,18.4
3.12,Zaccharie Risacher,3.9,+2.4x,1st,2,17.1
3.12,Devin Booker,6.7,+0.5x,1st,152,16.7
3.12,Spencer Jones,3.7,+3.0x,4th,3,16.3
3.12,Josh Minott,3.8,+3.0x,5th,2,16.1
3.12,Kasparas Jakucionis,3.2,+3.0x,1st,1,16
3.12,Justin Edwards,3.1,+3.0x,1st,2,15.6
3.12,Taylor Hendricks,3.1,+3.0x,1st,13,15.5
3.12,Daniel Gafford,3.6,+2.2x,1st,8,15.2
3.11,Jordan Clarkson,3.9,+3.0x,1st,5,19.5
3.11,Keon Ellis,3.8,+3.0x,1st,18,19.2
3.11,Kawhi Leonard,8.6,+0.2x,1st,1100,19
3.11,Nique Clifford,3.7,+3.0x,1st,60,18.5
3.11,DeMar DeRozan,6,+1.0x,1st,111,18.1
3.11,Brice Sensabaugh,4.2,+2.3x,1st,33,18.1
3.11,Desmond Bane,6.5,+0.7x,1st,374,17.5
3.11,Tristan da Silva,3.5,+2.6x,1st,23,16.3
3.11,Cameron Johnson,3.9,+2.1x,1st,4,15.9
3.11,Jordan Miller,3.3,+2.6x,1st,13,15.1
3.11,Christian Braun,3.7,+1.8x,1st,4,14.2
3.11,Miles Bridges,4.7,+1.0x,1st,68,14.2
3.11,Dejounte Murray,4.2,+1.3x,1st,45,13.7
3.10,Cameron Payne,7.7,+3.0x,1st,11,38.4
3.10,Bam Adebayo,12.7,+0.6x,1st,388,33.1
3.10,Matas Buzelis,6.5,+1.4x,1st,79,22.2
3.10,Kyle Kuzma,5.9,+1.7x,1st,1,21.8
3.10,Devin Carter,4.3,+3.0x,3rd,2,19.6
3.10,Ousmane Dieng,3.9,+3.0x,1st,12,19.5
3.10,Ron Harper Jr.,3.9,+3.0x,3rd,1,18.1
3.10,Kam Jones,3.5,+3.0x,1st,5,17.7
3.10,Aaron Nesmith,4.4,+1.9x,1st,12,17
3.10,Royce O'Neale,4.3,+2.0x,2nd,5,16.2
3.10,Myles Turner,4.3,+1.6x,1st,1,15.5
3.10,De'Aaron Fox,5.9,+0.6x,1st,34,15.4
3.10,Derrick White,5.7,+0.7x,1st,251,15.4
3.9,Jaylin Williams,5.2,+2.6x,1st,37,24
3.9,Shai Gilgeous-Alexander,9.5,,1st,2900,19.1
3.9,Karl-Anthony Towns,6.7,+0.6x,1st,666,17.4
3.9,Nolan Traoré,3.3,+3.0x,1st,25,16.6
3.9,Tim Hardaway Jr.,4.2,+1.9x,1st,46,16.4
3.9,Ochai Agbaji,3.8,+3.0x,5th,3,15.8
3.9,Gui Santos,3.3,+2.7x,1st,326,15.7
3.9,Kyle Filipowski,3.9,+2.0x,1st,642,15.5
3.9,Dean Wade,3,+3.0x,1st,1,15
3.9,Nikola Jokić,7.3,,1st,5300,14.6
3.9,Brook Lopez,2.9,+3.0x,1st,35,14.4
3.9,Derrick Jones Jr.,3.6,+2.0x,1st,29,14.4
3.9,Ajay Mitchell,4.3,+1.3x,1st,27,14.1
3.8,Scoot Henderson,5.3,+2.5x,1st,4,24
3.8,Daniel Gafford,4.8,+2.3x,1st,2,20.5
3.8,Malik Monk,5,+2.1x,1st,1,20.5
3.8,Collin Sexton,4.9,+1.9x,1st,73,18.9
3.8,Maxime Raynaud,4.4,+2.1x,1st,255,18
3.8,Paolo Banchero,6.3,+0.6x,1st,152,16.3
3.8,Baylor Scheierman,3.2,+3.0x,1st,2,16
3.8,Russell Westbrook,5,+1.1x,1st,85,15.5
3.8,RJ Barrett,4.9,+1.0x,1st,9,14.6
3.8,Jakob Poeltl,4,+1.6x,1st,17,14.5
3.8,Dylan Harper,3.8,+1.7x,1st,6,13.9
3.8,Jalen Green,3.1,+2.1x,1st,281,12.5
3.8,Victor Wembanyama,5.4,+0.3x,1st,2500,12.4
3.7,Taylor Hendricks,4.1,+3.0x,1st,1,20.5
3.7,Gui Santos,4.1,+2.9x,1st,191,20.2
3.7,Ousmane Dieng,3.1,+3.0x,1st,16,15.5
3.7,Quentin Grimes,3.8,+1.7x,1st,58,14.1
3.7,Isaiah Jackson,2.7,+3.0x,1st,2,13.5
3.7,Kelly Oubre Jr.,3.9,+1.5x,1st,15,13.5
3.7,Ziaire Williams,2.8,+3.0x,2nd,1,13.3
3.7,Cody Williams,2.6,+3.0x,1st,2,13
3.7,Kyle Filipowski,3.2,+2.0x,1st,598,12.7
3.7,Jalen Johnson,5.5,+0.3x,1st,734,12.6
3.7,Marcus Sasser,2.5,+3.0x,1st,1,12.5
3.7,Anthony Edwards,5.4,+0.3x,1st,1700,12.4
3.7,Tristan da Silva,2.6,+2.7x,1st,32,12.2
3.6,OG Anunoby,7.1,+1.1x,1st,88,22
3.6,Brook Lopez,4.3,+3.0x,1st,11,21.7
3.6,Tyler Herro,6.6,+0.8x,1st,16,18.4
3.6,Luka Dončić,8.3,,1st,3300,16.7
3.6,Amen Thompson,5.9,+0.6x,1st,170,15.9
3.6,Luke Kennard,3,+3.0x,1st,32,15.2
3.6,Kon Knueppel,5,+0.8x,1st,905,14.1
3.6,Grant Williams,2.9,+2.8x,1st,1,14.1
3.6,Jordan Miller,3,+2.5x,1st,3,13.4
3.6,Toumani Camara,3.7,+1.6x,1st,37,13.3
3.6,Brandon Miller,4.7,+0.8x,1st,134,13.1
3.6,Jay Huff,3.2,+2.0x,1st,66,12.9
3.6,Julian Champagnie,3.3,+1.9x,1st,16,12.7
3.5,Ace Bailey,5.9,+2.1x,1st,187,24.4
3.5,Precious Achiuwa,5.4,+2.3x,1st,231,23.2
3.5,Reed Sheppard,5.9,+1.5x,1st,693,20.8
3.5,Kyle Filipowski,4.8,+2.1x,1st,68,19.5
3.5,Tristan da Silva,4,+2.8x,1st,3,19.2
3.5,Julian Reese,3.7,+3.0x,1st,8,18.6
3.5,Collin Sexton,4.6,+2.0x,1st,23,18.3
3.5,Kel'el Ware,5.3,+1.4x,1st,111,18
3.5,Isaiah Collier,4.8,+1.5x,1st,144,16.7
3.5,Victor Wembanyama,7.1,+0.3x,1st,1400,16.4
3.5,Klay Thompson,3.5,+2.6x,1st,2,16
3.5,Cody Williams,3.3,+3.0x,2nd,1,15.8
3.5,Rudy Gobert,5,+1.1x,1st,50,15.6
"""

# ---------- helpers ----------

def _normalize_name(name: str) -> str:
    """Strip diacritics for matching (Dončić → Doncic)."""
    nfkd = unicodedata.normalize("NFKD", name)
    return nfkd.encode("ASCII", "ignore").decode("ASCII").strip().lower()


def _parse_drafts(raw: str) -> str:
    """Convert '2k' → '2000', '2.9k' → '2900', plain int stays."""
    raw = raw.strip()
    if raw.lower().endswith("k"):
        return str(int(float(raw[:-1]) * 1000))
    return raw


def _parse_multiplier(raw: str) -> str:
    """Convert '+1.1x' → '1.1', empty → ''."""
    raw = raw.strip()
    if not raw:
        return ""
    return raw.replace("+", "").replace("x", "")


def _parse_total_value(raw: str) -> str:
    """Strip trailing 'e' from values like '23.5e'."""
    raw = raw.strip()
    if raw.lower().endswith("e"):
        raw = raw[:-1]
    return raw


def _csv_escape(v: str) -> str:
    """Quote CSV field if it contains comma, quote, or newline."""
    s = str(v) if v is not None else ""
    if "," in s or '"' in s or "\n" in s:
        return '"' + s.replace('"', '""') + '"'
    return s


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# ---------- parse raw data ----------

def parse_raw_data() -> dict:
    """Returns {date_str: [player_dicts]} grouped by date."""
    result = {}
    for line in RAW_DATA.strip().splitlines():
        parts = line.split(",")
        if len(parts) < 7:
            continue
        date_raw = parts[0].strip()           # e.g. "3.5"
        month, day = date_raw.split(".")
        date_str = f"2026-{int(month):02d}-{int(day):02d}"

        player = {
            "player_name": parts[1].strip(),
            "actual_rs": parts[2].strip(),
            "actual_card_boost": _parse_multiplier(parts[3]),
            "drafts": _parse_drafts(parts[5]),
            "avg_finish": "",
            "total_value": _parse_total_value(parts[6]),
            "source": "highest_value",
        }
        result.setdefault(date_str, []).append(player)
    return result


# ---------- write actuals ----------

def write_actuals(date_str: str, new_players: list):
    """Write actuals CSV, preserving non-highest_value rows for 03-10."""
    path = os.path.join(ACTUALS_DIR, f"{date_str}.csv")
    existing_rows = []

    # For 2026-03-10, preserve starting_5 and real_scores rows
    if os.path.exists(path) and date_str == "2026-03-10":
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("source") != "highest_value":
                    existing_rows.append(row)

    # Dedup: remove existing players that appear in new data
    new_names = {_normalize_name(p["player_name"]) for p in new_players}
    existing_rows = [r for r in existing_rows
                     if _normalize_name(r.get("player_name", "")) not in new_names]

    # Build CSV
    fields = ["player_name", "actual_rs", "actual_card_boost", "drafts",
              "avg_finish", "total_value", "source"]
    lines = [ACT_HEADER]
    for row in existing_rows:
        lines.append(",".join(_csv_escape(row.get(f, "")) for f in fields))
    for p in new_players:
        lines.append(",".join(_csv_escape(p.get(f, "")) for f in fields))

    os.makedirs(ACTUALS_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    total = len(existing_rows) + len(new_players)
    print(f"  ✓ {path} ({total} rows, {len(existing_rows)} preserved, {len(new_players)} new)")


# ---------- compute audit ----------

def compute_audit(date_str: str) -> dict | None:
    """Replicate _compute_audit from api/index.py (lines 4053-4108)."""
    pred_path = os.path.join(PREDICTIONS_DIR, f"{date_str}.csv")
    act_path = os.path.join(ACTUALS_DIR, f"{date_str}.csv")

    if not os.path.exists(pred_path) or not os.path.exists(act_path):
        return None

    # Parse predictions
    preds = []
    with open(pred_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            preds.append(row)

    # Parse actuals
    actuals = []
    with open(act_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            actuals.append(row)

    # Build actuals map (normalized name → row)
    act_map = {}
    for r in actuals:
        key = _normalize_name(r.get("player_name", ""))
        if key:
            act_map[key] = r

    errors, dir_hits, misses = [], [], []
    for row in preds:
        pname = _normalize_name(row.get("player_name", ""))
        pred_rs = _safe_float(row.get("predicted_rs"))
        if pname not in act_map or pred_rs <= 0:
            continue
        a = act_map[pname]
        actual_rs = _safe_float(a.get("actual_rs"))
        if actual_rs <= 0:
            continue
        err = actual_rs - pred_rs
        errors.append(abs(err))
        dir_hits.append(1 if (err >= 0) == (pred_rs >= 3.0) else 0)
        misses.append({
            "player": row.get("player_name", ""),
            "team": row.get("team", ""),
            "predicted_rs": round(pred_rs, 2),
            "actual_rs": round(actual_rs, 2),
            "error": round(err, 2),
            "drafts": a.get("drafts", ""),
            "actual_card_boost": a.get("actual_card_boost", ""),
        })

    if not errors:
        return None

    misses.sort(key=lambda x: abs(x["error"]), reverse=True)
    mae = round(sum(errors) / len(errors), 3)
    dir_acc = round(sum(dir_hits) / len(dir_hits), 3) if dir_hits else None

    over = [e for e in misses if e["error"] < 0]
    under = [e for e in misses if e["error"] > 0]

    return {
        "date": date_str,
        "players_compared": len(errors),
        "mae": mae,
        "directional_accuracy": dir_acc,
        "over_projected": len(over),
        "under_projected": len(under),
        "biggest_misses": misses[:8],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def write_audit(date_str: str):
    """Compute and write audit JSON."""
    audit = compute_audit(date_str)
    if not audit:
        print(f"  ⚠ No audit for {date_str} (no matching predictions/actuals)")
        return

    os.makedirs(AUDIT_DIR, exist_ok=True)
    path = os.path.join(AUDIT_DIR, f"{date_str}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print(f"  ✓ {path} (MAE={audit['mae']}, {audit['players_compared']} players)")


# ---------- update skipped uploads ----------

def update_skipped_uploads(dates: list):
    """Remove backfilled dates from skipped-uploads.json."""
    if not os.path.exists(SKIPPED_PATH):
        print("  ⚠ No skipped-uploads.json found")
        return

    with open(SKIPPED_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    before = len(data.get("skipped_dates", []))
    data["skipped_dates"] = [d for d in data.get("skipped_dates", [])
                             if d not in dates]
    data["last_updated"] = datetime.now(timezone.utc).isoformat()
    after = len(data["skipped_dates"])

    with open(SKIPPED_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  ✓ skipped-uploads.json: {before} → {after} skipped dates")


# ---------- main ----------

def main():
    print("=== Backfilling Highest Value actuals (Mar 5-17, 2026) ===\n")

    data = parse_raw_data()
    dates = sorted(data.keys())
    print(f"Parsed {sum(len(v) for v in data.values())} players across {len(dates)} dates\n")

    print("--- Writing actuals CSVs ---")
    for date_str in dates:
        write_actuals(date_str, data[date_str])

    print("\n--- Generating audit JSONs ---")
    for date_str in dates:
        write_audit(date_str)

    print("\n--- Updating skipped-uploads.json ---")
    update_skipped_uploads(dates)

    print(f"\n=== Done! {len(dates)} dates backfilled. ===")
    print("Next: git add data/actuals/ data/audit/ data/skipped-uploads.json && git commit")


if __name__ == "__main__":
    main()
