"""
Train binary leaderboard classifier: P(player appears on daily top-performer leaderboard).

v62: Positives from data/top_performers.csv; negatives from players in data/actuals/*.csv (or synced per-day files) who are not top-performer rows for that date.
Features are all available at pre-game prediction time.

Output: leaderboard_clf.pkl — loaded by api/index.py for core pool scoring.
"""
import os
import re
import glob
import pickle
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    print("[ERROR] lightgbm not installed. Run: pip install lightgbm")
    import sys
    sys.exit(1)


def _normalize_name(name):
    n = str(name).lower().strip()
    n = re.sub(r"['\.\-]", "", n)
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", n)
    return re.sub(r"\s+", " ", n).strip()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOP_PERF_PATH = os.path.join(BASE_DIR, "data", "top_performers.csv")
ACTUALS_DIR = os.path.join(BASE_DIR, "data", "actuals")

# 1. Load positive examples (leaderboard entries)
print("1. Loading top performers (positive class)...")
if not os.path.exists(TOP_PERF_PATH):
    print(f"[ERROR] {TOP_PERF_PATH} not found")
    import sys
    sys.exit(1)

tp = pd.read_csv(TOP_PERF_PATH)
tp["norm_name"] = tp["player_name"].apply(_normalize_name)
tp["date"] = pd.to_datetime(tp["date"])
tp["actual_card_boost"] = pd.to_numeric(tp["actual_card_boost"], errors="coerce")
tp["drafts"] = pd.to_numeric(tp["drafts"], errors="coerce")
tp["is_leaderboard"] = 1
print(f"   {len(tp)} leaderboard entries across {tp['date'].nunique()} dates")

# 2. Load actuals (all players with actual RS)
print("2. Loading actuals (for negative class)...")
actuals_frames = []
for fpath in glob.glob(os.path.join(ACTUALS_DIR, "*.csv")):
    date_str = os.path.basename(fpath).replace(".csv", "")
    try:
        a = pd.read_csv(fpath)
        if "player_name" not in a.columns or "actual_rs" not in a.columns:
            continue
        a["date"] = pd.to_datetime(date_str)
        a["norm_name"] = a["player_name"].apply(_normalize_name)
        actuals_frames.append(a)
    except Exception:
        pass

if not actuals_frames:
    print("[ERROR] No actuals data found")
    import sys
    sys.exit(1)

actuals = pd.concat(actuals_frames, ignore_index=True)
actuals["date"] = actuals["date"].dt.normalize()
tp["date"] = tp["date"].dt.normalize()

# 3. Build training data
print("3. Building training data...")
# Mark leaderboard entries
tp_keys = set(zip(tp["norm_name"], tp["date"]))
actuals["is_leaderboard"] = [
    1 if (nn, d) in tp_keys else 0
    for nn, d in zip(actuals["norm_name"], actuals["date"])
]

# Only keep dates that have leaderboard data
leaderboard_dates = set(tp["date"])
actuals = actuals[actuals["date"].isin(leaderboard_dates)].copy()
print(f"   {len(actuals)} actuals entries on leaderboard dates")
print(f"   {actuals['is_leaderboard'].sum()} positives, {(actuals['is_leaderboard'] == 0).sum()} negatives")

# Features we can construct from actuals data
# Note: at inference time, these come from projections, not actuals
# We use actuals here to train the "ideal" separator
actuals["actual_rs"] = pd.to_numeric(actuals.get("actual_rs", 0), errors="coerce").fillna(0)
actuals["actual_card_boost"] = pd.to_numeric(actuals.get("actual_card_boost", pd.Series(dtype=float)), errors="coerce").fillna(1.5)

# Merge top_performers boost data for positive examples
tp_boost = tp[["norm_name", "date", "actual_card_boost", "drafts", "total_value"]].copy()
tp_boost = tp_boost.rename(columns={
    "actual_card_boost": "tp_boost",
    "drafts": "tp_drafts",
    "total_value": "tp_value",
})
actuals = actuals.merge(tp_boost, on=["norm_name", "date"], how="left")
actuals["actual_card_boost"] = actuals["tp_boost"].fillna(actuals["actual_card_boost"])

# Construct features
actuals["projected_value"] = actuals["actual_rs"] * (2.0 + actuals["actual_card_boost"])
actuals["log_drafts"] = np.log10(actuals.get("tp_drafts", pd.Series(dtype=float)).fillna(100).clip(lower=1))

# Use available columns
feature_cols = ["actual_rs", "actual_card_boost", "projected_value"]

# Add additional features if available
for col in ["season_pts", "season_min", "recent_pts", "recent_min"]:
    if col in actuals.columns:
        feature_cols.append(col)
        actuals[col] = pd.to_numeric(actuals[col], errors="coerce").fillna(0)

feature_cols.append("log_drafts")

# Drop rows with missing features
actuals = actuals.dropna(subset=feature_cols + ["is_leaderboard"])
print(f"   Training with {len(feature_cols)} features: {feature_cols}")

X = actuals[feature_cols].values
y = actuals["is_leaderboard"].values

# 4. Train classifier
print(f"4. Training LightGBM classifier ({len(X)} samples, {y.sum()} positive)...")
clf = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    num_leaves=32,
    random_state=42,
    subsample=0.85,
    colsample_bytree=0.85,
    class_weight="balanced",
    verbose=-1,
)
clf.fit(X, y)

# 5. Evaluate
from sklearn.metrics import classification_report, roc_auc_score
y_pred = clf.predict(X)
y_prob = clf.predict_proba(X)[:, 1]
print("\n5. Training set evaluation:")
print(classification_report(y, y_pred, target_names=["non-leaderboard", "leaderboard"]))
print(f"   ROC AUC: {roc_auc_score(y, y_prob):.4f}")

# 6. Save model
bundle = {
    "model": clf,
    "features": feature_cols,
    "n_features": len(feature_cols),
    "training_samples": len(X),
    "positive_count": int(y.sum()),
}
out_path = os.path.join(BASE_DIR, "leaderboard_clf.pkl")
print(f"\n6. Saving to {out_path}...")
with open(out_path, "wb") as f:
    pickle.dump(bundle, f)

print("\nDone! Feature importances:")
for feat, imp in sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat}: {imp}")
