import os
import re
import glob
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from nba_api.stats.endpoints import playergamelogs
from sklearn.model_selection import train_test_split

# Pull 3 seasons of NBA game logs
SEASONS = ['2023-24', '2024-25', '2025-26']

print(f"1. Fetching NBA game logs for {len(SEASONS)} seasons...")
frames = []
for season in SEASONS:
    print(f"   Fetching {season}...")
    logs = playergamelogs.PlayerGameLogs(season_nullable=season)
    df_s = logs.get_data_frames()[0]
    df_s['SEASON'] = season
    frames.append(df_s)
    print(f"   Got {len(df_s)} game logs for {season}")
    if season != SEASONS[-1]:
        time.sleep(3)  # Avoid Cloudflare rate limiting

df = pd.concat(frames, ignore_index=True)
print(f"Total: {len(df)} game logs across {len(SEASONS)} seasons. Engineering features...")

# Sort by player, season, and date to calculate accurate "past" stats
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'].astype(str).str[:10])
df = df.sort_values(by=['PLAYER_ID', 'SEASON', 'GAME_DATE'])

# ─────────────────────────────────────────────────────────────────────────────
# TARGET: Observed Real Scores from data/actuals/ (true target)
# Falls back to formula target if no actuals are available.
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_name(name):
    n = str(name).lower().strip()
    n = re.sub(r"['\.\-]", "", n)
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", n)
    return re.sub(r"\s+", " ", n).strip()

actuals_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "actuals")
actuals_frames = []
for fpath in glob.glob(os.path.join(actuals_dir, "*.csv")):
    date_str = os.path.basename(fpath).replace(".csv", "")
    try:
        a = pd.read_csv(fpath)
        if "player_name" not in a.columns or "actual_rs" not in a.columns:
            continue
        a = a[["player_name", "actual_rs"]].copy()
        a["date_key"] = pd.to_datetime(date_str)
        a["norm_name"] = a["player_name"].apply(_normalize_name)
        actuals_frames.append(a)
    except Exception as e:
        print(f"   [WARN] Could not load {fpath}: {e}")

if actuals_frames:
    actuals_df = pd.concat(actuals_frames, ignore_index=True).dropna(subset=["actual_rs"])
    print(f"   Loaded {len(actuals_df)} actuals rows from {len(actuals_frames)} files.")
    df["norm_name"] = df["PLAYER_NAME"].apply(_normalize_name)
    df["date_key"] = df["GAME_DATE"].dt.normalize()
    actuals_df["date_key"] = actuals_df["date_key"].dt.normalize()
    df = df.merge(actuals_df[["norm_name", "date_key", "actual_rs"]],
                  on=["norm_name", "date_key"], how="inner")
    print(f"   After merge with actuals: {len(df)} training samples.")
    target = "actual_rs"
else:
    print("   [WARN] No actuals found — falling back to formula target.")
    df["actual_base_score"] = (
        df["PTS"] + df["REB"] + (df["AST"] * 1.5) +
        (df["STL"] * 4.5) + (df["BLK"] * 4.0) - (df["TOV"] * 1.2)
    )
    target = "actual_base_score"

# ─────────────────────────────────────────────────────────────────────────────
# FEATURES — all computed as "what was known BEFORE this game" (shift(1))
# ─────────────────────────────────────────────────────────────────────────────

g = df.groupby(['PLAYER_ID', 'SEASON'])

# Feature 1: Season avg minutes (before this game)
df['avg_min'] = g['MIN'].transform(lambda x: x.expanding().mean().shift(1))

# Feature 2: Season avg points (before this game)
df['avg_pts'] = g['PTS'].transform(lambda x: x.expanding().mean().shift(1))

# Feature 3: Recent form — rolling 5-game minutes avg
df['recent_min'] = g['MIN'].transform(lambda x: x.rolling(5).mean().shift(1))

# Feature 4: Usage trend (recent/season minutes ratio)
# Clipping constants must match api/index.py inference
USAGE_TREND_MIN, USAGE_TREND_MAX = 0.90, 1.50
df = df.dropna(subset=['avg_min', 'avg_pts', 'recent_min'])
df['usage_trend'] = np.where(df['avg_min'] > 0, df['recent_min'] / df['avg_min'], 1.0)
df['usage_trend'] = df['usage_trend'].clip(USAGE_TREND_MIN, USAGE_TREND_MAX)

# Feature 5: Opponent defensive rating — avg points scored against each team
# (actual points allowed per player, a real measure of defensive permissiveness)
opp_pts_allowed = df.groupby('OPP_TEAM')['PTS'].mean().to_dict()
df['opp_def_rating'] = df['OPP_TEAM'].map(opp_pts_allowed)

# Feature 6: Home/away (home=1, away=0)
# MATCHUP format: "TEAM vs. OPP" = home, "TEAM @ OPP" = away
df['home_away'] = df['MATCHUP'].str.contains('vs\.', regex=True).astype(float)

# Feature 7: Assist rate — rolling 5-game AST/MIN (playmaker proxy)
df['avg_ast'] = g['AST'].transform(lambda x: x.rolling(5).mean().shift(1))
df['ast_rate'] = np.where(df['recent_min'] > 0, df['avg_ast'] / df['recent_min'], 0.0)

# Feature 8: Defensive rate — rolling 5-game (STL+BLK)/MIN
df['avg_stl'] = g['STL'].transform(lambda x: x.rolling(5).mean().shift(1))
df['avg_blk'] = g['BLK'].transform(lambda x: x.rolling(5).mean().shift(1))
df['def_rate'] = np.where(df['recent_min'] > 0, (df['avg_stl'] + df['avg_blk']) / df['recent_min'], 0.0)

# Feature 9: Points per minute — scoring efficiency
df['pts_per_min'] = np.where(df['recent_min'] > 0, df['avg_pts'] / df['recent_min'], 0.0)

# Feature 10: Rest days — days since last game (B2B = 1, normal = 2, long rest = 3+)
df['prev_date'] = g['GAME_DATE'].transform(lambda x: x.shift(1))
df['rest_days'] = (df['GAME_DATE'] - df['prev_date']).dt.days.fillna(3).clip(1, 7)

# Feature 11: Recent vs season scoring (must match inference: recent_pts / season_pts)
df['recent_5g_pts'] = g['PTS'].transform(lambda x: x.rolling(5).mean().shift(1))
df['recent_vs_season'] = np.where(
    df['avg_pts'] > 0,
    df['recent_5g_pts'] / df['avg_pts'],
    1.0
).clip(0.5, 2.0)

# Feature 12: Games played this season (sample size / reliability proxy)
df['games_played'] = g.cumcount()  # 0-indexed, represents games BEFORE this one

# Feature 13: Rebounds per minute — critical for C/PF accuracy.
# LightGBM was blind to rebounding volume, causing systematic underestimation
# of interior bigs like Poeltl (9reb, ~2.5 projected, 5.4 actual).
# Must also be updated in project_player() inference vector (api/index.py).
df['reb_per_min'] = np.where(
    df['avg_min'] > 0,
    df['REB'] / df['avg_min'],
    0.0
).clip(0.0, 1.5)  # guard vs. center range: ~0.1 to ~0.4 typical

# Drop rows with NaN in any feature
features = [
    'avg_min', 'avg_pts', 'usage_trend', 'opp_def_rating',
    'home_away', 'ast_rate', 'def_rate', 'pts_per_min',
    'rest_days', 'recent_vs_season', 'games_played', 'reb_per_min'
]
df = df.dropna(subset=features + [target])
print(f"After feature engineering: {len(df)} samples with complete features.")

if len(df) < 50:
    print(f"[ERROR] Only {len(df)} samples after merge — aborting.")
    import sys; sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────────
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"2. Training LightGBM on {len(X_train)} samples ({len(SEASONS)} seasons)...")
print(f"   Features ({len(features)}): {features}")
model = lgb.LGBMRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=40,
    random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

# Save feature list alongside model so inference can verify alignment
model_bundle = {"model": model, "features": features}

print("3. Saving AI model to lgbm_model.pkl...")
with open("lgbm_model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)
print(f"Done! Model trained on {len(X_train)} samples from seasons: {', '.join(SEASONS)}")
print(f"Feature importances:")
for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat}: {imp}")
