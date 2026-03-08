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
# TARGET: Real Score-aligned formula (boosted defensive stats)
# Must match _dfs_score() in api/index.py
# ─────────────────────────────────────────────────────────────────────────────
df['actual_base_score'] = (
    df['PTS'] + df['REB'] + (df['AST'] * 1.5) +
    (df['STL'] * 4.5) + (df['BLK'] * 4.0) - (df['TOV'] * 1.2)
)

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

# Feature 11: Recent 3-game scoring trend vs 5-game baseline
df['recent_3g_pts'] = g['PTS'].transform(lambda x: x.rolling(3).mean().shift(1))
df['recent_5g_pts'] = g['PTS'].transform(lambda x: x.rolling(5).mean().shift(1))
df['recent_3g_trend'] = np.where(
    df['recent_5g_pts'] > 0,
    df['recent_3g_pts'] / df['recent_5g_pts'],
    1.0
).clip(0.5, 2.0)

# Feature 12: Games played this season (sample size / reliability proxy)
df['games_played'] = g.cumcount()  # 0-indexed, represents games BEFORE this one

# Drop rows with NaN in any feature
features = [
    'avg_min', 'avg_pts', 'usage_trend', 'opp_def_rating',
    'home_away', 'ast_rate', 'def_rate', 'pts_per_min',
    'rest_days', 'recent_3g_trend', 'games_played'
]
target = 'actual_base_score'

df = df.dropna(subset=features + [target])
print(f"After feature engineering: {len(df)} samples with complete features.")

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
