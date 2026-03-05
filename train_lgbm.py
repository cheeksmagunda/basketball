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

# Target Variable: Real Score-aligned formula (boosted defensive stats)
# Must match _dfs_score() in api/index.py
df['actual_base_score'] = (
    df['PTS'] + df['REB'] + (df['AST'] * 1.5) +
    (df['STL'] * 4.5) + (df['BLK'] * 4.0) - (df['TOV'] * 1.2)
)

# Feature 1 & 2: Season Averages BEFORE the game happened (shift(1))
# Group by (PLAYER_ID, SEASON) so expanding averages reset at season boundaries
df['avg_min'] = df.groupby(['PLAYER_ID', 'SEASON'])['MIN'].transform(lambda x: x.expanding().mean().shift(1))
df['avg_pts'] = df.groupby(['PLAYER_ID', 'SEASON'])['PTS'].transform(lambda x: x.expanding().mean().shift(1))

# Feature 3: Recent Form / Usage Spike (Rolling 5 games before tonight)
df['recent_min'] = df.groupby(['PLAYER_ID', 'SEASON'])['MIN'].transform(lambda x: x.rolling(5).mean().shift(1))

# Drop rows without enough history (first 5 games per player per season)
df = df.dropna(subset=['avg_min', 'avg_pts', 'recent_min'])

# Calculate the exact usage_trend multiplier your app uses
df['usage_trend'] = np.where(df['avg_min'] > 0, df['recent_min'] / df['avg_min'], 1.0)
df['usage_trend'] = df['usage_trend'].clip(0.90, 1.50) # Cap it

# Feature 4: Opponent Defense Rating
df['OPP_TEAM'] = df['MATCHUP'].str[-3:].str.strip()
opp_pts_allowed = df.groupby('OPP_TEAM')['PTS'].mean().to_dict()
df['opp_def_rating'] = df['OPP_TEAM'].map(opp_pts_allowed)

# ---------------------------------------------------------
# TRAIN THE MODEL
# ---------------------------------------------------------
features = ['avg_min', 'avg_pts', 'usage_trend', 'opp_def_rating']
target = 'actual_base_score'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"2. Training LightGBM on {len(X_train)} samples ({len(SEASONS)} seasons)...")
model = lgb.LGBMRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=40,
    random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

print("3. Saving AI model to lgbm_model.pkl...")
with open("lgbm_model.pkl", "wb") as f:
    pickle.dump(model, f)
print(f"Done! Model trained on {len(X_train)} samples from seasons: {', '.join(SEASONS)}")
