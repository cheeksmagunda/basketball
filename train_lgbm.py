import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from nba_api.stats.endpoints import playergamelogs
from sklearn.model_selection import train_test_split

print("1. Fetching real NBA historical data from nba_api (this takes ~15 seconds)...")
# Pulling the entire 2023-24 NBA season game logs
logs = playergamelogs.PlayerGameLogs(season_nullable='2023-24')
df = logs.get_data_frames()[0]

print(f"Fetched {len(df)} game logs. Engineering features...")

# Sort by player and date to calculate accurate "past" stats
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'].astype(str).str[:10])
df = df.sort_values(by=['PLAYER_ID', 'GAME_DATE'])

# Target Variable: This is the exact DFS formula your app uses
df['actual_base_score'] = (
    df['PTS'] + df['REB'] + (df['AST'] * 1.5) + 
    (df['STL'] * 3.5) + (df['BLK'] * 3.0) - (df['TOV'] * 1.2)
)

# Feature 1 & 2: Season Averages BEFORE the game happened (shift(1))
df['avg_min'] = df.groupby('PLAYER_ID')['MIN'].transform(lambda x: x.expanding().mean().shift(1))
df['avg_pts'] = df.groupby('PLAYER_ID')['PTS'].transform(lambda x: x.expanding().mean().shift(1))

# Feature 3: Recent Form / Usage Spike (Rolling 5 games before tonight)
df['recent_min'] = df.groupby('PLAYER_ID')['MIN'].transform(lambda x: x.rolling(5).mean().shift(1))

# Drop the first 5 games of the season for each player (since they have no "recent form" yet)
df = df.dropna(subset=['avg_min', 'avg_pts', 'recent_min'])

# Calculate the exact usage_trend multiplier your app uses
df['usage_trend'] = np.where(df['avg_min'] > 0, df['recent_min'] / df['avg_min'], 1.0)
df['usage_trend'] = df['usage_trend'].clip(0.90, 1.50) # Cap it

# Feature 4: Opponent Defense Rating
# Extract the opponent's abbreviation from the Matchup string (e.g., 'BOS @ MIL' -> 'MIL')
df['OPP_TEAM'] = df['MATCHUP'].str[-3:].str.strip()
# Calculate how many points each team gives up on average to simulate Def Rating
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

print(f"2. Training LightGBM on {len(X_train)} historical samples...")
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
print("Done! You have successfully built a machine learning model.")
