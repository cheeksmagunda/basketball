"""
Train dual-head LightGBM bundle for Real Score (RS) with ranking-focused sample weights.

Head A (baseline): predicts core RS level for all players.
Head B (spike): predicts positive residual above baseline (role-player eruption signal).

Training objectives:
- Sample weights up-weight high actual_rs within each game date (top-10 RS days matter more).
- Evaluation: top-5 RS recall, NDCG@5 RS, MAE (reported on held-out dates).

Feature list must stay aligned with api/index.py::_lgbm_feature_vector().
"""
import os
import re
import glob
import json
import time
import math
import numpy as np
import pandas as pd
import lightgbm as lgb
from nba_api.stats.endpoints import playergamelogs
from requests.exceptions import RequestException
from sklearn.model_selection import train_test_split

# Pull 3 seasons of NBA game logs
SEASONS = ["2023-24", "2024-25", "2025-26"]

# Total feature count — must match inference (see api/index.py)
# v63: removed 5 dead features (cascade_signal, usage_share, teammate_out_count, spread_abs, game_total)
# 17 features (12 original + 5 new: l3_vs_l5_pts, min_volatility, starter_proxy, opp_pts_allowed, team_pace_proxy)
N_FEATURES = 17


def _fetch_season_logs_with_retry(season: str, max_attempts: int = 6) -> pd.DataFrame:
    """Fetch one season of logs with retry/backoff for transient NBA API timeouts."""
    for attempt in range(1, max_attempts + 1):
        try:
            logs = playergamelogs.PlayerGameLogs(season_nullable=season, timeout=60)
            df_s = logs.get_data_frames()[0]
            if df_s is None or df_s.empty:
                raise RuntimeError(f"empty response for season {season}")
            return df_s
        except (
            RequestException,
            TimeoutError,
            RuntimeError,
            json.JSONDecodeError,
            ValueError,
        ) as e:
            if attempt == max_attempts:
                raise
            backoff = min(45, 5 * attempt)
            print(
                f"   [WARN] {season} fetch attempt {attempt}/{max_attempts} failed: {e}; retrying in {backoff}s..."
            )
            time.sleep(backoff)


def _normalize_name(name):
    n = str(name).lower().strip()
    n = re.sub(r"['\.\-]", "", n)
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", n)
    return re.sub(r"\s+", " ", n).strip()


def _ndcg_at_k(pred_order: list, relevance: dict, k: int = 5) -> float:
    dcg = 0.0
    for i, pid in enumerate(pred_order[:k]):
        rel = relevance.get(pid, 0.0)
        dcg += rel / math.log2(i + 2)
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return (dcg / idcg) if idcg > 0 else 0.0


def _top5_recall(pred_top5: set, actual_top5: set) -> float:
    return len(pred_top5 & actual_top5) / 5.0


print(f"1. Fetching NBA game logs for {len(SEASONS)} seasons...")
frames = []
for season in SEASONS:
    print(f"   Fetching {season}...")
    df_s = _fetch_season_logs_with_retry(season)
    df_s["SEASON"] = season
    frames.append(df_s)
    print(f"   Got {len(df_s)} game logs for {season}")
    if season != SEASONS[-1]:
        time.sleep(3)

df = pd.concat(frames, ignore_index=True)
print(f"Total: {len(df)} game logs across {len(SEASONS)} seasons. Engineering features...")

# playergamelogs (nba_api) exposes MATCHUP + TEAM_ABBREVIATION, not OPP_TEAM
if "OPP_TEAM" not in df.columns and "MATCHUP" in df.columns and "TEAM_ABBREVIATION" in df.columns:
    def _parse_opp(mu: str, team_abbr: str) -> str:
        mu = str(mu)
        tb = str(team_abbr).strip()
        if " vs. " in mu:
            t1, t2 = [x.strip() for x in mu.split(" vs. ", 1)]
            return t2 if t1 == tb else t1
        if " @ " in mu:
            away, home = [x.strip() for x in mu.split(" @ ", 1)]
            return home if tb == away else away
        return tb

    df["OPP_TEAM"] = [_parse_opp(m, t) for m, t in zip(df["MATCHUP"], df["TEAM_ABBREVIATION"])]

df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"].astype(str).str[:10])
df = df.sort_values(by=["PLAYER_ID", "SEASON", "GAME_DATE"])
df["norm_name"] = df["PLAYER_NAME"].apply(_normalize_name)
df["date_key"] = df["GAME_DATE"].dt.normalize()

# ── Load actuals for labeling (merge AFTER feature engineering on full logs) ─
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

actuals_df = None
if actuals_frames:
    actuals_df = pd.concat(actuals_frames, ignore_index=True).dropna(subset=["actual_rs"])
    actuals_df["date_key"] = actuals_df["date_key"].dt.normalize()
    print(f"   Loaded {len(actuals_df)} actuals rows from {len(actuals_frames)} files.")

g = df.groupby(["PLAYER_ID", "SEASON"])

df["avg_min"] = g["MIN"].transform(lambda x: x.expanding().mean().shift(1))
df["avg_pts"] = g["PTS"].transform(lambda x: x.expanding().mean().shift(1))
df["recent_min"] = g["MIN"].transform(lambda x: x.rolling(5).mean().shift(1))

USAGE_TREND_MIN, USAGE_TREND_MAX = 0.90, 1.50
df = df.dropna(subset=["avg_min", "avg_pts", "recent_min"]).copy()
g = df.groupby(["PLAYER_ID", "SEASON"])
df["usage_trend"] = np.where(df["avg_min"] > 0, df["recent_min"] / df["avg_min"], 1.0)
df["usage_trend"] = df["usage_trend"].clip(USAGE_TREND_MIN, USAGE_TREND_MAX)

opp_pts_allowed_map = df.groupby("OPP_TEAM")["PTS"].mean().to_dict()
df["opp_def_rating"] = df["OPP_TEAM"].map(opp_pts_allowed_map)
df["home_away"] = df["MATCHUP"].str.contains(r"vs\.", regex=True).astype(float)

df["avg_ast"] = g["AST"].transform(lambda x: x.rolling(5).mean().shift(1))
df["ast_rate"] = np.where(df["recent_min"] > 0, df["avg_ast"] / df["recent_min"], 0.0)

df["avg_stl"] = g["STL"].transform(lambda x: x.rolling(5).mean().shift(1))
df["avg_blk"] = g["BLK"].transform(lambda x: x.rolling(5).mean().shift(1))
df["def_rate"] = np.where(
    df["recent_min"] > 0, (df["avg_stl"] + df["avg_blk"]) / df["recent_min"], 0.0
)

df["pts_per_min"] = np.where(df["recent_min"] > 0, df["avg_pts"] / df["recent_min"], 0.0)

df["prev_date"] = g["GAME_DATE"].transform(lambda x: x.shift(1))
df["rest_days"] = (df["GAME_DATE"] - df["prev_date"]).dt.days.fillna(3).clip(1, 7)

df["recent_5g_pts"] = g["PTS"].transform(lambda x: x.rolling(5).mean().shift(1))
df["recent_vs_season"] = np.where(
    df["avg_pts"] > 0, df["recent_5g_pts"] / df["avg_pts"], 1.0
).clip(0.5, 2.0)

df["games_played"] = g.cumcount()

# v63: Fix data leakage — use season average REB, not same-game actual REB
df["avg_reb"] = g["REB"].transform(lambda x: x.expanding().mean().shift(1))
df["reb_per_min"] = np.where(
    df["avg_min"] > 0, df["avg_reb"] / df["avg_min"], 0.0
).clip(0.0, 1.5)

# Role-volatility features (known before game)
df["roll3_pts"] = g["PTS"].transform(lambda x: x.rolling(3).mean().shift(1))
df["l3_vs_l5_pts"] = np.where(
    df["recent_5g_pts"] > 0, df["roll3_pts"] / df["recent_5g_pts"], 1.0
).clip(0.4, 2.5)

df["min_roll5_std"] = g["MIN"].transform(lambda x: x.rolling(5).std().shift(1))
df["min_volatility"] = np.where(
    df["avg_min"] > 0, df["min_roll5_std"] / (df["avg_min"] + 0.5), 0.0
).clip(0.0, 1.2)

df["starter_proxy"] = (df["avg_min"] >= 26.0).astype(float)

# v63: Removed cascade_signal (dead feature — zero importance, trained on constant 0.0)

# v62: 6 new features from top-performer analysis
# 1. Opponent points allowed (defensive weakness signal)
df["opp_pts_allowed"] = df["OPP_TEAM"].map(opp_pts_allowed_map).fillna(110.0)

# 2. Team pace proxy (team's average total points per game — possessions indicator)
team_total = df.groupby(["TEAM_ABBREVIATION", "GAME_DATE"])["PTS"].sum().reset_index()
team_pace = team_total.groupby("TEAM_ABBREVIATION")["PTS"].mean().to_dict()
df["team_pace_proxy"] = df["TEAM_ABBREVIATION"].map(team_pace).fillna(110.0)

# v63: Removed usage_share (dead feature — zero importance) and teammate_out_count (dead feature)

# v63: Removed game_total and spread_abs (dead features — zero importance in model)

features = [
    "avg_min",
    "avg_pts",
    "usage_trend",
    "opp_def_rating",
    "home_away",
    "ast_rate",
    "def_rate",
    "pts_per_min",
    "rest_days",
    "recent_vs_season",
    "games_played",
    "reb_per_min",
    "l3_vs_l5_pts",
    "min_volatility",
    "starter_proxy",
    # v62: 2 viable new features (removed 4 dead: cascade_signal, usage_share, teammate_out, spread_abs)
    "opp_pts_allowed",
    "team_pace_proxy",
]

assert len(features) == N_FEATURES

if actuals_df is not None:
    df = df.merge(
        actuals_df[["norm_name", "date_key", "actual_rs"]],
        on=["norm_name", "date_key"],
        how="left",
    )
    n_labeled = int(df["actual_rs"].notna().sum())
    print(f"   Rows with actual_rs label: {n_labeled} (of {len(df)} after feature prep).")
    target = "actual_rs"
    df = df.dropna(subset=["actual_rs"])
else:
    print("   [WARN] No actuals found — falling back to formula target.")
    df["actual_base_score"] = (
        df["PTS"]
        + df["REB"]
        + (df["AST"] * 1.5)
        + (df["STL"] * 4.5)
        + (df["BLK"] * 4.0)
        - (df["TOV"] * 1.2)
    )
    target = "actual_base_score"

df = df.dropna(subset=features + [target])
print(f"After feature engineering: {len(df)} samples with complete features.")

if len(df) < 80:
    print(f"[ERROR] Only {len(df)} samples — need more for stable two-head training.")
    import sys

    sys.exit(1)

# ── Ranking-focused sample weights (per calendar date) ───────────────────────
date_col = df["GAME_DATE"].dt.normalize()


def _assign_weights(y_series: pd.Series) -> pd.Series:
    """Higher weight for top actual RS within the same day (approximate game slate)."""
    ranks = y_series.rank(pct=True, method="average")
    w = 1.0 + 4.0 * (ranks >= 0.90) + 2.0 * ((ranks >= 0.75) & (ranks < 0.90))
    return w


sample_weight = df.groupby(date_col, group_keys=False)[target].transform(_assign_weights)

# v62: Top-performers sample weighting — players on the actual leaderboard get 3x weight
top_perf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "top_performers.csv")
if os.path.exists(top_perf_path):
    try:
        tp_df = pd.read_csv(top_perf_path)
        tp_df["norm_name"] = tp_df["player_name"].apply(_normalize_name)
        tp_df["date_key"] = pd.to_datetime(tp_df["date"]).dt.normalize()
        tp_keys = set(zip(tp_df["norm_name"], tp_df["date_key"]))
        is_leaderboard = [
            (nn, dk) in tp_keys
            for nn, dk in zip(df["norm_name"], df["date_key"])
        ]
        lb_boost = pd.Series([3.0 if is_lb else 1.0 for is_lb in is_leaderboard], index=df.index)
        sample_weight = sample_weight * lb_boost
        n_lb = sum(is_leaderboard)
        print(f"   Top-performers boost: {n_lb} samples get 3x weight (from {len(tp_keys)} leaderboard entries)")
    except Exception as e:
        print(f"   [WARN] Could not load top_performers.csv for weighting: {e}")

X = df[features].copy()
y = df[target].astype(float)

(
    X_train,
    X_test,
    y_train,
    y_test,
    w_train,
    w_test,
    _date_train,
    _date_test,
    _idx_train,
    test_idx,
) = train_test_split(
    X,
    y,
    sample_weight,
    df["GAME_DATE"],
    df.index,
    test_size=0.2,
    random_state=42,
    shuffle=False,  # v63: Temporal split (not random) to avoid leakage
)

print(f"2. Training dual-head LightGBM on {len(X_train)} samples...")
print(f"   Features ({len(features)}): {features}")

base_params = dict(
    n_estimators=900,       # v62: 700→900 for 22 features
    learning_rate=0.035,    # v62: slower learning for more features
    max_depth=8,            # v62: 7→8 for additional feature interactions
    num_leaves=64,          # v62: 48→64 for increased capacity
    random_state=42,
    subsample=0.80,         # v62: slight reduction for regularization
    colsample_bytree=0.80,  # v62: same reason
)

model_baseline = lgb.LGBMRegressor(**base_params)
model_baseline.fit(
    X_train,
    y_train,
    sample_weight=w_train,
    eval_set=[(X_test, y_test)],
)

baseline_pred_full = model_baseline.predict(X)
spike_y = np.maximum(0.0, y.values - baseline_pred_full)

model_spike = lgb.LGBMRegressor(**base_params)
model_spike.fit(X, spike_y, sample_weight=sample_weight.values)

# ── Evaluation on test split (ranking KPIs) ────────────────────────────────
test_df = df.loc[test_idx].copy()
test_df["pred_rs"] = model_baseline.predict(X_test) + model_spike.predict(X_test)

recalls = []
ndcgs = []
for d, sub in test_df.groupby("GAME_DATE"):
    if len(sub) < 10:
        continue
    # top 5 by actual
    sub = sub.sort_values(target, ascending=False)
    actual_top5 = set(sub.head(5)["PLAYER_ID"])
    sub_pred = sub.sort_values("pred_rs", ascending=False)
    pred_top5 = set(sub_pred.head(5)["PLAYER_ID"])
    recalls.append(_top5_recall(pred_top5, actual_top5))
    rel = {row.PLAYER_ID: row[target] for _, row in sub.iterrows()}
    order = sub_pred["PLAYER_ID"].tolist()
    ndcgs.append(_ndcg_at_k(order, rel, k=5))

print(
    f"3. Test metrics — top5_recall_avg: {np.mean(recalls):.3f} | ndcg5_avg: {np.mean(ndcgs):.3f} | "
    f"MAE: {np.mean(np.abs(test_df[target] - test_df['pred_rs'])):.3f}"
)

BASELINE_TXT = "lgbm_baseline.txt"
SPIKE_TXT = "lgbm_spike.txt"

def _strip_lgbm_noise_params(path: str):
    """Remove training-only params that cause harmless but noisy warnings on load."""
    txt = Path(path).read_text()
    for param in ("early_stopping_min_delta", "bagging_by_query"):
        txt = re.sub(rf"^{param}=.*\n", "", txt, flags=re.MULTILINE)
        txt = re.sub(rf"^\[{param}:.*\]\n", "", txt, flags=re.MULTILINE)
    Path(path).write_text(txt)

print("4. Saving native LightGBM models + lgbm_model.json...")
model_baseline.booster_.save_model(BASELINE_TXT)
model_spike.booster_.save_model(SPIKE_TXT)
_strip_lgbm_noise_params(BASELINE_TXT)
_strip_lgbm_noise_params(SPIKE_TXT)
meta = {
    "format": "lightgbm_native",
    "bundle_version": 2,
    "features": features,
    "baseline_file": BASELINE_TXT,
    "spike_file": SPIKE_TXT,
}
with open("lgbm_model.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
    f.write("\n")

print("Done! Feature importances (baseline):")
for feat, imp in sorted(
    zip(features, model_baseline.feature_importances_), key=lambda x: -x[1]
):
    print(f"  {feat}: {imp}")
