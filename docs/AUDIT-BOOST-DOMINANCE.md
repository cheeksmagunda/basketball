# Audit: Boost Dominance — Strip to Winning Formula

**Document Status:** Historical Snapshot

**Date:** 2026-03-19
**Scope:** Full pipeline audit against leaderboard-proven winning strategy
**Core thesis:** Boost leverage dominates everything. Any code that dilutes, overrides, or contradicts this principle is flagged and addressed.

---

## PHASE 1: HARD FILTER AUDIT

### 1.1 — Boost Floor

| Check | Status | Detail |
|-------|--------|--------|
| Hard minimum boost cutoff exists | FIXED | Was 1.2x chalk / 1.5x moonshot. **Now 1.0x for both** — configurable via `projection.chalk_min_boost_floor` and `moonshot.min_card_boost` |
| Players below cutoff completely excluded | YES | `continue` statement removes them from candidate pool |
| Star anchor pathway REMOVED | FIXED | **Was:** Players with 20+ PPG season avg bypassed boost floor entirely (chalk: 0.6x min, moonshot: no min). Allowed Jokic (+0.2x), Bam (+0.6x), KD (+0.5x) into lineups despite terrible boost leverage. **Now:** Star anchor pathway fully removed from both chalk and moonshot pool building. All players must meet the boost floor, period. |
| Boost cap REMOVED | FIXED | `scoring_thresholds.chalk_boost_cap` was 1.5x, artificially capping high-boost players in chalk. **Now 3.0x** (effectively uncapped) — let the math work. |

**Code removed:**
- `star_anchor_ppg` config reads and `is_star_anchor` logic in chalk pool (was ~lines 3032-3046)
- `is_star_anchor` logic in moonshot pool (was ~lines 3149-3155)
- `chalk_max_stars` and `chalk_star_boost_threshold` constraints
- MILP `max_low_boost` constraint for moonshot

### 1.2 — Active-Only Filter

| Check | Status | Detail |
|-------|--------|--------|
| OUT players hard-excluded | YES | `project_player()` returns None for `is_out` |
| DTD/DOUBT excluded | YES | Returns None for DTD, DOUBT injury status |
| GTD gets minute reduction | YES | 0.75x penalty (configurable) |
| RotoWire integration | YES | 30-min cache, checked before pool building |
| No "Out player speculation" | YES | No hedge logic found |

### 1.3 — Minimum Real Score Floor

| Check | Status | Detail |
|-------|--------|--------|
| Chalk rating floor | YES | 3.5 (`min_chalk_rating`) |
| Moonshot rating floor | YES | 3.0 (`min_rating_floor`) |
| Based on recent form | PARTIAL | Uses blended season/recent (now 80/20 recent-weighted, was 50/50) |

---

## PHASE 2: SCORING FORMULA AUDIT

### 2.1 — Core Value Calculation

**Target formula:** `Player_Value = Expected_RS × (Slot_Multiplier + Boost)`

**Chalk EV (BEFORE):**
```
chalk_ev = rating × (avg_slot + card_boost) × reliability × chalk_matchup × team_motivation_mult
```

**Chalk EV (AFTER):**
```
chalk_ev = rating × (avg_slot + card_boost)
```

**Removed from chalk_ev:**
- `reliability` multiplier (minute consistency penalty, 0.70-1.0 range) — bake stability into RS projection instead of post-hoc adjustments
- `team_motivation_mult` (0.88-1.12 range) — disabled via config
- `chalk_matchup` kept at narrow [0.92, 1.10] — this is the "light matchup context" the audit approves (pace/defensive quality). Claude matchup intelligence disabled.

**Moonshot EV (BEFORE):**
```
base_rating = rating × max(0.85, 1.0 - variance × 0.15)
combined_factor = matchup × claude_factor  (range [0.75, 1.30])
boost_leverage = est_mult^1.2
moonshot_ev = base_rating × combined_factor × boost_leverage × (avg_slot + est_mult)
```

**Moonshot EV (AFTER):**
```
boost_leverage = est_mult^1.2
moonshot_ev = rating × boost_leverage × (avg_slot + est_mult)
```

**Removed from moonshot_ev:**
- Variance penalty (`max(0.85, 1.0 - variance × 0.15)`) — moonshot IS variance. High-boost role players are volatile; that's the point.
- Claude matchup factor — disabled via config
- Math matchup factor kept at narrow range

### 2.2 — Lineup Optimization

| Check | Status | Detail |
|-------|--------|--------|
| Combinatorial (not greedy) | YES | MILP via PuLP/CBC — optimal |
| Core pool → MILP | YES | 8-player core, two 5-of-8 configs |
| No greedy sequential picking | YES | MILP solves globally |

### 2.3 — Slot Assignment

| Check | Status | Detail |
|-------|--------|--------|
| Highest RS → highest slot | YES | MILP optimizes simultaneously |
| Two-phase moonshot | YES | Phase 1 selects with shaped ratings, Phase 2 assigns slots with raw RS |
| No manual slot overrides | YES | MILP handles optimally |

---

## PHASE 3: OVERTHINKING MODULES — REMOVED/DISABLED

### 3.1 — Ceiling vs Floor Projections

| Item | Action | Detail |
|------|--------|--------|
| Variance penalty in chalk MILP | REMAINS | Still downweights volatile players for Starting 5 reliability — appropriate for chalk |
| Variance penalty in moonshot pre-MILP | REMOVED | Was `base_rating × max(0.85, 1.0 - variance × 0.15)`. Now uses raw rating. |
| Variance uplift in moonshot MILP | REMAINS | MILP upweights variance for moonshot — appropriate |
| Game stacking bonus (ceiling_score × 1.08) | REMAINS | Used only for ceiling_score field, not for pool selection or MILP |

### 3.2 — Stack/Correlation Logic

| Item | Status | Detail |
|------|--------|--------|
| No stacking bonuses | CLEAN | No teammate correlation logic found |
| max_per_team = 3 | KEPT | Prevents overconcentration, not a stacking bonus |

### 3.3 — Matchup Adjustments

| Item | Action | Detail |
|------|--------|--------|
| Math-based matchup [0.92, 1.10] chalk | KEPT | Light, data-driven (opponent pts_allowed vs league avg) |
| Claude DvP web intelligence | DISABLED | `matchup.claude_enabled: false` — subjective, adds noise |
| Claude context pass (Layer 2) | DISABLED | `context_layer.enabled: false` — ±40% RS override based on subjective reasoning |
| Claude lineup review (Layer 3) | DISABLED | `lineup_review.enabled: false` — can override MILP math post-optimization |
| Moonshot matchup [0.75, 1.30] | TIGHTENED | Now [0.90, 1.15] — matchup should not swamp boost signal |

### 3.4 — Contrarian Logic

| Item | Status | Detail |
|------|--------|--------|
| Boost IS the contrarian signal | CORRECT | High boost = low ownership = contrarian by definition |
| No explicit "be different" logic | CLEAN | No ownership-based tiebreakers found outside boost |

### 3.5 — Recency Bias Dampening

| Item | Action | Detail |
|------|--------|--------|
| season_recent_blend | FIXED | 0.5 → 0.2 (now 80% recent, 20% season) |
| Major role change | KEPT | 80% recent when >25% drop — already correct |
| Moderate decline | ADJUSTED | 65% → 80% recent |

### 3.6 — Position/Role Balancing

| Item | Action | Detail |
|------|--------|--------|
| No position constraints in MILP | CLEAN | Confirmed: "Real Sports has no position requirements" |
| Center cap in moonshot | REMOVED | Was max 3 centers. Now 5 (effectively unlimited). Poeltl, Queta, Achiuwa all appear in winning lineups. |

---

## PHASE 4: DATA INPUTS — VERIFIED

| Input | Status | Detail |
|-------|--------|--------|
| Boost values | 4-layer cascade | Daily ingestion → config overrides → ownership data → sigmoid fallback |
| Active/Out status | RotoWire 30-min cache | Checked at pool building time |
| Expected RS | LightGBM + heuristic (35/65 blend) | Compressed RS with asymmetric ceiling |
| Slot multipliers | Hardcoded [2.0, 1.8, 1.6, 1.4, 1.2] | Configurable via `lineup.slot_multipliers` |
| Vegas O/U | Game script adjustments | Light per-stat multipliers based on total/spread |

---

## PHASE 5: SIMPLIFIED FLOW

The pipeline is now:

```
1. Pull tonight's player pool (ESPN rosters + stats)
2. Injury cascade (redistribute minutes from OUT players)
3. Project each player (LightGBM + heuristic RS, 80% recent form)
4. Card boost (4-layer: daily ingestion → config → ownership → sigmoid)
5. HARD FILTER: Active only, boost >= 1.0x, rating >= 3.0 (moonshot) / 3.5 (chalk)
6. Light matchup adjustment [0.92, 1.10] from opponent defense quality
7. Score: chalk_ev = rating × (avg_slot + boost)
         moonshot_ev = rating × boost_leverage × (avg_slot + boost)
8. Core pool: top 8 by max(chalk_ev, moonshot_ev)
9. MILP: Starting 5 (reliability) + Moonshot (ceiling) from core pool
10. Return top lineups
```

**Removed from flow:**
- Claude context pass (Layer 2) — was adjusting RS ±40%
- Claude lineup review (Layer 3) — was swapping players post-MILP
- Claude matchup intelligence — was injecting subjective DvP analysis
- Team motivation multiplier — was ±12% based on standings
- Star anchor pathway — was letting low-boost stars bypass filters
- Reliability multiplier — was penalizing minute-inconsistent players post-RS
- Variance penalty on moonshot pre-MILP — was dampening the exact volatility moonshot wants

---

## Summary: Changes Made

| Component | Before | After | Rationale |
|-----------|--------|-------|-----------|
| Star anchor pathway | Enabled (20+ PPG bypass) | **Removed** | Low-boost stars should never bypass boost floor |
| Chalk boost floor | 1.2x | **1.0x** | Include all non-zero boost players |
| Moonshot boost floor | 1.5x | **1.0x** | Include all non-zero boost players |
| Chalk boost cap | 1.5x | **3.0x** (uncapped) | Don't artificially limit high-boost value |
| Claude context pass | Enabled (±40% RS) | **Disabled** | Subjective override of math |
| Claude lineup review | Enabled (post-MILP swaps) | **Disabled** | Override of optimal MILP solution |
| Claude matchup intel | Enabled | **Disabled** | Noise; boost swamps matchup edge |
| Team motivation | Enabled (±12%) | **Disabled** | Marginal signal, adds complexity |
| Variance penalty (moonshot) | 0.15 | **0** | Moonshot IS variance |
| Center cap (moonshot) | 3 | **5** (unlimited) | No position balancing needed |
| Recent form weight | 50/50 | **80/20** | Recent form is primary signal |
| Reliability in chalk_ev | 0.70-1.0 multiplier | **Removed** | Bake into RS, not post-hoc |
| Moonshot matchup range | [0.75, 1.30] | **[0.90, 1.15]** | Tighter; boost dominates |
