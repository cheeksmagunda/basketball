# Global Variables & Design Tokens Audit

**Document Status:** Historical Snapshot

**Date:** 2026-03-08

## Frontend (index.html)

### Design tokens (:root)
- **Single source of truth:** `--radius-card: 14px`; `--r` is an alias (`var(--radius-card)`) so card radius is defined once.
- **Semantic vs legacy:** Prefer `--color-success`, `--color-danger`, `--radius-card`, `--font-size-micro`, `--tracking-caps` for new code. `--green` / `--red` kept for backward compatibility (slightly different hex from semantic tokens).
- **Token groups:** Background/surface, border, tab accents (chalk/upside/line/lab), log palette, semantic (success/danger/warning/text), radius/typography, decorative one-offs (injury, oracle, date chip).

### Global state (JavaScript)
- **Documented in-file:** Comment block before `SLATE` lists all app-level state: `SLATE`, `PICKS_DATA`, `isFetching`, `SLATE_LOADED_AT`, `LOG`, `LINE_*`, `LAB`, `PREDICT_SUB`, `TAB_ACCENT`.
- **LAB:** `initDate` added to initializer so the “date when Lab was opened” property is declared (was previously set only at runtime).
- **Constants:** `TEAM_COLORS`, `TAB_ACCENT`, `ORACLE_MSGS`, `_SLOT_HEAT`, `LINE_STAT_LABEL` are read-only. Internal/private use: `_oracleMsgIdx`, `_oracleMsgTimer`, `_predSavedDate` (underscore prefix).

## Backend (api/index.py)

### Module-level constants
- **Convention:** `UPPER_SNAKE` = public; `_lower` = private (do not mutate). Documented in the CONSTANTS & CACHE UTILITIES section.
- **Public:** `ESPN`, `ODDS_API_BASE`, `MIN_GATE`, `DEFAULT_TOTAL`, `CSV_HEADER`, `CACHE_DIR`, `LOCK_DIR`, `CONFIG_CACHE_DIR`, `CRON_SECRET`, `PRED_FIELDS`, `ACT_FIELDS`, `LINE_CSV_HEADER`, `LINE_FIELDS`, `POS_GROUPS`, `_LINE_PICK_CONTRACT_FIELDS` (used for API contract), `_RATE_LIMITS`.
- **Private:** `_CONFIG_DEFAULTS`, `_ABBR_TO_NAME_FRAG`, `_STAT_MARKET`, `_BEN_TOOLS`, `_BEN_WRITABLE_PATHS`, `_PLAYER_INTERNAL_FIELDS`, `_RATE_LIMIT_STORE`, `_RATE_LIMIT_LOCK`, `_REQUIRED_ENV`, etc. All are read-only except rate-limit store (protected by lock).

## Fixes applied
1. **:root** — Comment block added; `--r` set to `var(--radius-card)` so card radius has one definition.
2. **Frontend globals** — Comment block listing global state; `LAB.initDate` added to `LAB` initializer.
3. **Backend** — Comment in constants section: “Module-level: UPPER_SNAKE = public; _lower = private (do not mutate).”
