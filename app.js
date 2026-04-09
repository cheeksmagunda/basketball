// ════════════════════════════════════════════════════════════════════
// FETCH TIMEOUT WRAPPER — Prevent indefinite hangs on slow backend
// ════════════════════════════════════════════════════════════════════
/**
 * fetchWithTimeout(url, options, timeoutMs, externalSignal)
 * Wraps fetch() with a timeout using Promise.race().
 * If timeout expires OR externalSignal fires, aborts the request and rejects.
 * @param {string} url - Fetch URL
 * @param {object} options - Standard fetch options
 * @param {number} timeoutMs - Timeout in milliseconds (default 10s for blocking, 30s for screenshot)
 * @param {AbortSignal} [externalSignal] - Optional external abort signal (e.g. tab switch)
 * @returns {Promise} - fetch response or timeout error
 */

// ── Per-tab AbortController system ──────────────────────────────────
// Aborts in-flight fetches when user switches tabs, preventing stale
// responses from updating global state and wasting bandwidth.
let _tabAbortControllers = {};

function _abortTab(tab) {
  if (_tabAbortControllers[tab]) {
    _tabAbortControllers[tab].abort();
    _tabAbortControllers[tab] = null;
  }
}

function _getTabSignal(tab) {
  if (!_tabAbortControllers[tab] || _tabAbortControllers[tab].signal.aborted) {
    _tabAbortControllers[tab] = new AbortController();
  }
  return _tabAbortControllers[tab].signal;
}

function fetchWithTimeout(url, options = {}, timeoutMs = 10000, externalSignal) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  // Compose timeout + external signal (tab switch abort)
  let signal = controller.signal;
  if (externalSignal && typeof AbortSignal.any === 'function') {
    signal = AbortSignal.any([controller.signal, externalSignal]);
  } else if (externalSignal) {
    // Fallback for older browsers: listen on external signal manually
    if (externalSignal.aborted) { controller.abort(); }
    else { externalSignal.addEventListener('abort', () => controller.abort(), { once: true }); }
  }

  options.signal = signal;

  return fetch(url, options)
    .then(response => {
      clearTimeout(timeoutId);
      return response;
    })
    .catch(error => {
      clearTimeout(timeoutId);
      if (error.name === 'AbortError') {
        // Distinguish tab-switch abort (silent) from timeout (user-facing)
        if (externalSignal && externalSignal.aborted) {
          throw new DOMException('Tab switched', 'AbortError');
        }
        throw new Error(`Request timed out — tap Retry`);
      }
      throw error;
    });
}

/** Safe localStorage parse: returns fallback on missing key or invalid JSON (avoids throw on corrupted data). */
function _safeParseLocalStorage(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    if (raw == null) return fallback;
    return JSON.parse(raw);
  } catch (e) {
    return fallback;
  }
}

/** Safe getElementById for resilience; use before accessing .style, .innerHTML, etc. */
function _el(id) { return document.getElementById(id); }

/** Hex color (#rrggbb) to rgba string. */
function _hexToRgba(hex, a) {
  const h = hex.replace('#', '');
  return 'rgba(' + parseInt(h.slice(0,2),16) + ',' + parseInt(h.slice(2,4),16) + ',' + parseInt(h.slice(4,6),16) + ',' + a + ')';
}

/** Fetch JSON with timeout — wraps fetchWithTimeout + .ok check. */
function _fetchJson(url, timeout) {
  return fetchWithTimeout(url, {}, timeout || 10000).then(function(r) {
    return r.ok ? r.json() : Promise.reject(new Error('HTTP ' + r.status));
  });
}

/** Format date as "Mon, Mar 22" style (short weekday + month + day). */
function _fmtDateShort(d, upper) {
  const s = d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
  return upper ? s.toUpperCase() : s;
}

// ════════════════════════════════════════════════════════════════════
// CONSTANTS — grep: TEAM_COLORS, team colors, accent bar, tcolor
// ════════════════════════════════════════════════════════════════════
// grep: TEAM_COLORS — ESPN abbreviations used by the backend (WSH not WAS, UTAH not UTA)
const TEAM_COLORS = {
  ATL:"#E03A3E",  // Hawks red
  BOS:"#00A550",  // Celtics green
  BKN:"#B8B8B8",  // Nets silver
  CHA:"#00B4CC",  // Hornets teal
  CHI:"#CE1141",  // Bulls crimson
  CLE:"#FDBB30",  // Cavs gold
  DAL:"#00538C",  // Mavs royal blue
  DEN:"#FDB927",  // Nuggets gold
  DET:"#0C6DC4",  // Pistons motor-city blue (distinct from CHI/LAC red)
  GS:"#FFC72C",   // Warriors gold (ESPN alt abbr)
  GSW:"#FFC72C",  // Warriors gold
  HOU:"#FF3A5C",  // Rockets bright red (distinct from CHI crimson)
  IND:"#FFC843",  // Pacers yellow-gold (lighter than CLE/DEN)
  LAC:"#C8102E",  // Clippers red
  LAL:"#FDB927",  // Lakers gold
  MEM:"#6B8FC0",  // Grizzlies slate blue
  MIA:"#98002E",  // Heat red
  MIL:"#3ECF7E",  // Bucks bright green
  MIN:"#78BE20",  // Wolves lime green
  NO:"#C8A96A",   // Pelicans gold (ESPN alt abbr)
  NOP:"#C8A96A",  // Pelicans gold
  NYK:"#F58426",  // Knicks orange
  OKC:"#EF8D2F",  // Thunder warm orange
  ORL:"#0077C0",  // Magic blue
  PHI:"#006BB6",  // 76ers blue
  PHX:"#E56020",  // Suns burnt orange
  POR:"#CB2025",  // Blazers deep red (distinct from ATL bright red)
  SA:"#C4CED4",   // Spurs silver (ESPN alt abbr)
  SAC:"#7B4DB5",  // Kings purple
  SAS:"#C4CED4",  // Spurs silver
  TOR:"#A8002C",  // Raptors dark wine red (distinct from CHI)
  UTA:"#F9A01B",  // Jazz gold (fallback key)
  UTAH:"#F9A01B", // Jazz gold — ESPN returns UTAH not UTA
  WAS:"#E31837",  // Wizards red (fallback key)
  WSH:"#E31837",  // Wizards red — ESPN returns WSH not WAS
};

// ════════════════════════════════════════════════════════════════════
// GLOBAL STATE — grep: SLATE, PICKS_DATA, LAB, LINE_LOADED
// ════════════════════════════════════════════════════════════════════
/**
 * API shapes and tab init (grep: GLOBAL STATE)
 * - SLATE: /api/slate response — { date, locked, games[], lineups: { chalk[], upside[] }, error? }.
 * - PICKS_DATA: /api/picks per gameId — { lineups: { the_lineup[] }, game, gameScript, locked, injuries }.
 * - LINE_LOTD_STATE: AsyncState for Line tab — data = { pick, over_pick?, under_pick?, error? }; pick has stat_type, line, narrative, game_time, etc.
 * - LAB: Ben tab — messages[], briefing, cachedStatus, system; init on first visit; lock poll when locked.
 * Tab init: Predict (loadSlate at startup, non-blocking; game dropdown from SLATE.games on success); Line/Lab lazy-init on first switchTab (initLinePage, initLabPage).
 */
let SLATE = null;
let PICKS_DATA = null;
let isFetching = false;
let SLATE_LOADED_AT = 0;
let _slateAutoRefreshCount = 0; // Cap for the delay<=0 retry loop (max 3)
let _slateAutoRefreshTimer = null; // Prevent duplicate auto-refresh timers
let _slateNextDayPoll = null;     // Poll for next-day slate after all_complete
let _slateLoadInFlight = false;   // Prevent concurrent loadSlate() calls
let _slateFailRetryCount = 0;    // Auto-retry on slate_failed (max 3)
let _slateFailRetryTimer = null; // Timer for auto-retry countdown


// ════════════════════════════════════════════════════════════════════
// ASYNC STATE — grep: asyncStateInitial, asyncStateLoading, asyncStateSuccess, asyncStateError
// Generic state wrapper for network-backed views: Initial -> Loading -> Success(Data) | Error
// Shape: { status: 'initial'|'loading'|'success'|'error', data: T|null, error: Error|null [, loadedAt?: number ] }
// ════════════════════════════════════════════════════════════════════
function asyncStateInitial() {
  return { status: 'initial', data: null, error: null };
}
function asyncStateLoading(prev) {
  const s = prev ? { ...prev } : asyncStateInitial();
  s.status = 'loading';
  s.error = null;
  return s;
}
function asyncStateSuccess(prev, data) {
  const s = prev ? { ...prev } : asyncStateInitial();
  s.status = 'success';
  s.data = data;
  s.error = null;
  s.loadedAt = Date.now();
  return s;
}
function asyncStateError(prev, err) {
  const s = prev ? { ...prev } : asyncStateInitial();
  s.status = 'error';
  s.error = err;
  return s;
}

let SLATE_STATE = asyncStateInitial();  // Predict slate AsyncState
let PICKS_STATE = asyncStateInitial();  // Per-game picks AsyncState

// ════════════════════════════════════════════════════════════════════
// ZERO-TRUST GATEWAYS — enforce guaranteed shapes before UI sees data
// ════════════════════════════════════════════════════════════════════

/** Enforces the exact structure required by the Predict Tab. */
function TransformSlateData(raw) {
  if (!raw || typeof raw !== 'object') raw = {};
  return {
    date:         typeof raw.date === 'string' ? raw.date : _etToday(),
    locked:       Boolean(raw.locked),
    all_complete: Boolean(raw.all_complete),
    lock_time:    typeof raw.lock_time === 'string' ? raw.lock_time : null,
    error:        typeof raw.error === 'string' ? raw.error : null,
    games:        Array.isArray(raw.games) ? raw.games : [],
    lineups: {
      chalk:  Array.isArray(raw.lineups?.chalk)  ? raw.lineups.chalk  : [],
      upside: Array.isArray(raw.lineups?.upside) ? raw.lineups.upside : []
    }
  };
}

/** Enforces the exact structure required by the Per-Game Analysis. */
function TransformPicksData(raw) {
  if (!raw || typeof raw !== 'object') raw = {};
  return {
    locked:     Boolean(raw.locked),
    gameScript: typeof raw.gameScript === 'string' ? raw.gameScript : '',
    strategy: {
      label:        typeof raw.strategy?.label       === 'string' ? raw.strategy.label       : '',
      description:  typeof raw.strategy?.description === 'string' ? raw.strategy.description : '',
      type:         typeof raw.strategy?.type        === 'string' ? raw.strategy.type        : 'standard',
      total_mult:   typeof raw.strategy?.total_mult  === 'number' ? raw.strategy.total_mult  : 0,
      favored_team: typeof raw.strategy?.favored_team=== 'string' ? raw.strategy.favored_team: ''
    },
    lineups: {
      the_lineup: Array.isArray(raw.lineups?.the_lineup) ? raw.lineups.the_lineup : []
    },
    game: {
      total:  typeof raw.game?.total  === 'number' ? raw.game.total  : null,
      spread: typeof raw.game?.spread === 'number' ? raw.game.spread : null
    }
  };
}

// ════════════════════════════════════════════════════════════════════
// TAB NAVIGATION — grep: switchTab, movePill, setPillAccent, segmented control
// ════════════════════════════════════════════════════════════════════
function switchTab(tab) {
  hideLoader(); // Dismiss 8-ball overlay on ANY tab switch (scoped to Predict but belt-and-suspenders)

  // Abort in-flight fetches for tabs we're leaving (prevents stale response processing)
  ['predictions', 'lab'].forEach(function(t) {
    if (t !== tab) _abortTab(t);
  });

  document.querySelectorAll('.bnav-icon-btn').forEach(b => {
    const isActive = b.dataset.tab === tab;
    b.classList.toggle('active', isActive);
    if (isActive) {
      const accent = TAB_ACCENT[tab] || TAB_ACCENT.predictions;
      b.style.color = `rgb(${accent.rgb})`;
    } else {
      b.style.color = '';
    }
  });
  document.querySelectorAll('.tab-page').forEach(p => p.classList.toggle('active', p.id === 'tab-' + tab));

  // Ben tab needs fixed layout — prevent body scroll and snap to top
  // Other tabs: restore body scroll
  if (tab === 'lab') {
    window.scrollTo(0, 0);
    document.body.style.overflow = 'hidden';
  } else {
    document.body.style.overflow = '';
  }

  movePill(tab);
  setPillAccent(tab);
  const tabGlows = { predictions: 'rgba(20,184,166,0.06)', lab: 'rgba(20,184,166,0.06)' };
  document.documentElement.style.setProperty('--tab-glow', tabGlows[tab] || tabGlows.predictions);
  if (tab === 'predictions') {
    let stale = SLATE_LOADED_AT > 0 && (Date.now() - SLATE_LOADED_AT) > 5 * 60 * 1000;
    let dateStale = SLATE && SLATE.locked && SLATE.date && SLATE.date !== _etToday();
    if (stale || dateStale) loadSlate();
  }
  if (tab === 'lab') initLabPage();  // has its own staleness guard
  // Re-position toggle pills in newly visible tab (elements may have been hidden)
  setTimeout(_initAllTogglePills, 30);
}

// ── Predict sub-page routing ──
let PREDICT_SUB = 'slate'; // 'slate' | 'game'

function switchPredictSub(sub) {
  PREDICT_SUB = sub;
  const nav = document.getElementById('predictSubNav');
  let activeBtn = null;
  nav.querySelectorAll('.predict-sub-btn').forEach(b => {
    const isActive = b.dataset.sub === sub;
    b.classList.toggle('active', isActive);
    if (isActive) activeBtn = b;
  });
  moveTogglePill(nav, activeBtn, '20,184,166');
  document.querySelectorAll('.predict-sub-page').forEach(p => {
    const isActive = p.id === 'predict-sub-' + sub;
    p.classList.toggle('active', isActive);
    p.style.display = isActive ? 'block' : 'none';
  });
}

function _backToGameGrid() {
  document.getElementById('gameHeaderRow').style.display = 'none';
  document.getElementById('picksList').innerHTML = '';
  const si = document.getElementById('strategyInsight');
  if (si) si.style.display = 'none';
  document.getElementById('gameAnalysisEmpty').style.display = 'block';
  document.getElementById('gameSel').value = '';
}

function movePill(tab) {
  const pill = document.getElementById('bnavPill');
  const btn = document.querySelector(`.bnav-icon-btn[data-tab="${tab}"]`);
  const row = document.querySelector('.bnav-icon-row');
  if (!pill || !btn || !row) return;
  const rowRect = row.getBoundingClientRect();
  const btnRect = btn.getBoundingClientRect();
  pill.style.width = btnRect.width + 'px';
  pill.style.transform = `translateX(${btnRect.left - rowRect.left}px)`;
}

// Pill accent color per tab
const TAB_ACCENT = {
  predictions: { rgb: '212,166,64' },
  line:        { rgb: '212,166,64' },
  parlay:      { rgb: '212,166,64' },
  lab:         { rgb: '212,166,64' },
};

function setPillAccent(tab) {
  const pill = document.getElementById('bnavPill');
  if (!pill) return;
  const accent = TAB_ACCENT[tab] || TAB_ACCENT.predictions;
  pill.style.background = `linear-gradient(135deg, rgba(${accent.rgb},0.14) 0%, rgba(${accent.rgb},0.06) 100%)`;
  pill.style.border = `1px solid rgba(${accent.rgb},0.10)`;
  pill.style.boxShadow = `0 0 10px rgba(${accent.rgb},0.10), inset 0 1px 1px rgba(${accent.rgb},0.08)`;
}

// ── Global sliding pill — reusable for all toggle groups ──
// grep: moveTogglePill
function moveTogglePill(container, activeBtn, accentRgb) {
  if (!container || !activeBtn) return;
  const pill = container.querySelector('.slide-pill');
  if (!pill) return;
  const containerRect = container.getBoundingClientRect();
  const btnRect = activeBtn.getBoundingClientRect();
  const padLeft = parseFloat(getComputedStyle(container).paddingLeft) || 3;
  pill.style.width = btnRect.width + 'px';
  pill.style.transform = `translateX(${btnRect.left - containerRect.left - padLeft}px)`;
  if (accentRgb) {
    pill.style.background = `linear-gradient(135deg, rgba(${accentRgb},0.14) 0%, rgba(${accentRgb},0.06) 100%)`;
    pill.style.border = `1px solid rgba(${accentRgb},0.10)`;
    pill.style.boxShadow = `0 0 10px rgba(${accentRgb},0.10), inset 0 1px 1px rgba(${accentRgb},0.08)`;
  }
}

// Init all toggle pills on load + resize
function _initAllTogglePills() {
  // Predict sub-nav (Slate-Wide | Game)
  const predictNav = document.getElementById('predictSubNav');
  if (predictNav) {
    const activeBtn = predictNav.querySelector('.predict-sub-btn.active');
    moveTogglePill(predictNav, activeBtn, '20,184,166');
  }
  // Slate tabs (Starting 5 | Moonshot) — only if visible
  const slateTabs = document.getElementById('slateTabs');
  if (slateTabs && slateTabs.style.display !== 'none') {
    const activeBtn = slateTabs.querySelector('.mode-tab.active');
    moveTogglePill(slateTabs, activeBtn, '20,184,166');
  }
  // Line direction (Over | Under)
  const lineNav = document.getElementById('lineSubNav');
  if (lineNav) {
    const activeBtn = lineNav.querySelector('.predict-sub-btn.active');
    const dir = activeBtn && activeBtn.dataset.dir;
    const rgb = dir === 'over' ? '212,166,64' : '20,184,166';
    moveTogglePill(lineNav, activeBtn, rgb);
  }
  // Line history (All | Over | Under)
  const lineHist = document.getElementById('lineHistWrap');
  if (lineHist) {
    const activeBtn = lineHist.querySelector('.line-hist-tab.active');
    const dir = activeBtn && activeBtn.dataset.dir;
    const rgb = dir === 'over' ? '212,166,64' : dir === 'under' ? '20,184,166' : '255,255,255';
    moveTogglePill(lineHist, activeBtn, rgb);
  }
}
window.addEventListener('resize', _initAllTogglePills);

// Init pill position + accent on load
window.addEventListener('load', () => {
  setTimeout(() => { movePill('predictions'); setPillAccent('predictions'); _initAllTogglePills(); }, 30);
});

// ════════════════════════════════════════════════════════════════════
// MAGIC 8-BALL LOADER — grep: showLoader, hideLoader, oracleLoader, oracleMsg
// Shown during loadSlate() and runAnalysis() API calls
// ════════════════════════════════════════════════════════════════════
const ORACLE_MSGS = [
  'READING THE GAME',
  'CONSULTING THE ORACLE',
  'CALCULATING EDGE',
  'SEEKING THE LINE',
  'ANALYZING THE SLATE',
  'SENSING THE FIELD',
  'FOLLOWING THE MONEY',
];
let _oracleMsgIdx = 0;
let _oracleMsgTimer = null;

function showLoader() {
  const el = document.getElementById('oracleLoader');
  const msg = document.getElementById('oracleMsg');
  if (!el) return;
  // Clear any existing interval to prevent duplicate message rotators
  clearInterval(_oracleMsgTimer);
  // Rotate messages every 1.8s while visible
  _oracleMsgIdx = Math.floor(Math.random() * ORACLE_MSGS.length);
  if (msg) msg.textContent = ORACLE_MSGS[_oracleMsgIdx];
  _oracleMsgTimer = setInterval(() => {
    _oracleMsgIdx = (_oracleMsgIdx + 1) % ORACLE_MSGS.length;
    if (msg) {
      msg.style.animation = 'none';
      msg.offsetHeight; // reflow to restart animation
      msg.style.animation = '';
      msg.textContent = ORACLE_MSGS[_oracleMsgIdx];
    }
  }, 1800);
  el.classList.add('visible');
}

function hideLoader() {
  clearInterval(_oracleMsgTimer);
  const el = document.getElementById('oracleLoader');
  if (el) el.classList.remove('visible');
}

// ════════════════════════════════════════════════════════════════════
// APP-LEVEL EAGER HYDRATION — grep: _hydrateApp, _HydrationState
// Fetches all critical endpoints in parallel on startup, caches results in globals.
// Eliminates tab-switch loading spinners by pre-loading all data at app init time.
// ════════════════════════════════════════════════════════════════════

let _HydrationState = {
  phase: 0,
  progress: '0/7',
  loaded: {},
  startedAt: 0,
  completedAt: 0,
};

// Define all endpoints to hydrate in parallel.
// Each has: name (cache key), url, timeout (ms), optional init function.
const _HYDRATION_ENDPOINTS = [
  { name: 'slate', url: '/api/slate', timeout: 30000, init: null },
  { name: 'games', url: '/api/games', timeout: 10000, init: function(data) { populateGameSelector(data.data || data); } },
];

/**
 * Eager hydration: fetch all critical endpoints in parallel, store in globals.
 * Shows progress on magic 8-ball loader. Falls back gracefully on per-endpoint failures.
 */
async function _hydrateApp() {
  _HydrationState.startedAt = Date.now();
  showLoader('HYDRATING...');

  const _fetchHydrationEndpoint = async function(ep) {
    try {
      const r = await fetchWithTimeout(ep.url, {}, ep.timeout);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();

      // Store in appropriate global
      if (ep.name === 'slate') {
        SLATE = TransformSlateData(data);
        SLATE_LOADED_AT = Date.now();
      } else if (ep.name === 'games') {
        if (ep.init) ep.init(data);
      }

      _HydrationState.loaded[ep.name] = true;
      _HydrationState.phase++;
      _HydrationState.progress = `${_HydrationState.phase}/${_HYDRATION_ENDPOINTS.length}`;
      console.log(`[hydrate] ${ep.name} loaded (${_HydrationState.progress})`);
      return { name: ep.name, ok: true };
    } catch(e) {
      console.warn(`[hydrate] ${ep.name} failed:`, e.message);
      _HydrationState.loaded[ep.name] = ep.optional ? 'skipped' : false;
      if (!ep.optional) _HydrationState.phase++;
      return { name: ep.name, ok: false, error: e.message };
    }
  };

  // Hydrate: slate + games — must complete before app is interactive
  const criticalResults = await Promise.allSettled(_HYDRATION_ENDPOINTS.map(_fetchHydrationEndpoint));
  _HydrationState.completedAt = Date.now();
  const critical_ms = _HydrationState.completedAt - _HydrationState.startedAt;
  console.log(`[hydrate] complete in ${critical_ms}ms`);

  hideLoader();
  _postHydrateRender();

  return criticalResults;
}

/**
 * After hydration, run all post-load rendering that loadSlate() normally handles.
 * This is the critical piece: hydration fetches data into globals, but without
 * rendering calls the UI stays blank.
 */
function _postHydrateRender() {
  // ── Slate / Predict tab rendering ──
  if (SLATE) {
    if (SLATE.error === 'slate_failed') {
      const tabs = _el('slateTabs');
      const list = _el('slateList');
      if (tabs) tabs.style.display = 'none';
      if (list) list.innerHTML =
        '<div class="empty-state">' +
          '<span class="icon">📡</span>' +
          '<p>The projection pipeline is rebuilding.<br>Tap Retry to try again.</p>' +
          '<button type="button" class="secondary-btn" style="margin-top:12px" onclick="loadSlate()">Try again</button>' +
        '</div>';
      return;
    }

    // Persist valid slate to localStorage for cold-start recovery
    if (SLATE.lineups?.chalk?.length || SLATE.lineups?.upside?.length) {
      try {
        const lsKey = 'slate_' + (SLATE.date || '');
        localStorage.setItem(lsKey, JSON.stringify(SLATE));
        const todayET = _etToday();
        for (const k of Object.keys(localStorage)) {
          if (k.startsWith('slate_') && k !== lsKey && k !== 'slate_' + todayET) {
            localStorage.removeItem(k);
          }
        }
      } catch(e) {}
    }

    // Show slate tabs and position the active pill
    const slateTabsEl = _el('slateTabs');
    if (slateTabsEl) {
      slateTabsEl.style.display = 'flex';
      setTimeout(function() {
        const activeBtn = slateTabsEl.querySelector('.mode-tab.active');
        moveTogglePill(slateTabsEl, activeBtn, '20,184,166');
      }, 30);
    }

    LAB.cachedStatus = null; LAB.statusFetchedAt = 0;

    // Populate game selector from slate games
    if (SLATE.games && SLATE.games.length) { populateGameSelector(SLATE.games); }


    // Header meta badges
    const headerMeta = _el('headerMeta');
    if (headerMeta) {
      const isLocked = SLATE.locked === true;
      const totalGames = (SLATE.games || []).length;
      const draftable = (SLATE.games || []).filter(function(g) { return !g.locked; }).length;
      const gameCount = isLocked ? totalGames : draftable;
      const slateDate = SLATE.date || _etToday();
      const d = new Date(slateDate + 'T12:00:00');
      const dateStr = _fmtDateShort(d);
      const timeNow = new Date().toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', timeZone: 'America/Chicago' });
      let lockLabel = 'Locked';
      if (isLocked && SLATE.lock_time) {
        try {
          const raw = SLATE.lock_time;
          const utcStr = raw.endsWith('Z') ? raw : raw.replace(/([+-]\d{2}:\d{2})?$/, 'Z');
          const lt = new Date(utcStr);
          const ltStr = lt.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', timeZone: 'America/Chicago' });
          lockLabel = 'Locked ' + ltStr + ' CT';
        } catch(e) {}
      }
      const statusBadge = isLocked
        ? '<div class="header-badge locked">' + lockLabel + '</div>'
        : '<div class="header-badge">Updated ' + timeNow + ' CT</div>';
      const countBadge = '<div class="header-badge">' + gameCount + ' Game' + (gameCount !== 1 ? 's' : '') + ' &middot; ' + dateStr + '</div>';
      headerMeta.innerHTML = countBadge + statusBadge;
    }

    // Render the default slate view (Starting 5)
    switchSlate('chalk');
    savePredictions();
    showLateDraftBanner();

    // Pre-warm Lab status
    _fetchJson('/api/lab/status', 10000).then(function(s) {
      LAB.cachedStatus = s;
      LAB.statusFetchedAt = Date.now();
    }).catch(function(e) { console.warn('[lab/status] pre-warm failed:', e); });

    // Next-day polling when slate is complete
    if (SLATE.all_complete && !_slateNextDayPoll) {
      _slateNextDayPoll = setInterval(function() {
        const prevDate = SLATE.date;
        loadSlate().then(function() {
          if (SLATE.date !== prevDate || !SLATE.all_complete) {
            clearInterval(_slateNextDayPoll);
            _slateNextDayPoll = null;
          }
        }).catch(function() {});
      }, 5 * 60 * 1000);
    }

    // Auto-refresh when lock window opens
    if (!SLATE.locked && !SLATE.all_complete) {
      const now = Date.now();
      const startMs = (SLATE.games || [])
        .map(function(g) { return g.startTime ? new Date(g.startTime).getTime() : Infinity; })
        .filter(function(t) { return isFinite(t); });
      if (startMs.length) {
        const lockAt = Math.min.apply(null, startMs) - 5 * 60 * 1000;
        const delay = lockAt - now;
        if (delay > 0 && delay < 4 * 60 * 60 * 1000) {
          clearTimeout(_slateAutoRefreshTimer);
          _slateAutoRefreshTimer = setTimeout(function() { _slateAutoRefreshTimer = null; if (!SLATE.locked) loadSlate(); }, delay + 2000);
        } else if (delay <= 0 && _slateAutoRefreshCount < 3) {
          _slateAutoRefreshCount++;
          clearTimeout(_slateAutoRefreshTimer);
          _slateAutoRefreshTimer = setTimeout(function() { _slateAutoRefreshTimer = null; if (!SLATE.locked && !SLATE.all_complete) loadSlate(); }, 5000);
        }
      }
    }
    if (SLATE.locked) _slateAutoRefreshCount = 0;
  } else {
    // Slate hydration failed — show retry
    const list = _el('slateList');
    if (list) list.innerHTML =
      '<div class="empty-state">' +
        '<span class="icon">📡</span>' +
        '<p>Could not load slate.<br>Tap Retry to try again.</p>' +
        '<button type="button" class="secondary-btn" style="margin-top:12px" onclick="loadSlate()">Try again</button>' +
      '</div>';
  }

  // ── Parlay: render ticket if hydrated ──
  if (PARLAY_STATE && PARLAY_STATE.status === 'success' && PARLAY_STATE.data) {
    const data = PARLAY_STATE.data;
    if (data.legs && data.legs.length > 0) {
      PARLAY_LOADED_DATE = _etToday();
      renderParlayTicket(data);
    }
  }

  // ── Line: paint LOTD + history from hydration (first Line open skips duplicate /api/line-of-the-day) ──
  if (LINE_LOTD_STATE && LINE_LOTD_STATE.status === 'success' && LINE_LOTD_STATE.data) {
    const lotdData = LINE_LOTD_STATE.data;
    LINE_LOADED_DATE = _etToday();
    _renderLineLOTDFromState();
    if (lotdData.pick) {
      _startLineLivePoll(LINE_OVER_PICK, LINE_UNDER_PICK);
    }
  }
  if (LINE_HIST_DATA && LINE_HIST_DATA.picks && LINE_HIST_DATA.picks.length) {
    const hw = _el('lineHistoryWrap');
    if (hw) {
      hw.style.display = 'block';
      renderLineHistory(LINE_HIST_DATA);
      filterLineHistory(LINE_DIR);
    }
  }
}

// ════════════════════════════════════════════════════════════════════
// APP INIT — grep: DOMContentLoaded, initGameSelector, loadSlate, startup
// ════════════════════════════════════════════════════════════════════
(function init() {
  // Show predict sub-nav on initial load (Predict is the default tab)
  document.getElementById('predictSubNav').style.display = 'flex';

  // NEW: Eager hydration replaces lazy loadSlate + initGameSelector
  // This fetches all critical endpoints in parallel before showing the app.
  _hydrateApp();
})();

// Visibility-based stale slate detection: if the user leaves the app open
// overnight and returns the next day with a locked previous-day slate,
// reload automatically so they see today's picks without a manual refresh.
document.addEventListener('visibilitychange', function() {
  if (document.visibilityState !== 'visible') return;
  if (!SLATE || !SLATE.locked) return;
  if (!SLATE.date) return;
  if (SLATE.date !== _etToday()) loadSlate();
});

// ════════════════════════════════════════════════════════════════════
// GAME SELECTOR — grep: initGameSelector, populateGameSelector, gameSel, /api/games
// ════════════════════════════════════════════════════════════════════
/** Populate the game dropdown from an array of game objects. Source-agnostic (from /api/games or SLATE.games). */
function populateGameSelector(games, opts) {
  const sel = _el('gameSel');
  const btn = _el('analyzeBtn');
  if (!sel) return;
  opts = opts || {};
  const slateLocked = opts.slateLocked === true;
  if (!Array.isArray(games) || !games.length) {
    sel.innerHTML = '<option value="">No games today</option>';
    return;
  }
  sel.innerHTML = '<option value="">Select a game...</option>';
  games.forEach(g => {
    const o = document.createElement('option');
    o.value = g.gameId || g.id || '';
    let label = (g.label || '') + (g.total ? ` \u00B7 O/U ${g.total}` : '');
    if (g.startTime) {
      const start = new Date(g.startTime);
      label += ` \u00B7 ${start.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })}`;
    }
    const locked = slateLocked || (g.locked === true) || (g.startTime && (new Date(g.startTime).getTime() - Date.now()) <= 5 * 60 * 1000);
    if (locked) label += ' \uD83D\uDD12';
    o.textContent = label;
    sel.appendChild(o);
  });
  sel.onchange = () => { if (btn) btn.disabled = !sel.value; };
  if (btn) btn.disabled = !sel.value;
  // Also populate quick-pick game cards
  _populateGameQuickPicks(games, slateLocked);
}

/** Render tappable game cards in the Game sub-page empty state */
function _populateGameQuickPicks(games, slateLocked) {
  const grid = _el('gameQuickPicks');
  if (!grid || !Array.isArray(games) || !games.length) return;
  grid.innerHTML = games.map(g => {
    const gid = g.gameId || g.id || '';
    const parts = (g.label || '').split(/ vs | @ /);
    const away = parts[0] || '?';
    const home = parts[1] || '?';
    const locked = slateLocked || (g.locked === true) || (g.startTime && (new Date(g.startTime).getTime() - Date.now()) <= 5 * 60 * 1000);
    let timeStr = '';
    if (g.startTime) {
      const s = new Date(g.startTime);
      timeStr = s.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
    }
    const totalStr = g.total ? 'O/U ' + g.total : '';
    const meta = [timeStr, totalStr].filter(Boolean).join(' · ');
    return `<div role="button" tabindex="0" onclick="document.getElementById('gameSel').value='${gid}';document.getElementById('gameSel').dispatchEvent(new Event('change'));runAnalysis()"
      onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();this.click()}"
      style="background:var(--surface);border:1px solid var(--border);border-top-color:rgba(255,255,255,0.14);border-radius:var(--radius-card);padding:14px;cursor:pointer;transition:all 0.18s;backdrop-filter:blur(16px) saturate(150%);box-shadow:0 2px 12px rgba(0,0,0,0.22), 0 1px 0 rgba(255,255,255,0.08) inset"
      onmouseover="this.style.borderColor='rgba(20,184,166,0.3)'" onmouseout="this.style.borderColor='rgba(255,255,255,0.09)'">
      <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.92rem;font-weight:800;color:var(--text)">${_escapeHtml(away)} <span style="color:var(--muted);font-weight:600">vs</span> ${_escapeHtml(home)}</div>
      <div style="font-size:0.65rem;color:var(--muted);margin-top:4px">${_escapeHtml(meta)}${locked ? ' 🔒' : ''}</div>
    </div>`;
  }).join('');
}

async function initGameSelector() {
  const sel = document.getElementById('gameSel');
  const btn = document.getElementById('analyzeBtn');
  try {
    const r = await fetchWithTimeout('/api/games', {}, 10000);
    if (!r.ok) throw new Error('Failed to load games (' + r.status + ')');
    const games = await r.json();
    populateGameSelector(games);
  } catch(e) {
    if (SLATE && Array.isArray(SLATE.games) && SLATE.games.length) {
      populateGameSelector(SLATE.games);
    } else {
      if (sel) sel.innerHTML = '<option value="">Offline — check backend</option>';
      // Self-heal: loadSlate() runs in parallel and may still be in-flight on cold start
      // (can take 10-20s). Check once after 5s in case SLATE has since populated.
      setTimeout(function() {
        if (sel && sel.options.length === 1 && sel.options[0].value === '' &&
            SLATE && Array.isArray(SLATE.games) && SLATE.games.length) {
          populateGameSelector(SLATE.games);
        }
      }, 5000);
    }
  }
}

// ════════════════════════════════════════════════════════════════════
// SLATE — grep: loadSlate, /api/slate, slateList, Starting 5, Moonshot
// ════════════════════════════════════════════════════════════════════
let _slateRetryCount = 0;
let _LINE_PREWARMED_DATE = '';
let _PARLAY_PREWARMED_DATE = '';

function _prewarmLineAndParlay(todayET) {
  // Run at most once per ET date to avoid repeated background churn.
  // Skip endpoints already loaded by _hydrateApp() to prevent duplicate requests.
  if (!todayET) return;
  if (_LINE_PREWARMED_DATE !== todayET) {
    _LINE_PREWARMED_DATE = todayET;
    // Skip line fetch if hydration already loaded it
    if (!_HydrationState.loaded['line']) {
      _fetchJson('/api/line-of-the-day', 90000)
        .then(function(data) {
          LINE_LOTD_STATE = asyncStateSuccess(LINE_LOTD_STATE, data);
        })
        .catch(function() {});
    }
    // Skip line_history fetch if hydration already loaded it
    if (!_HydrationState.loaded['line_history']) {
      _fetchJson('/api/line-history', 25000)
        .then(function(hist) {
          LINE_HIST_DATA = hist;
          LINE_HISTORY_STATE = asyncStateSuccess(LINE_HISTORY_STATE, hist);
        })
        .catch(function() {});
    }
  }
  if (_PARLAY_PREWARMED_DATE !== todayET) {
    _PARLAY_PREWARMED_DATE = todayET;
    _fetchJson('/api/parlay-history', 15000)
      .then(function(hist) {
        if (hist && hist.error) {
          PARLAY_HIST_ERROR = hist.narrative || hist.error;
          var tabPw = _el('tab-parlay');
          if (tabPw && tabPw.classList.contains('active')) {
            renderParlayHistoryError(PARLAY_HIST_ERROR);
          }
          return;
        }
        PARLAY_HIST_ERROR = null;
        PARLAY_HIST_DATA = hist;
        PARLAY_HIST_LOADED_DATE = todayET;
        // Render immediately so history is visible even if user opens Parlay tab
        // after prewarm completes (fixes race where initParlayPage skips fetch
        // because PARLAY_HIST_LOADED_DATE is already set but DOM is empty)
        renderParlayHistory(hist);
      })
      .catch(function(err) {
        console.warn('[parlay] history prewarm failed:', err);
        PARLAY_HIST_ERROR = 'Could not load parlay history.';
        var tabPe = _el('tab-parlay');
        if (tabPe && tabPe.classList.contains('active')) {
          renderParlayHistoryError(PARLAY_HIST_ERROR);
        }
      });
  }
}

function _cancelSlateRetry() {
  if (_slateFailRetryTimer) { clearInterval(_slateFailRetryTimer); _slateFailRetryTimer = null; }
  _slateFailRetryCount = 0;
}

async function loadSlate() {
  if (_slateLoadInFlight) return; // Prevent concurrent calls (visibilitychange + poll overlap)
  if (_slateFailRetryTimer) { clearInterval(_slateFailRetryTimer); _slateFailRetryTimer = null; } // Cancel pending auto-retry
  _slateLoadInFlight = true;
  SLATE_STATE = asyncStateLoading(SLATE_STATE);
  showLoader();
  // Only show skeletons on first load — background polls keep existing cards visible
  if (!SLATE) renderSkeletons('slateList', 5);
  try {
    const r = await fetchWithTimeout('/api/slate', {}, 30000, _getTabSignal('predictions'));
    hideLoader();
    if (!r.ok) throw new Error('Slate fetch failed');
    const slateData = TransformSlateData(await r.json());
    SLATE_STATE = asyncStateSuccess(SLATE_STATE, slateData);
    if (slateData.error === 'slate_failed') {
      const tabs = _el('slateTabs');
      const list = _el('slateList');
      if (tabs) tabs.style.display = 'none';
      SLATE = slateData;
      SLATE_LOADED_AT = 0;
      initGameSelector(); // fallback: try to fill game dropdown when slate failed
      const _maxRetries = 3;
      const _retryDelays = [8000, 20000, 40000];
      if (_slateFailRetryCount < _maxRetries) {
        const _delay = _retryDelays[_slateFailRetryCount] || 40000;
        const _secs = Math.round(_delay / 1000);
        _slateFailRetryCount++;
        if (list) list.innerHTML = `
          <div class="empty-state">
            <span class="icon">📡</span>
            <p>Slate temporarily unavailable.<br>Retrying in <span id="slateRetryCountdown">${_secs}</span>s&hellip;</p>
            <button type="button" class="secondary-btn" style="margin-top:12px" onclick="_cancelSlateRetry();_slateFailRetryCount=0;loadSlate()">Retry now</button>
          </div>`;
        let _cd = _secs;
        if (_slateFailRetryTimer) clearInterval(_slateFailRetryTimer);
        _slateFailRetryTimer = setInterval(function() {
          _cd--;
          const el = _el('slateRetryCountdown');
          if (el) el.textContent = _cd;
          if (_cd <= 0) {
            clearInterval(_slateFailRetryTimer);
            _slateFailRetryTimer = null;
            loadSlate();
          }
        }, 1000);
      } else {
        _slateFailRetryCount = 0;
        if (list) list.innerHTML = `
          <div class="empty-state">
            <span class="icon">📡</span>
            <p>The projection pipeline is rebuilding.<br>Auto-retrying in <span id="slateRetryCountdown">120</span>s.</p>
            <button type="button" class="secondary-btn" style="margin-top:12px" onclick="_cancelSlateRetry();_slateFailRetryCount=0;loadSlate()">Try now</button>
          </div>`;
        let _cd = 120;
        if (_slateFailRetryTimer) clearInterval(_slateFailRetryTimer);
        _slateFailRetryTimer = setInterval(function() {
          _cd--;
          const el = _el('slateRetryCountdown');
          if (el) el.textContent = _cd;
          if (_cd <= 0) {
            clearInterval(_slateFailRetryTimer);
            _slateFailRetryTimer = null;
            loadSlate();
          }
        }, 1000);
      }
      return;
    }
    const isEmptyLocked = slateData.locked && !slateData.lineups?.chalk?.length;
    if (isEmptyLocked) {
      // Cold-start after lock: server has no cache. Try in-memory SLATE first, then localStorage.
      const lsKey = `slate_${slateData.date || ''}`;
      const todayET = _etToday();
      let restored = SLATE;
      // Discard in-memory SLATE if it's from a different ET date
      if (restored && restored.date && restored.date !== todayET) restored = null;
      if (!restored || !restored.lineups?.chalk?.length) {
        try {
          const raw = localStorage.getItem(lsKey);
          if (raw) {
            const parsed = JSON.parse(raw);
            // Only restore if the stored slate is for today (ET date)
            restored = (parsed?.date === todayET) ? parsed : null;
          }
        } catch(e) {}
      }
      if (restored && restored.lineups?.chalk?.length) {
        SLATE = { ...restored, locked: true, lock_time: slateData.lock_time || restored.lock_time, all_complete: slateData.all_complete || false };
      } else {
        SLATE = slateData;
      }
    } else {
      SLATE = slateData;
      // Persist valid slate to localStorage for cold-start recovery after lock.
      if (slateData.lineups?.chalk?.length || slateData.lineups?.upside?.length) {
        try {
          const lsKey = `slate_${slateData.date || ''}`;
          localStorage.setItem(lsKey, JSON.stringify(slateData));
          // Prune any slate keys older than 2 days to avoid unbounded storage growth
          const todayET = _etToday();
          for (const k of Object.keys(localStorage)) {
            if (k.startsWith('slate_') && k !== lsKey && k !== `slate_${todayET}`) {
              localStorage.removeItem(k);
            }
          }
        } catch(e) {}
      }
    }
    const slateTabsEl = _el('slateTabs');
    if (slateTabsEl) {
      slateTabsEl.style.display = 'flex';
      setTimeout(() => {
        const activeBtn = slateTabsEl.querySelector('.mode-tab.active');
        moveTogglePill(slateTabsEl, activeBtn, '20,184,166');
      }, 30);
    }
    SLATE_LOADED_AT = Date.now();
    LAB.cachedStatus = null; LAB.statusFetchedAt = 0;

    if (SLATE.games && SLATE.games.length) { populateGameSelector(SLATE.games); }
    _prewarmLineAndParlay(SLATE.date || _etToday());

    // When today's slate is fully done, poll every 5 min until next-day games appear.
    if (SLATE.all_complete && !_slateNextDayPoll) {
      _slateNextDayPoll = setInterval(() => {
        const prevDate = SLATE.date;
        // Wait for loadSlate to actually complete before checking if date changed.
        // Previous 3s setTimeout raced with the async fetch (can take 10-30s on cold start).
        loadSlate().then(() => {
          if (SLATE.date !== prevDate || !SLATE.all_complete) {
            clearInterval(_slateNextDayPoll);
            _slateNextDayPoll = null;
          }
        }).catch(() => {});
      }, 5 * 60 * 1000);
    }

    // If slate not yet locked AND games haven't all finished, schedule an auto-re-fetch
    // when the lock window opens so savePredictions fires once the slate locks.
    // Skip entirely when all_complete — no reason to re-fetch a finished slate.
    if (!SLATE.locked && !SLATE.all_complete) {
      const now = Date.now();
      const startMs = (SLATE.games || [])
        .map(g => g.startTime ? new Date(g.startTime).getTime() : Infinity)
        .filter(t => isFinite(t));
      if (startMs.length) {
        const lockAt = Math.min(...startMs) - 5 * 60 * 1000; // 5 min before first tipoff
        const delay = lockAt - now;
        if (delay > 0 && delay < 4 * 60 * 60 * 1000) { // only arm if ≤ 4 hours out
          clearTimeout(_slateAutoRefreshTimer);
          _slateAutoRefreshTimer = setTimeout(() => { _slateAutoRefreshTimer = null; if (!SLATE.locked) loadSlate(); }, delay + 2000);
        } else if (delay <= 0 && _slateAutoRefreshCount < 3) {
          // Lock window already passed but backend returned unlocked (stale cache).
          // Retry after a short delay to pick up the locked state (max 3 retries).
          _slateAutoRefreshCount++;
          clearTimeout(_slateAutoRefreshTimer);
          _slateAutoRefreshTimer = setTimeout(() => { _slateAutoRefreshTimer = null; if (!SLATE.locked && !SLATE.all_complete) loadSlate(); }, 5000);
        }
      }
    }

    if (SLATE.locked) _slateAutoRefreshCount = 0; // Reset retry cap on successful lock

    // Header meta badges (top-right: game count + date, lock/updated time)
    const headerMeta = document.getElementById('headerMeta');
    if (headerMeta) {
      const isLocked = SLATE.locked === true;
      const totalGames = (SLATE.games || []).length;
      const draftable = (SLATE.games || []).filter(function(g) { return !g.locked; }).length;
      const gameCount = isLocked ? totalGames : draftable;
      const slateDate = SLATE.date || _etToday();
      const d = new Date(slateDate + 'T12:00:00');
      const dateStr = _fmtDateShort(d);
      const timeNow = new Date().toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', timeZone: 'America/Chicago' });
      let lockLabel = 'Locked';
      if (isLocked && SLATE.lock_time) {
        try {
          const raw = SLATE.lock_time;
          const utcStr = raw.endsWith('Z') ? raw : raw.replace(/([+-]\d{2}:\d{2})?$/, 'Z');
          const lt = new Date(utcStr);
          const ltStr = lt.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', timeZone: 'America/Chicago' });
          lockLabel = 'Locked ' + ltStr + ' CT';
        } catch(e) {}
      }
      const statusBadge = isLocked
        ? '<div class="header-badge locked">' + lockLabel + '</div>'
        : '<div class="header-badge">Updated ' + timeNow + ' CT</div>';
      const countBadge = '<div class="header-badge">' + gameCount + ' Game' + (gameCount !== 1 ? 's' : '') + ' &middot; ' + dateStr + '</div>';
      headerMeta.innerHTML = countBadge + statusBadge;
    }

    switchSlate('chalk');
    savePredictions();
    showLateDraftBanner();
    // Pre-warm Lab status so Ben tab opens without blocking on /api/lab/status
    _fetchJson('/api/lab/status', 10000).then(function(s) {
      LAB.cachedStatus = s;
      LAB.statusFetchedAt = Date.now();
    }).catch(function(e) { console.warn('[lab/status] pre-warm failed:', e); });
    _slateRetryCount = 0; // reset AFTER all processing succeeds (not before)
  } catch(e) {
    // Auto-retry up to 2 times — cold start warms the function so retries succeed fast
    if (_slateRetryCount < 2) {
      _slateRetryCount++;
      console.warn('[slate] Auto-retry ' + _slateRetryCount + '/2 after: ' + e.message);
      let msg = document.getElementById('oracleMsg');
      if (msg) msg.textContent = _slateRetryCount === 1 ? 'WARMING UP' : 'ALMOST THERE';
      setTimeout(loadSlate, _slateRetryCount * 2000); // 2s, 4s backoff
      return;
    }
    _slateRetryCount = 0;
    SLATE_STATE = asyncStateError(SLATE_STATE, e);
    hideLoader();
    document.getElementById('slateList').innerHTML = `
      <div class="empty-state">
        <span class="icon">📡</span>
        <p>Slate didn't load.<br>Tap Retry, or try again in a moment.</p>
        <button type="button" class="secondary-btn" style="margin-top:12px" onclick="loadSlate()">Try again</button>
      </div>`;
    initGameSelector(); // fallback: try to fill game dropdown when slate fetch failed
  } finally {
    _slateLoadInFlight = false;
  }
}

function switchSlate(tab) {
  const container = document.getElementById('slateTabs');
  let activeBtn = null;
  container.querySelectorAll('.mode-tab').forEach(t => {
    const isActive = t.dataset.t === tab;
    t.classList.toggle('active', isActive);
    if (isActive) activeBtn = t;
  });
  moveTogglePill(container, activeBtn, '20,184,166');
  if (SLATE) renderCards((SLATE.lineups && SLATE.lineups[tab]) || [], 'slateList', tab);
  showLateDraftBanner();
}

// ════════════════════════════════════════════════════════════════════
// PER-GAME ANALYSIS — grep: runAnalysis, /api/picks, analyzeBtn, game analysis
// ════════════════════════════════════════════════════════════════════
async function runAnalysis() {
  if (isFetching) return;
  const sel = document.getElementById('gameSel');
  const btn = document.getElementById('analyzeBtn');
  const banner = document.getElementById('errorBanner');
  const gameId = sel.value;
  if (!gameId) return;

  isFetching = true;
  PICKS_STATE = asyncStateLoading(PICKS_STATE);
  banner.style.display = 'none';
  document.getElementById('gameAnalysisEmpty').style.display = 'none';
  btn.disabled = true;
  btn.innerHTML = '<div class="spinner"></div> Analyzing...';
  renderSkeletons('picksList', 5);
  showLoader();

  const _requestedGameId = gameId;
  try {
    const r = await fetchWithTimeout(`/api/picks?gameId=${gameId}`, {}, 15000, _getTabSignal('predictions'));
    hideLoader();
    if (document.getElementById('gameSel').value !== _requestedGameId) {
      isFetching = false;
      btn.disabled = false; btn.innerHTML = 'Analyze Draft';
      return;
    }
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      const errMsg = err.error || (r.status === 503 ? 'Server warming up — tap Retry' : r.status === 429 ? 'Too many requests — wait a moment' : `Couldn't load picks (${r.status})`);
      throw new Error(errMsg);
    }
    const picksData = TransformPicksData(await r.json());
    PICKS_STATE = asyncStateSuccess(PICKS_STATE, picksData);
    if (picksData.locked && !picksData.lineups?.the_lineup?.length && PICKS_DATA) {
      PICKS_DATA.locked = true;
    } else {
      PICKS_DATA = picksData;
    }
    // Show the header row (back button + THE LINE UP + game script side-by-side)
    document.getElementById('gameHeaderRow').style.display = 'flex';

    const badge = document.getElementById('gameScriptBadge');
    const strat = PICKS_DATA.strategy;
    if (strat && strat.label) {
      const total = PICKS_DATA.game?.total;
      const spread = PICKS_DATA.game?.spread ? Math.abs(PICKS_DATA.game.spread) : null;
      let badgeText = `<b>${_escapeHtml(strat.label)}</b>`;
      const parts = [];
      if (total) parts.push(`O/U ${total}`);
      if (spread) parts.push(`Spread ${spread}`);
      if (parts.length) badgeText += ` · ${parts.join(' · ')}`;
      badge.innerHTML = badgeText;
      badge.style.visibility = 'visible';
      // Strategy color based on type
      if (strat.type === 'balanced') badge.style.borderColor = 'var(--color-success)';
      else if (strat.type === 'top_heavy') badge.style.borderColor = 'var(--color-danger)';
      else badge.style.borderColor = 'var(--border)';
    } else if (PICKS_DATA.gameScript) {
      const total = PICKS_DATA.game?.total;
      badge.innerHTML = `<b>${PICKS_DATA.gameScript}</b>${total ? ` · O/U ${total}` : ''}`;
      badge.style.visibility = 'visible';
      badge.style.borderColor = 'var(--border)';
    } else {
      badge.style.visibility = 'hidden';
    }

    // Strategy insight bar (below header, above cards)
    _renderStrategyInsight(PICKS_DATA.strategy);

    renderCards(PICKS_DATA.lineups.the_lineup, 'picksList', 'the_lineup');
    savePredictions();
  } catch(e) {
    PICKS_STATE = asyncStateError(PICKS_STATE, e);
    hideLoader();
    document.getElementById('picksList').innerHTML = '';
    banner.innerHTML = `⚠ ${_escapeHtml(e.message)} <button type="button" class="retry-inline-btn" onclick="runAnalysis()">Retry</button>`;
    banner.style.display = 'block';
  } finally {
    isFetching = false;
    btn.disabled = false;
    btn.innerHTML = 'Analyze Draft';
  }
}

// ════════════════════════════════════════════════════════════════════
// PER-GAME STRATEGY INSIGHT — grep: _renderStrategyInsight, strategyInsight
// ════════════════════════════════════════════════════════════════════
function _renderStrategyInsight(strategy) {
  const el = document.getElementById('strategyInsight');
  if (!el) return;
  if (!strategy || !strategy.description) {
    el.style.display = 'none';
    return;
  }
  const typeIcon = strategy.type === 'balanced' ? '⚖' : strategy.type === 'top_heavy' ? '🎯' : '📊';
  const totalMult = strategy.total_mult ? ` · Ceiling mult ${strategy.total_mult}x` : '';
  const favored = strategy.favored_team ? ` · Favored: <b>${_escapeHtml(strategy.favored_team)}</b>` : '';
  el.innerHTML = `${typeIcon} <span style="color:var(--color-text-primary);font-weight:700">${_escapeHtml(strategy.label)}</span> — ${_escapeHtml(strategy.description)}${totalMult}${favored}`;
  el.style.display = 'block';
}

// ════════════════════════════════════════════════════════════════════
// CARD RENDERING — grep: renderCards, player-card, TEAM_COLORS, tcolor, rank-badge
// ════════════════════════════════════════════════════════════════════
function renderCards(picks, containerId, mode, opts = {}) {
  const el = document.getElementById(containerId);
  if (!picks || !picks.length) {
    let msg = 'No players to display yet.';
    if (SLATE?.no_games) {
      if (SLATE.next_slate_date) {
        const d = new Date(SLATE.next_slate_date + 'T12:00:00');
        const dateStr = d.toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric' });
        msg = `No NBA games today. Next slate: <b>${_escapeHtml(dateStr)}</b>.`;
      } else {
        msg = 'No NBA games scheduled in the next 30 days.';
      }
    } else if (SLATE?.locked && SLATE?.all_complete) {
      msg = 'All games final. Check back tomorrow for the next slate.';
    } else if (SLATE?.locked) {
      msg = 'Slate is locked. Picks will appear shortly.';
    }
    el.innerHTML = `<div class="empty-state"><span class="icon">🏀</span><p>${msg}</p></div>`;
    return;
  }

  const isUpside = mode === 'upside';
  const isLog = opts.log || false;
  const scoreColor = isLog
    ? (isUpside ? 'var(--log-upside)' : 'var(--log-chalk)')
    : (isUpside ? 'var(--upside)' : 'var(--chalk)');
  const actualsMap = opts.actualsMap || {};
  const statsActualsMap = opts.statsActualsMap || {};

  const html = picks.map((p, i) => {
    const tc = TEAM_COLORS[p.team] || (isLog ? '#8ec5b8' : '#14b8a6');
    const tcAlpha = _hexToRgba(tc, 0.06);

    const actRS = isLog ? actualsMap[p.name] : undefined;
    const hasActuals = actRS !== undefined;
    const predRS = parseFloat(p.rating) || 0;
    const isHit = hasActuals && actRS >= predRS * 0.85;

    // Find ESPN box score stats for this player (fuzzy match by last name)
    let playerStats = null;
    if (isLog) {
      playerStats = statsActualsMap[p.name];
      if (!playerStats) {
        const nameLower = p.name.toLowerCase();
        for (const [k, v] of Object.entries(statsActualsMap)) {
          if (k.toLowerCase() === nameLower ||
              k.toLowerCase().split(' ').pop() === nameLower.split(' ').pop()) {
            playerStats = v; break;
          }
        }
      }
    }

    // Dynamic stat column config — drives all card stat grids from data
    const _STAT_CONFIG = [
      { k: 'pts', sk: 'season_pts', l: 'PTS' },
      { k: 'reb', sk: 'season_reb', l: 'REB' },
      { k: 'ast', sk: 'season_ast', l: 'AST' },
      { k: 'stl', sk: 'season_stl', l: 'STL' },
      { k: 'blk', sk: 'season_blk', l: 'BLK' },
    ];
    const statCols = _STAT_CONFIG
      .filter(c => p[c.k])
      .map(c => ({ v: p[c.k], avg: p[c.sk] != null ? p[c.sk] : null, l: c.l, k: c.k }));

    // Minutes pill: projected + season avg (gracefully handles missing season_min)
    const avgMin = p.season_min || p.avg_min || null;
    const minPillText = (p.predMin && avgMin)
      ? `${p.predMin}m <span style="color:var(--color-text-muted);font-weight:600">/ ${avgMin} avg</span>`
      : (p.predMin ? `${p.predMin} min` : '');

    // Context pills (card boost + minutes)
    const contextPills = [
      p.est_mult && `<span class="stat-context-pill">+${p.est_mult}x card</span>`,
      minPillText && `<span class="stat-context-pill">${minPillText}</span>`,
      (p._decline && p._decline < 0.85) && `<span class="stat-context-pill" style="color:var(--color-danger);border-color:rgba(255,180,180,0.15)">↓${Math.round((1-p._decline)*100)}% usage</span>`,
      p._hot_streak && `<span class="stat-context-pill overperform-hot">🔥 HOT</span>`,
      p._odds_adjusted && `<span class="stat-context-pill overperform-odds">📈 ODDS EDGE</span>`,
      (p._context_adj && p._context_adj > 1.05) && `<span class="stat-context-pill overperform-ai">🧠 +${Math.round((p._context_adj - 1) * 100)}%</span>`,
      // Per-game strategy pills (value anchor + favored team indicator)
      (mode === 'the_lineup' && p._is_value_anchor) && `<span class="stat-context-pill" style="color:#f0c040;border-color:rgba(240,192,64,0.25)">⚓ ANCHOR</span>`,
      (mode === 'the_lineup' && p._favored_team) && `<span class="stat-context-pill" style="color:var(--color-success);border-color:rgba(74,222,128,0.2)">★ FAV</span>`,
      (isLog && !hasActuals && !playerStats) ? '<span class="pending-results">Waiting for results</span>' : '',
    ].filter(Boolean);

    let statsHtml;
    if (isLog && playerStats) {
      // Graded layout: shows actual ESPN stats.
      // Hit/miss vs projection only when Log has actual RS (top_performers / legacy actuals).
      const gradedCols = statCols.map(s => {
        const actual = playerStats[s.k] != null ? parseFloat(playerStats[s.k]).toFixed(0) : null;
        const proj = parseFloat(s.v) || 0;
        const colorCls = (hasActuals && actual != null) ? (parseFloat(actual) >= proj ? 'stat-hit' : 'stat-miss') : '';
        const projHtml = hasActuals ? `<div class="stat-col-proj">${s.v}</div>` : '';
        return `<div class="stat-col">
          <div class="stat-col-lbl">${s.l}</div>
          ${actual != null
            ? `<div class="stat-col-actual ${colorCls}">${actual}</div>${projHtml}`
            : `<div class="stat-col-val">${s.v}</div>`}
        </div>`;
      }).join('');
      statsHtml = `
        ${contextPills.length ? `<div class="stat-context-row">${contextPills.join('')}</div>` : ''}
        ${gradedCols ? `<div class="stat-grid-row">${gradedCols}</div>` : ''}`;
    } else {
      // Standard 3-tier: Projection (bold) / Avg (muted) / Label
      const stdCols = statCols.map(s => {
        const avgHtml = s.avg != null ? `<div class="stat-col-avg">${s.avg}</div>` : '';
        return `<div class="stat-col"><div class="stat-col-val">${s.v}</div>${avgHtml}<div class="stat-col-lbl">${s.l}</div></div>`;
      }).join('');
      statsHtml = `
        ${contextPills.length ? `<div class="stat-context-row">${contextPills.join('')}</div>` : ''}
        ${stdCols ? `<div class="stat-grid-row">${stdCols}</div>` : ''}`;
    }

    const logCardClass = isLog ? ' log-card' : '';
    const gradedClass = (isLog && hasActuals) ? (isHit ? ' graded-hit' : ' graded-miss') : '';
    const logScoreColorVar = isLog ? `--log-score-color:${scoreColor};` : '';

    // Score column: graded shows Actual RS prominently, Projected RS beneath
    let scoreColHtml;
    if (isLog && hasActuals) {
      const scoreCls = isHit ? 'score-hit' : 'score-miss';
      scoreColHtml = `
        <div class="score-col" style="--score-color:${scoreColor};">
          <div class="score-actual ${scoreCls}">${actRS}</div>
          <span class="score-label">RS</span>
          <div class="score-proj-label">Proj ${p.rating}</div>
        </div>`;
    } else {
      scoreColHtml = `
        <div class="score-col" style="--score-color:${scoreColor};">
          <div class="score-num">${p.rating}</div>
          <span class="score-label">RS</span>
          ${p.chalk_ev ? `<div style="font-family:'Barlow Condensed',sans-serif;font-size:0.62rem;font-weight:700;color:var(--color-text-muted);margin-top:2px">${p.chalk_ev} EV</div>` : ''}
        </div>`;
    }

    return `
      <div class="player-card${logCardClass}${gradedClass}" style="--tcolor:${tc};--tcolor-alpha:${tcAlpha};${logScoreColorVar}${!isLog ? `--score-color:${scoreColor};` : ''}animation-delay:${i*0.06}s">
        <div class="rank-badge">${p.slot || (i+1)}</div>
        <div class="player-info">
          <div class="player-meta">
            ${_escapeHtml(p.team || '—')} <span class="pos-pill">${_escapeHtml(p.pos || '—')}</span>${p.injury_status ? ` <span class="injury-badge">${_escapeHtml(p.injury_status)}</span>` : ''}
          </div>
          <div class="player-name">${_escapeHtml(p.name)}</div>
          <div class="stat-pills">${statsHtml}</div>
        </div>
        ${scoreColHtml}
      </div>`;
  }).join('');
  const doWrite = () => { el.innerHTML = html; };
  if (picks.length > 10) requestAnimationFrame(doWrite);
  else doWrite();
}

// ════════════════════════════════════════════════════════════════════
// SKELETON LOADING — grep: renderSkeletons, skel-block, shimmer, loading state
// ════════════════════════════════════════════════════════════════════
function renderSkeletons(containerId, count) {
  document.getElementById(containerId).innerHTML = Array.from({length: count}, (_, i) => `
    <div class="skeleton" style="animation-delay:${i*0.07}s">
      <div class="skel-block skel-circle"></div>
      <div class="skel-lines">
        <div class="skel-block skel-line wide"></div>
        <div class="skel-block skel-line narrow"></div>
      </div>
      <div class="skel-block skel-score"></div>
    </div>`).join('');
}

// ════════════════════════════════════════════════════════════════════
// PREDICTION PERSISTENCE — grep: savePredictions, /api/save-predictions, GitHub CSV
// ════════════════════════════════════════════════════════════════════
let _predSavedDate = '';
let _predSavedLockedCount = 0;
function _etToday() {
  return new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' });
}
async function savePredictions() {
  const today = _etToday();
  // Only save when the slate is locked — predictions are finalized at that point.
  // If called pre-lock, skip without consuming the flag so it can fire on next call.
  if (!SLATE || !SLATE.locked) return;
  // Count games currently past their lock window (5 min before tip).
  // Re-fire if more games have locked since last save (split-window days).
  let lockedNow = (SLATE.games || []).filter(function(g) {
    return g.startTime && (new Date(g.startTime).getTime() - Date.now()) <= 5 * 60 * 1000;
  }).length;
  if (_predSavedDate === today && lockedNow <= _predSavedLockedCount) return;
  _predSavedDate = today;
  _predSavedLockedCount = lockedNow;
  try {
    const r = await fetchWithTimeout('/api/save-predictions', { method: 'POST' }, 25000);
    if (!r.ok) throw new Error('HTTP ' + r.status);
  } catch(e) {
    _predSavedDate = ''; // allow retry on next call
    _predSavedLockedCount = 0;
    console.warn('Prediction save failed:', e);
  }
}

// ──── Late Draft: regenerate picks for remaining games when user missed lock ────
// Persisted across page reload via localStorage (keyed by ET date) so the banner
// does not reappear after the user already triggered a late draft today.
let _lateDraftTriggered = (function() {
  try { return localStorage.getItem('lateDraft_' + _etToday()) === '1'; } catch(e) { return false; }
})();

function showLateDraftBanner() {
  let wrap = _el('lateDraftBelowCards');
  if (!wrap) return;
  // Show button only when: slate is locked, at least one game has not started,
  // all_complete is false, and user has not already triggered a late draft today.
  if (!SLATE || !SLATE.locked || SLATE.all_complete || _lateDraftTriggered) {
    wrap.style.display = 'none';
    return;
  }
  let now = Date.now();
  let remaining = (SLATE.games || []).filter(function(g) {
    return g.startTime && (new Date(g.startTime).getTime() - now) > 5 * 60 * 1000;
  });
  if (remaining.length === 0) {
    wrap.style.display = 'none';
    return;
  }
  let btn = _el('lateDraftBtn');
  let msg = _el('lateDraftMsg');
  if (btn) { btn.disabled = false; btn.textContent = 'Late Draft'; }
  if (msg) { msg.style.display = 'none'; msg.textContent = ''; }
  wrap.style.display = 'block';
}

async function triggerLateDraft() {
  let wrap = _el('lateDraftBelowCards');
  let btn = _el('lateDraftBtn');
  let msg = _el('lateDraftMsg');
  if (btn) { btn.disabled = true; btn.textContent = 'Generating...'; }
  if (msg) { msg.style.display = 'none'; }
  showLoader();
  try {
    let r = await fetchWithTimeout('/api/force-regenerate?scope=remaining', {}, 60000);
    hideLoader();
    if (!r.ok) throw new Error('HTTP ' + r.status);
    let data = await r.json();
    if (data.status === 'no_remaining_games') {
      if (msg) { msg.textContent = 'All games have already started.'; msg.style.display = 'block'; }
      if (btn) { btn.disabled = false; btn.textContent = 'Late Draft'; }
      return;
    }
    if (data.status === 'regenerated') {
      _lateDraftTriggered = true;
      try { localStorage.setItem('lateDraft_' + _etToday(), '1'); } catch(e) {}
      // Update SLATE with fresh lineups from the response
      if (data.lineups) {
        SLATE.lineups = data.lineups;
        SLATE.draftable_count = data.games_regenerated || SLATE.draftable_count;
      }
      // Switch to Slate-Wide view and re-render with fresh picks
      switchPredictSub('slate');
      let _st = _el('slateTabs'); if (_st) _st.style.display = 'flex';
      switchSlate('chalk');
      // Force re-save to confirm CSV update
      _predSavedDate = '';
      _predSavedLockedCount = 0;
      savePredictions();
      // Hide header button; show brief success message below sub-nav
      if (wrap) wrap.style.display = 'none';
      if (msg) {
        msg.textContent = 'Picks regenerated for ' + (data.games_regenerated || 0) + ' remaining game' + ((data.games_regenerated || 0) !== 1 ? 's' : '') + '.';
        msg.style.color = 'var(--color-success)';
        msg.style.display = 'block';
        setTimeout(function() { if (msg) msg.style.display = 'none'; }, 3000);
      }
    } else {
      if (msg) { msg.textContent = data.message || 'Something went wrong. Try again.'; msg.style.display = 'block'; }
      if (btn) { btn.disabled = false; btn.textContent = 'Late Draft'; }
    }
  } catch(e) {
    hideLoader();
    console.warn('Late draft error:', e);
    if (msg) { msg.textContent = 'Could not regenerate. Try again.'; msg.style.display = 'block'; }
    if (btn) { btn.disabled = false; btn.textContent = 'Late Draft'; }
  }
}

// ════════════════════════════════════════════════════════
// PARLAY PAGE — grep: initParlayPage, fetchParlay, renderParlayTicket, PARLAY_STATE
// Safest 3-leg player prop parlay. Optimizes for certainty (floor).
// ════════════════════════════════════════════════════════
let PARLAY_STATE = asyncStateInitial();
let PARLAY_LOADED_DATE = '';
let PARLAY_HIST_DATA = null;
let PARLAY_HIST_LOADED_DATE = '';
let PARLAY_HIST_ERROR = null;
let PARLAY_LIVE_CACHE = null; // last SSE payload — restored instantly on tab return

function _parlaySkeletons() {
  const skel = `
    <div class="line-pick-card" style="pointer-events:none;margin-bottom:0;padding:10px 14px;border-radius:0;border-bottom:1px solid rgba(255,255,255,0.06)">
      <div class="parlay-leg-header">
        <div class="parlay-leg-who">
          <div class="skel-block skel-line" style="height:14px;width:110px"></div>
          <div class="skel-block skel-line narrow" style="height:10px;width:70px"></div>
        </div>
        <div class="parlay-leg-play">
          <div class="skel-block" style="width:42px;height:18px;border-radius:var(--radius-pill)"></div>
          <div class="skel-block skel-line" style="height:14px;width:50px"></div>
        </div>
      </div>
      <div class="parlay-leg-meta" style="margin-top:6px">
        <div class="parlay-leg-chips">
          <div class="skel-block" style="width:52px;height:20px;border-radius:6px"></div>
          <div class="skel-block" style="width:46px;height:20px;border-radius:6px"></div>
          <div class="skel-block" style="width:44px;height:20px;border-radius:6px"></div>
        </div>
        <div style="display:flex;gap:3px">
          <div class="skel-block" style="width:22px;height:20px;border-radius:4px"></div>
          <div class="skel-block" style="width:22px;height:20px;border-radius:4px"></div>
          <div class="skel-block" style="width:22px;height:20px;border-radius:4px"></div>
          <div class="skel-block" style="width:22px;height:20px;border-radius:4px"></div>
          <div class="skel-block" style="width:22px;height:20px;border-radius:4px"></div>
        </div>
      </div>
    </div>`;
  return `<div style="border-radius:var(--radius-card);overflow:hidden;border:1px solid rgba(20,184,166,0.16);box-shadow:0 4px 20px rgba(0,0,0,0.28)">${skel}${skel}${skel}</div>`;
}

function renderParlayLeg(leg, index, total, forTicket) {
  forTicket = !!forTicket;
  const dir = leg.direction || 'over';
  const statLabel = {'points':'PTS','rebounds':'REB','assists':'AST'}[leg.stat_type] || leg.stat_type.toUpperCase();
  const statShort = statLabel;
  const tcolor = TEAM_COLORS[leg.team] || '#14b8a6';
  const tcolorBorder = _hexToRgba(tcolor, 0.30);
  const isLast = index === total - 1;

  function _ceilHalf(v) { return Math.ceil(Number(v) * 2) / 2; }
  const projVal = leg.projection != null ? _ceilHalf(leg.projection) : '\u2014';
  const seasonAvg = leg.season_avg != null ? leg.season_avg : '\u2014';
  const avgMin = leg.avg_min != null ? leg.avg_min : '\u2014';

  const l5Tiles = (leg.recent_values && leg.recent_values.length)
    ? leg.recent_values.map(function(v) {
        const hit = dir === 'over' ? v >= (leg.line || 0) : v <= (leg.line || 0);
        return '<span class="parlay-leg-tile ' + (hit ? 'hit' : 'miss') + '">' + v + '</span>';
      }).join('')
    : '';

  const lineDisplay = (leg.line != null && leg.line !== '') ? Number(leg.line).toFixed(1) : '\u2014';
  const liveDot = forTicket
    ? '<span class="parlay-leg-live-dot" id="parlayLegLiveDot-' + index + '" style="display:none" aria-hidden="true"></span>'
    : '';
  const liveRow = forTicket
    ? '<div class="parlay-leg-live-row" id="parlayLegLiveRow-' + index + '" style="display:none" aria-live="polite">' +
        '<span id="parlayLegLiveStat-' + index + '"></span>' +
        '<span class="parlay-leg-live-meta" id="parlayLegLiveMeta-' + index + '"></span>' +
      '</div>' +
      '<div class="parlay-leg-progress-wrap" id="parlayLegProgressWrap-' + index + '" style="display:none">' +
        '<div class="parlay-leg-progress-fill" id="parlayLegProgress-' + index + '"></div>' +
      '</div>'
    : '';
  const l5Block = l5Tiles
    ? '<div class="parlay-leg-l5-col"><span class="parlay-leg-l5-lbl">L5</span><div class="parlay-leg-l5">' + l5Tiles + '</div></div>'
    : '';

  return '<div class="line-pick-card parlay-leg-card" id="parlayLegBlock-' + index + '" style="--tcolor:' + tcolor +
    ';--tcolor-border:' + tcolorBorder +
    ';border-radius:0;margin-bottom:0;padding:12px 16px;border-color:rgba(255,255,255,0.08);border-top-color:rgba(255,255,255,0.08);' +
    (isLast ? '' : 'border-bottom:1px solid rgba(255,255,255,0.06);') + '">' +
    '<div class="parlay-leg-header">' +
      '<div class="parlay-leg-who" style="align-items:center">' +
        liveDot +
        '<span class="parlay-leg-name">' + _escapeHtml(leg.player_name) + '</span>' +
        '<span class="parlay-leg-matchup">' + _escapeHtml(leg.team) + ' vs ' + _escapeHtml(leg.opponent) + '</span>' +
      '</div>' +
      '<div class="parlay-leg-play">' +
        '<span class="parlay-leg-pill ' + dir + '" id="parlayLegPlayPill-' + index + '">' + dir.toUpperCase() + '</span>' +
        '<span class="parlay-leg-line" id="parlayLegPlayLine-' + index + '">' + lineDisplay + ' ' + statShort + '</span>' +
      '</div>' +
    '</div>' +
    liveRow +
    '<div class="parlay-leg-meta">' +
      '<div class="parlay-leg-chips">' +
        '<span class="parlay-leg-chip accent">PROJ ' + projVal + '</span>' +
        '<span class="parlay-leg-chip">AVG ' + seasonAvg + '</span>' +
        '<span class="parlay-leg-chip">MIN ' + avgMin + '</span>' +
      '</div>' +
      l5Block +
    '</div>' +
  '</div>';
}

function renderParlayTicket(data) {
  const wrap = _el('parlayTicket');
  if (!wrap) return;
  const legs = data.legs || [];
  if (!legs.length) {
    wrap.style.display = 'none';
    const empty = _el('parlayEmpty');
    if (empty) { empty.style.display = 'block'; const msg = _el('parlayEmptyMsg'); if (msg) msg.textContent = data.narrative || 'No valid parlay found on today\'s slate.'; }
    return;
  }

  const corrMult = data.correlation_multiplier != null ? data.correlation_multiplier.toFixed(2) : '1.00';
  const numGames = [...new Set((data.legs || []).map(function(l) { return l.gameId; }))].length;

  let legsHtml = '';
  for (let i = 0; i < legs.length; i++) {
    legsHtml += renderParlayLeg(legs[i], i, legs.length, true);
  }

  const lockBadge = data.locked
    ? '<span style="font-size:0.60rem;padding:2px 7px;border-radius:99px;background:rgba(20,184,166,0.12);color:var(--parlay);font-weight:800;letter-spacing:0.06em;margin-left:6px;vertical-align:middle">LOCKED</span>'
    : '';
  const sourceBadge = data.projection_only
    ? '<span style="font-size:0.58rem;padding:2px 7px;border-radius:99px;background:rgba(255,255,255,0.08);color:var(--color-text-muted);font-weight:800;letter-spacing:0.06em;margin-left:6px;vertical-align:middle">MODEL</span>'
    : '';

  wrap.innerHTML =
    '<div style="border-radius:var(--radius-card);overflow:hidden;border:1px solid rgba(255,255,255,0.08);box-shadow:0 4px 24px rgba(20,184,166,0.08), 0 2px 12px rgba(0,0,0,0.28)">' +
      '<div style="display:flex;align-items:center;justify-content:space-between;padding:14px 16px 10px;background:rgba(20,184,166,0.05);border-bottom:1px solid rgba(255,255,255,0.06)">' +
        '<div style="display:flex;align-items:center;font-family:\'Barlow Condensed\',sans-serif;font-weight:800;font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;color:var(--parlay)">3-Leg Parlay Ticket' + sourceBadge + lockBadge + '</div>' +
        '<div style="display:flex;gap:12px;align-items:center">' +
          (numGames > 1 ? '<div style="text-align:center"><div class="line-pick-micro-lbl" style="margin-bottom:1px">Games</div><div style="font-weight:800;font-size:0.78rem;color:var(--text)">' + numGames + '</div></div>' : '') +
          (parseFloat(corrMult) !== 1.0 ? '<div style="text-align:center"><div class="line-pick-micro-lbl" style="margin-bottom:1px">Corr</div><div style="font-weight:700;font-size:0.72rem;color:var(--color-success)">' + corrMult + 'x</div></div>' : '') +
        '</div>' +
      '</div>' +
      legsHtml +
    '</div>';

  wrap.style.display = 'block';
}

async function forceRegenerateParlay() {
  const empty = _el('parlayEmpty');
  const loading = _el('parlayLoading');
  const ticket = _el('parlayTicket');
  const btn = _el('parlayForceBtn');
  if (empty) empty.style.display = 'none';
  if (ticket) ticket.style.display = 'none';
  if (btn) { btn.disabled = true; btn.textContent = 'Generating…'; }
  if (loading) { loading.innerHTML = _parlaySkeletons(); loading.style.display = 'block'; }
  PARLAY_STATE = asyncStateLoading(PARLAY_STATE);
  try {
    const r = await fetchWithTimeout('/api/parlay-force-regenerate', {}, 120000, _getTabSignal('parlay'));
    if (loading) loading.style.display = 'none';
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const data = await r.json();
    if (data.legs && data.legs.length > 0) {
      PARLAY_STATE = asyncStateSuccess(PARLAY_STATE, data);
      PARLAY_LOADED_DATE = _etToday();
      renderParlayTicket(data);
      _labSyncParlayContext(data);
      if (data.legs && data.legs.length) {
        _stopParlayLiveSse();
        _startParlayLiveSse();
      }
    } else {
      PARLAY_STATE = asyncStateError(PARLAY_STATE, data.error || 'no_valid_parlay');
      _labSyncParlayContext(data);
      if (empty) { empty.style.display = 'block'; const msg = _el('parlayEmptyMsg'); if (msg) msg.textContent = data.narrative || 'Could not generate a parlay for today.'; }
      if (btn) { btn.disabled = false; btn.textContent = 'Generate Parlay'; }
    }
  } catch (err) {
    if (loading) loading.style.display = 'none';
    PARLAY_STATE = asyncStateError(PARLAY_STATE, err);
    _labSyncParlayContext(null);
    if (empty) { empty.style.display = 'block'; const msg = _el('parlayEmptyMsg'); if (msg) msg.textContent = 'Could not reach the server. Please try again.'; }
    if (btn) { btn.disabled = false; btn.textContent = 'Generate Parlay'; }
    console.warn('[parlay] force-regen error:', err);
  }
}

async function fetchParlay(background = false) {
  const empty = _el('parlayEmpty');
  const loading = _el('parlayLoading');
  const ticket = _el('parlayTicket');
  if (empty) empty.style.display = 'none';
  if (!background) {
    if (ticket) ticket.style.display = 'none';
    if (loading) { loading.innerHTML = _parlaySkeletons(); loading.style.display = 'block'; }
  }

  PARLAY_STATE = asyncStateLoading(PARLAY_STATE);

  try {
    const r = await fetchWithTimeout('/api/parlay', {}, 45000, _getTabSignal('parlay'));
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const data = await r.json();
    if (loading) loading.style.display = 'none';

    if (data.error && (!data.legs || data.legs.length === 0)) {
      PARLAY_STATE = asyncStateError(PARLAY_STATE, data.error);
      _labSyncParlayContext(data);
      if (empty) { empty.style.display = 'block'; const msg = _el('parlayEmptyMsg'); if (msg) msg.textContent = data.narrative || 'No valid parlay found.'; }
      return;
    }

    PARLAY_STATE = asyncStateSuccess(PARLAY_STATE, data);
    PARLAY_STATE.loadedAt = Date.now();
    renderParlayTicket(data);
    _labSyncParlayContext(data);
    if (data.legs && data.legs.length) {
      _stopParlayLiveSse();
      _startParlayLiveSse();
    }
  } catch (err) {
    if (loading) loading.style.display = 'none';
    PARLAY_STATE = asyncStateError(PARLAY_STATE, err);
    _labSyncParlayContext(null);
    const hasTicketData = !!(PARLAY_STATE && PARLAY_STATE.data && (PARLAY_STATE.data.legs || []).length);
    // If we already have a rendered ticket, keep it visible and avoid contradictory giant error state.
    if (!hasTicketData && empty) {
      empty.style.display = 'block';
      const msg = _el('parlayEmptyMsg');
      if (msg) msg.textContent = 'Could not reach the server. Please try again.';
    } else if (empty) {
      empty.style.display = 'none';
    }
    console.warn('[parlay] fetch error:', err);
  }
}

let _parlayLiveES = null;

function _stopParlayLiveSse() {
  if (_parlayLiveES) {
    try { _parlayLiveES.close(); } catch (e) {}
    _parlayLiveES = null;
  }
}

function _statLabelParlay(st) {
  const m = {'points':'PTS','rebounds':'REB','assists':'AST'};
  return m[(st || '').toLowerCase()] || String(st || 'PTS').toUpperCase();
}

function _applyParlayLiveUpdate(payload) {
  if (!payload || !payload.legs || !payload.legs.length) return;
  const legs = payload.legs;
  for (let i = 0; i < legs.length; i++) {
    const L = legs[i];
    const dot = _el('parlayLegLiveDot-' + i);
    const row = _el('parlayLegLiveRow-' + i);
    const statEl = _el('parlayLegLiveStat-' + i);
    const metaEl = _el('parlayLegLiveMeta-' + i);
    const wrap = _el('parlayLegProgressWrap-' + i);
    const bar = _el('parlayLegProgress-' + i);
    const playLine = _el('parlayLegPlayLine-' + i);
    if (!row || !statEl) continue;

    const st = L.status;
    const line = L.line;
    const sc = L.stat_current;
    const sl = _statLabelParlay(L.stat_type);
    const dir = (L.direction || 'over').toLowerCase();

    if (st === 'live') {
      row.style.display = '';
      if (dot) dot.style.display = '';
      if (wrap) wrap.style.display = '';
      if (sc != null && line != null) {
        statEl.textContent = sc + ' / ' + Number(line).toFixed(1) + ' ' + sl;
      } else {
        statEl.textContent = 'Live';
      }
      if (metaEl) {
        metaEl.textContent = (L.period != null && L.clock) ? ('Q' + L.period + ' · ' + L.clock) : '';
      }
      if (bar) {
        const prevHit = bar.classList.contains('hit-locked');
        const prevMiss = bar.classList.contains('miss-locked');
        if (L.hit_threshold_met || L.leg_result_preview === 'hit') {
          bar.classList.remove('miss-locked');
          bar.classList.add('hit-locked');
          bar.style.width = '100%';
        } else if (L.leg_result_preview === 'miss') {
          bar.classList.remove('hit-locked');
          bar.classList.add('miss-locked');
          bar.style.width = '100%';
        } else if (!prevHit && !prevMiss && L.progress != null) {
          bar.classList.remove('hit-locked');
          bar.classList.remove('miss-locked');
          bar.style.width = Math.round(Math.min(100, Math.max(0, L.progress * 100))) + '%';
        }
      }
      if (playLine && sc != null && line != null) {
        playLine.textContent = Number(line).toFixed(1) + ' ' + sl + ' · live ' + sc;
      }
    } else if (st === 'final') {
      row.style.display = '';
      if (dot) dot.style.display = 'none';
      if (wrap) wrap.style.display = '';
      if (metaEl) metaEl.textContent = 'Final';
      if (sc != null && line != null) {
        statEl.textContent = sc + ' / ' + Number(line).toFixed(1) + ' ' + sl;
      } else {
        statEl.textContent = 'Final';
      }
      if (bar) {
        if (L.leg_result_preview === 'hit' || L.hit_threshold_met) {
          bar.classList.remove('miss-locked');
          bar.classList.add('hit-locked');
          bar.style.width = '100%';
        } else if (L.leg_result_preview === 'miss') {
          bar.classList.remove('hit-locked');
          bar.classList.add('miss-locked');
          bar.style.width = '100%';
        } else {
          bar.classList.remove('hit-locked');
          bar.classList.remove('miss-locked');
          bar.style.width = L.progress != null ? Math.round(Math.min(100, Math.max(0, L.progress * 100))) + '%' : '0%';
        }
      }
      if (playLine && sc != null && line != null) {
        playLine.textContent = Number(line).toFixed(1) + ' ' + sl;
      }
    } else {
      row.style.display = 'none';
      if (dot) dot.style.display = 'none';
      if (wrap) wrap.style.display = 'none';
      if (bar) {
        bar.classList.remove('hit-locked');
        bar.classList.remove('miss-locked');
        bar.style.width = '0%';
      }
    }
  }
}

function _startParlayLiveSse() {
  if (_parlayLiveES || typeof EventSource === 'undefined') return;
  // Immediately show live rows with a loading placeholder so the tracker
  // area is visible before the first SSE event arrives (1-3s ESPN fetch).
  // If we have a cached live snapshot from a previous session on this tab,
  // apply it immediately so the tracker is populated before the first SSE tick.
  if (PARLAY_LIVE_CACHE && PARLAY_LIVE_CACHE.legs && PARLAY_LIVE_CACHE.legs.length) {
    _applyParlayLiveUpdate(PARLAY_LIVE_CACHE);
  } else {
    // No cache — show "…" placeholder so the tracker area is visible while SSE connects.
    const _sseLegs = PARLAY_STATE && PARLAY_STATE.data && PARLAY_STATE.data.legs || [];
    for (let _si = 0; _si < _sseLegs.length; _si++) {
      const _sRow = _el('parlayLegLiveRow-' + _si);
      const _sStat = _el('parlayLegLiveStat-' + _si);
      const _sWrap = _el('parlayLegProgressWrap-' + _si);
      if (_sRow && !_sRow.textContent.trim()) {
        _sRow.style.display = '';
        if (_sStat) _sStat.textContent = '…';
      }
      if (_sWrap) _sWrap.style.display = '';
    }
  }
  try {
    _parlayLiveES = new EventSource('/api/parlay-live-stream');
    _parlayLiveES.onmessage = function(ev) {
      try {
        const d = JSON.parse(ev.data);
        if (d.legs && d.legs.length && !d.no_ticket) {
          PARLAY_LIVE_CACHE = d; // cache for instant restore on next tab return
          _applyParlayLiveUpdate(d);
        }
        if (d.all_games_final) {
          _stopParlayLiveSse();
          fetchParlayHistory(true).then(function() { fetchParlay(true); }).catch(function() { fetchParlay(true); });
        }
      } catch (err) { console.warn('[parlay-sse]', err); }
    };
    _parlayLiveES.onerror = function() {
      /* Browser may auto-reconnect; closing avoids duplicate streams when tab is idle */
    };
  } catch (e) {
    console.warn('[parlay-sse] init', e);
  }
}

function initParlayPage() {
  const todayET = _etToday();
  const sameDayLoaded = PARLAY_LOADED_DATE === todayET && PARLAY_STATE.status === 'success';
  const isStale = PARLAY_STATE.loadedAt ? (Date.now() - PARLAY_STATE.loadedAt) > 15 * 60 * 1000 : true;

  // Hydration check: if data freshly loaded (< 30s), skip fetch but still render
  const _parlayAge = PARLAY_STATE && PARLAY_STATE.loadedAt ? (Date.now() - PARLAY_STATE.loadedAt) : Infinity;
  const _isHydrated = _parlayAge < 30000; // < 30s: freshly hydrated

  PARLAY_LOADED_DATE = todayET;

  if (sameDayLoaded && !isStale) {
    // Same day, fresh data — don't fetch, but ensure ticket is rendered
    console.log('[parlay] data still fresh, skipping fetch');
    if (PARLAY_STATE.data) renderParlayTicket(PARLAY_STATE.data);
  } else if (_isHydrated) {
    // Freshly hydrated — skip fetch, render from hydrated data
    console.log('[parlay] using hydrated data, rendering');
    if (PARLAY_STATE.data) renderParlayTicket(PARLAY_STATE.data);
  } else {
    // Stale or missing — preserve existing ticket during refresh when available.
    fetchParlay(!!PARLAY_STATE.data);
  }

  // Always render history if data is available but DOM is empty;
  // otherwise fetch if not yet loaded for today
  const _histWrap = _el('parlayHistoryWrap');
  const _histDomEmpty = !_histWrap || _histWrap.style.display === 'none' || !_histWrap.innerHTML.trim();
  if (PARLAY_HIST_DATA && PARLAY_HIST_LOADED_DATE === todayET && _histDomEmpty) {
    // Data prewarmed but DOM never rendered — render now
    renderParlayHistory(PARLAY_HIST_DATA);
  } else if (!sameDayLoaded || PARLAY_HIST_LOADED_DATE !== todayET) {
    fetchParlayHistory(!sameDayLoaded || PARLAY_HIST_LOADED_DATE !== todayET);
  }

  if (PARLAY_STATE.status === 'success' && PARLAY_STATE.data && (PARLAY_STATE.data.legs || []).length) {
    _startParlayLiveSse();
  }

  if (PARLAY_STATE.status === 'success' && PARLAY_STATE.data) _labSyncParlayContext(PARLAY_STATE.data);
}

// ── Parlay History ──────────────────────────────────────────────────────

function _parlayHistoryRowSummary(legs) {
  if (!legs || !legs.length) return '';
  const leg = legs[0];
  const sl = {'points':'PTS','rebounds':'REB','assists':'AST'}[leg.stat_type] || (leg.stat_type || '').toString().toUpperCase();
  const dirCh = (leg.direction || 'o').charAt(0).toUpperCase();
  const first = _escapeHtml(leg.player_name || '') + ' ' + dirCh + ' ' + (leg.line != null ? leg.line : '') + ' ' + sl;
  if (legs.length === 1) return first;
  return first + ' +' + (legs.length - 1) + ' others';
}

function renderParlayHistoryError(msg) {
  const wrap = _el('parlayHistoryWrap');
  const statsEl = _el('parlayHistoryStats');
  const listEl = _el('parlayHistoryList');
  const msgEl = _el('parlayHistoryMessage');
  if (!wrap || !msgEl) return;
  if (statsEl) statsEl.innerHTML = '';
  if (listEl) listEl.innerHTML = '';
  const text = msg || 'Could not load parlay history.';
  msgEl.innerHTML = '<div class="empty-state" style="padding:14px;margin:0"><span class="icon" aria-hidden="true">📡</span><p style="margin:8px 0 0">' + _escapeHtml(text) + '</p>' +
    '<button type="button" class="secondary-btn" style="margin-top:12px;font-size:0.72rem" onclick="fetchParlayHistory(true)" aria-label="Retry loading parlay history">Retry</button></div>';
  msgEl.style.display = 'block';
  wrap.style.display = 'block';
}

async function fetchParlayHistory(nocache = false) {
  try {
    PARLAY_HIST_ERROR = null;
    const todayET = _etToday();
    const url = nocache ? '/api/parlay-history?nocache=1' : '/api/parlay-history';
    const r = await fetchWithTimeout(url, {}, 15000, _getTabSignal('parlay'));
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const data = await r.json();
    if (data.error) {
      PARLAY_HIST_ERROR = data.narrative || data.error;
      renderParlayHistoryError(PARLAY_HIST_ERROR);
      return;
    }
    PARLAY_HIST_DATA = data;
    PARLAY_HIST_LOADED_DATE = todayET;
    renderParlayHistory(data);
  } catch (err) {
    console.warn('[parlay] history fetch error:', err);
    PARLAY_HIST_ERROR = 'Could not load parlay history. Check your connection and try again.';
    renderParlayHistoryError(PARLAY_HIST_ERROR);
  }
}

function renderParlayHistory(hist) {
  const wrap = _el('parlayHistoryWrap');
  const msgEl = _el('parlayHistoryMessage');
  if (!wrap) return;
  PARLAY_HIST_ERROR = null;
  if (msgEl) {
    msgEl.innerHTML = '';
    msgEl.style.display = 'none';
  }

  const parlays = hist.parlays || [];
  const displayParlays = parlays.filter(function(p) {
    return (p.legs || []).length > 0 && (p.result === 'hit' || p.result === 'miss');
  });

  if (msgEl && !displayParlays.length) {
    msgEl.innerHTML = '<p style="margin:0;font-size:0.72rem;color:var(--muted);line-height:1.45">No concluded parlays yet. Finished tickets appear here after every leg is final. The Safest Parlay card above shows today\'s active ticket.</p>';
    msgEl.style.display = 'block';
  }

  const statsEl = _el('parlayHistoryStats');
  if (statsEl) {
    let statsHtml = '';
    if (displayParlays.length) {
      const hitRate = hist.hit_rate;
      const streak = hist.streak || 0;
      const streakType = hist.streak_type;
      if (hitRate !== null || streak) {
        const streakColor = streakType === 'hit' ? 'var(--color-success)' : 'var(--color-danger)';
        statsHtml = '<div style="display:flex;gap:12px;margin-bottom:16px">' +
          '<div style="background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius-card);padding:10px 16px;flex:1;text-align:center">' +
            '<div style="font-family:\'Barlow Condensed\',sans-serif;font-size:1.3rem;font-weight:900;color:var(--parlay)">' + (hitRate != null ? hitRate + '%' : '\u2014') + '</div>' +
            '<div style="font-size:0.62rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em">Hit Rate</div>' +
          '</div>' +
          (streak ? '<div style="background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius-card);padding:10px 16px;flex:1;text-align:center">' +
            '<div style="font-family:\'Barlow Condensed\',sans-serif;font-size:1.3rem;font-weight:900;color:' + streakColor + ';line-height:1">' + streak + '\u00d7 ' + streakType.toUpperCase() + '</div>' +
            '<div style="font-size:0.62rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em">Streak</div>' +
          '</div>' : '') +
        '</div>';
      }
    }
    statsEl.innerHTML = statsHtml;
  }

  const listEl = _el('parlayHistoryList');
  if (!listEl) return;
  if (!displayParlays.length) {
    listEl.innerHTML = '';
    wrap.style.display = 'block';
    return;
  }

  let rowsHtml = '';
  for (let i = 0; i < displayParlays.length; i++) {
    const p = displayParlays[i];
    const result = p.result === 'hit' ? 'hit' : 'miss';
    const legs = p.legs || [];
    const legSummary = _parlayHistoryRowSummary(legs);
    const dateLabel = p.date || '';
    const legsHit = legs.filter(function(l) { return l.result === 'hit'; }).length;
    const legsPill = ' <span style="font-size:0.6rem;color:var(--muted);margin-left:4px">' + legsHit + '/' + legs.length + '</span>';

    rowsHtml += '<div class="line-history-row" style="cursor:pointer;align-items:flex-start;padding-top:12px;padding-bottom:12px" data-parlay-idx="' + i + '" onclick="openParlayModalByIdx(this)" onkeydown="if(event.key===\'Enter\'||event.key===\' \'){event.preventDefault();this.click()}" role="button" tabindex="0" aria-label="View parlay detail for ' + _escapeHtml(dateLabel) + '">' +
      '<div style="display:flex;align-items:flex-start;gap:7px;min-width:0;flex:1">' +
        '<span class="line-hist-dir-pill" style="background:rgba(20,184,166,0.10);color:var(--parlay);border:1px solid rgba(20,184,166,0.20);font-size:0.55rem;flex-shrink:0;margin-top:2px">' + _escapeHtml(dateLabel.slice(5)) + '</span>' +
        '<div style="min-width:0;flex:1">' +
          '<div class="line-history-name" style="font-size:0.78rem;font-weight:600;color:var(--text);line-height:1.35;word-break:break-word">' + legSummary + '</div>' +
        '</div>' +
      '</div>' +
      '<div style="display:flex;align-items:center;flex-shrink:0;gap:4px">' +
        '<span class="line-result-pill ' + result + '">' + result.toUpperCase() + '</span>' +
        legsPill +
        '<span class="parlay-hist-chevron" aria-hidden="true">\u203a</span>' +
      '</div>' +
    '</div>';
  }
  listEl.innerHTML = rowsHtml;
  wrap.style.display = 'block';
}

// ── Parlay Modal ──────────────────────────────────────────────────────

function openParlayModal(parlay) {
  const modal = _el('parlayModal');
  const content = _el('parlayModalContent');
  if (!modal || !content) return;

  const legs = parlay.legs || [];
  const result = parlay.result || 'pending';
  const comboPct = parlay.combined_probability != null ? (parlay.combined_probability * 100).toFixed(1) : '\u2014';

  // Header
  let html = '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px">' +
    '<div style="font-family:\'Barlow Condensed\',sans-serif;font-weight:800;font-size:0.82rem;letter-spacing:0.06em;text-transform:uppercase;color:var(--parlay)">Parlay \u2014 ' + _escapeHtml(parlay.date || '') + '</div>' +
    '<span class="line-result-pill ' + result + '">' + result.toUpperCase() + '</span>' +
  '</div>';

  // Ticket with result overlays
  html += '<div style="border-radius:var(--radius-card);overflow:hidden;border:1px solid rgba(20,184,166,0.16);margin-bottom:12px">';
  for (let i = 0; i < legs.length; i++) {
    const leg = legs[i];
    html += renderParlayLeg(leg, i, legs.length, false);
    // Result overlay row for resolved legs
    if (leg.result && leg.result !== 'pending') {
      const actualVal = leg.actual_stat != null ? leg.actual_stat : '\u2014';
      const legResult = leg.result;
      html += '<div style="display:flex;align-items:center;justify-content:space-between;padding:6px 20px 8px;' +
        'background:' + (legResult === 'hit' ? 'rgba(var(--green-rgb),0.08)' : 'rgba(var(--red-rgb),0.06)') + ';border-bottom:1px solid rgba(255,255,255,0.04)">' +
        '<span style="font-size:0.68rem;color:var(--muted)">Actual: <strong style="color:var(--text)">' + _escapeHtml(String(actualVal)) + '</strong></span>' +
        '<span class="line-result-pill ' + legResult + '" style="font-size:0.58rem">' + legResult.toUpperCase() + '</span>' +
      '</div>';
    }
  }
  html += '</div>';

  // Stats row
  html += '<div style="display:flex;gap:12px;margin-bottom:12px">' +
    '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-card);padding:8px 14px;flex:1;text-align:center">' +
      '<div style="font-size:0.62rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:2px">Combined</div>' +
      '<div style="font-weight:800;font-size:0.82rem;color:var(--parlay)">' + comboPct + '%</div>' +
    '</div>' +
    '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-card);padding:8px 14px;flex:1;text-align:center">' +
      '<div style="font-size:0.62rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:2px">Legs Hit</div>' +
      '<div style="font-weight:800;font-size:0.82rem;color:var(--text)">' + legs.filter(function(l){ return l.result === 'hit'; }).length + '/' + legs.length + '</div>' +
    '</div>' +
  '</div>';

  // Narrative
  if (parlay.narrative) {
    html += '<div class="line-pick-conclusion-wrap"><p class="line-pick-conclusion">' + _escapeHtml(parlay.narrative) + '</p></div>';
  }

  content.innerHTML = html;
  modal.style.display = 'flex';
  document.body.style.overflow = 'hidden';
}

function openParlayModalByIdx(el) {
  const idx = parseInt(el.getAttribute('data-parlay-idx'), 10);
  if (PARLAY_HIST_DATA && PARLAY_HIST_DATA.parlays && PARLAY_HIST_DATA.parlays[idx]) {
    openParlayModal(PARLAY_HIST_DATA.parlays[idx]);
  }
}

function closeParlayModal() {
  const modal = _el('parlayModal');
  if (modal) modal.style.display = 'none';
  document.body.style.overflow = '';
}


// ════════════════════════════════════════════════════════
// LINE PAGE — grep: initLinePage, renderLinePickCard, switchLineDir, filterLineHistory, LINE_DIR
// ════════════════════════════════════════════════════════

let LINE_LOADED_DATE = '';
let LINE_DIR = 'under'; // active direction filter for line sub-nav
let LINE_OVER_PICK  = null; // today's over pick
let LINE_UNDER_PICK = null; // today's under pick
let LINE_RESOLVE_POLL = null; // setInterval id for pending pick polling
let LINE_LIVE_POLL = null;   // setInterval id for live in-game stat tracking
let _lineRotationTriggered = false; // gate: true once per-direction rotation re-fetch fired
let _lineRotationDoneFor = new Set(); // keys "over-Team-Opp" / "under-Team-Opp" we already ran rotation for (stops cycling)
let _lineLastLiveKey = '';   // last-rendered live snapshot key; avoid re-painting card when unchanged (stops flash)
let LINE_HIST_DATA = null;   // cached history payload; cleared on date change
let _lineLotdFetchPromise = null; // dedupe in-flight LOTD fetches
let _lineLotdLoadingTimer = null; // delayed skeleton timer (avoid flash on fast responses)
// Decoupled AsyncState for Line tab — grep: LINE_LOTD_STATE, LINE_HISTORY_STATE
let LINE_LOTD_STATE = asyncStateInitial();
let LINE_HISTORY_STATE = asyncStateInitial();

/** Classify fetch failure for logging. Returns 'timeout' | 'http_XXX' | 'parse' | 'network'. */
function classifyFetchError(err, response) {
  if (err && err.name === 'AbortError') return 'timeout';
  if (response && response.ok === false) return 'http_' + (response.status || 0);
  if (response && response.ok) return 'parse';
  return 'network';
}


/** Return true if the given pick's game is currently in its live window.
 *  Live window = game_start - 30min to game_start + 4.5h.
 *  Returns true when game_start_iso is absent (fail-open: poll if unknown). */
function _isPickGameLive(pick) {
  if (!pick || !pick.game_start_iso) return true; // unknown — allow poll
  try {
    const startMs = new Date(pick.game_start_iso).getTime();
    if (isNaN(startMs)) return true;
    const now = Date.now();
    return now >= startMs - 30 * 60 * 1000 && now <= startMs + 4.5 * 60 * 60 * 1000;
  } catch(e) { return true; }
}

/** Return ms until 30 min before the pick's game starts, or 0 if already in window. */
function _msUntilPickLive(pick) {
  if (!pick || !pick.game_start_iso) return 0;
  try {
    const startMs = new Date(pick.game_start_iso).getTime();
    if (isNaN(startMs)) return 0;
    const windowOpen = startMs - 30 * 60 * 1000;
    return Math.max(0, windowOpen - Date.now());
  } catch(e) { return 0; }
}

let _lineLiveScheduleTimer = null; // setTimeout id for pre-game schedule

async function _startLineLivePoll(overPick, underPick) {
  if (LINE_LIVE_POLL) { clearInterval(LINE_LIVE_POLL); LINE_LIVE_POLL = null; }
  if (_lineLiveScheduleTimer) { clearTimeout(_lineLiveScheduleTimer); _lineLiveScheduleTimer = null; }

  // Don't poll if both games are outside their live window — schedule a wakeup instead.
  const overLive  = _isPickGameLive(overPick);
  const underLive = _isPickGameLive(underPick);
  if (!overLive && !underLive) {
    const msOver  = overPick  ? _msUntilPickLive(overPick)  : Infinity;
    const msUnder = underPick ? _msUntilPickLive(underPick) : Infinity;
    const msWait  = Math.min(msOver, msUnder);
    if (msWait > 0 && isFinite(msWait)) {
      _lineLiveScheduleTimer = setTimeout(() => {
        _lineLiveScheduleTimer = null;
        _startLineLivePoll(LINE_OVER_PICK, LINE_UNDER_PICK);
      }, msWait);
    }
    return; // nothing to poll right now
  }

  let failureCount = 0;  // Track consecutive API failures
  const MAX_FAILURES = 5;  // Stop polling after 5 consecutive failures (150s tolerance for ESPN slowness)

  async function _fetchLive(pick) {
    if (!pick) return null;
    const p = new URLSearchParams({ team: pick.team || '', stat_type: pick.stat_type || 'points',
      player_name: pick.player_name || '' });
    if (pick.player_id) p.set('player_id', pick.player_id);
    try {
      const r = await fetchWithTimeout('/api/line-live-stat?' + p, {}, 20000);
      return r.ok ? r.json() : null;
    } catch(e) { return null; }
  }

  async function _poll() {
    if (!document.getElementById('tab-line')?.classList.contains('active')) return;

    // Deduplicate ESPN call if both picks are for the same team/game
    const sameGame = overPick && underPick && overPick.team === underPick.team;
    let overData, underData;
    if (sameGame) {
      overData = await _fetchLive(overPick);
      underData = overData;
    } else {
      [overData, underData] = await Promise.all([_fetchLive(overPick), _fetchLive(underPick)]);
    }

    // Track API failures: if both fetches fail and we can't detect game status, increment failure count
    const bothFailed = !overData && !underData;
    if (bothFailed) {
      failureCount++;
      // Stop polling if ESPN is down for 3+ consecutive polls; fall back to auto-resolve cron
      if (failureCount >= MAX_FAILURES) {
        console.warn('[line-live-poll] ESPN unavailable for 90+ sec; stopping polling (will resume via auto-resolve cron)');
        clearInterval(LINE_LIVE_POLL); LINE_LIVE_POLL = null;
        return;
      }
    } else {
      failureCount = 0;  // Reset on successful fetch
    }

    let anyLive = false;
    if (overData?.status === 'live' && LINE_OVER_PICK) {
      LINE_OVER_PICK = { ...LINE_OVER_PICK, _live: overData }; anyLive = true;
    }
    if (underData?.status === 'live' && LINE_UNDER_PICK) {
      LINE_UNDER_PICK = { ...LINE_UNDER_PICK, _live: underData }; anyLive = true;
    }

    if (anyLive) {
      const pick = LINE_DIR === 'over' ? LINE_OVER_PICK : LINE_UNDER_PICK;
      const card = document.getElementById('linePickCard');
      if (card && pick?._live) {
        const lv = pick._live;
        const newKey = JSON.stringify({ stat_current: lv.stat_current, clock: lv.clock, period: lv.period, pace: lv.pace });
        if (newKey !== _lineLastLiveKey) {
          _lineLastLiveKey = newKey;
          card.innerHTML = renderLinePickCard(pick, true);
        }
      }
    }

    // Per-direction game completion: when a direction's game finishes,
    // re-fetch line-of-the-day to trigger per-direction rotation.
    const overDone = !overPick || (overData && overData.status === 'final');
    const underDone = !underPick || (underData && underData.status === 'final');
    const overKey = overPick ? `over-${(overPick.team || '').trim()}-${(overPick.opponent || '').trim()}` : null;
    const underKey = underPick ? `under-${(underPick.team || '').trim()}-${(underPick.opponent || '').trim()}` : null;
    const alreadyDoneOver = overKey != null && _lineRotationDoneFor.has(overKey);
    const alreadyDoneUnder = underKey != null && _lineRotationDoneFor.has(underKey);
    const needRotation = (overDone && !alreadyDoneOver) || (underDone && !alreadyDoneUnder);
    if (needRotation) {
      if (overDone && overKey) _lineRotationDoneFor.add(overKey);
      if (underDone && underKey) _lineRotationDoneFor.add(underKey);
      _lineRotationTriggered = true;
      LINE_LOADED_DATE = _etToday();
      setTimeout(() => {
        // Keep last-good card visible; rotate in place once fresh payload arrives.
        fetchLineOfTheDay(true, true);   // nocache — backend will inline-resolve and rotate
        fetchLineHistory();        // reload history so resolved pick shows up there
      }, 1500);
    }
    if (overDone && underDone) {
      clearInterval(LINE_LIVE_POLL); LINE_LIVE_POLL = null;
      if (_lineLiveScheduleTimer) { clearTimeout(_lineLiveScheduleTimer); _lineLiveScheduleTimer = null; }
      // EVENT-BASED TRIGGER: Games finished — immediately check if slate is now unlocked
      if (document.getElementById('tab-lab')?.classList.contains('active')) {
        console.warn('[event] Games final detected — triggering immediate lab status check');
        setTimeout(() => {
          _fetchJson('/api/lab/status', 10000)
            .then(s => {
              if (!s.locked && LAB.locked) {
                console.warn('[event] Lab unlocked by game final event');
                showLabUnlocked(s);
              }
            })
            .catch(() => {});
        }, 500);
      }
    }
  }

  await _poll();
  if (LINE_LIVE_POLL === null) { // only start interval if game still live after first poll
    LINE_LIVE_POLL = setInterval(_poll, 60000);
  }
}

function _renderLineLOTDFromState() {
  const st = LINE_LOTD_STATE;
  const loading = st.status === 'initial' || st.status === 'loading';
  const lineEmptyEl = _el('lineEmpty');
  const linePickWrap = _el('linePickWrap');
  const linePickCard = _el('linePickCard');
  const lineSlateSummary = _el('lineSlateSummary');
  if (!linePickWrap || !linePickCard) return;

  if (loading) {
    linePickWrap.style.display = 'block';
    linePickCard.style.display = '';  // reset in case switchLineDir hid it
    linePickCard.innerHTML = renderLinePickCardSkeleton();
    lineSlateSummary.innerHTML = '';
    lineEmptyEl.style.display = 'none';
    return;
  }
  if (st.status === 'error') {
    linePickWrap.style.display = 'none';
    if (lineEmptyEl) {
      const icon = lineEmptyEl.querySelector('.icon');
      if (icon) icon.textContent = '📡';
      const msg = _el('lineEmptyMsg');
      if (msg) msg.textContent = 'Couldn\'t reach the server. Tap Retry.';
      lineEmptyEl.style.display = 'flex';
    }
    return;
  }
  if (st.status === 'success' && st.data) {
    const data = st.data;
    // Extract directional picks first — they may exist even when primary pick is null
    LINE_OVER_PICK  = data.over_pick  || (data.pick && data.pick.direction === 'over'  ? data.pick : null);
    LINE_UNDER_PICK = data.under_pick || (data.pick && data.pick.direction === 'under' ? data.pick : null);
    _lineLastLiveKey = ''; // reset live snapshot key for fresh card updates
    // If pick exists but no directional picks, populate from primary
    if (!LINE_OVER_PICK && !LINE_UNDER_PICK && data.pick) {
      if (data.pick.direction === 'over') LINE_OVER_PICK = data.pick;
      else LINE_UNDER_PICK = data.pick;
    }
    if (!data.pick && !LINE_OVER_PICK && !LINE_UNDER_PICK) {
      const msgs = {
        no_api_key:       'Odds API key not configured — contact admin.',
        no_games:         'No NBA games on the slate today.',
        no_edges:         "No strong edges found on today's props. Check back closer to tip-off.",
        odds_unavailable: 'Odds API unavailable. Try refreshing.',
        no_projections:   'Player projections are still generating. Please wait a moment and try again.',
        server_error:    'Line pick service temporarily unavailable. Tap Retry.',
      };
      const icons = { no_games: '🏀', no_edges: '📊', no_api_key: '🔑', default: '📡' };
      if (lineEmptyEl) {
        const icon = lineEmptyEl.querySelector('.icon');
        if (icon) icon.textContent = icons[data.error] || icons.default;
        const msg = _el('lineEmptyMsg');
        if (msg) msg.textContent = msgs[data.error] || 'No line pick available.';
        lineEmptyEl.style.display = 'flex';
      }
      linePickWrap.style.display = 'none';
      return;
    }
    linePickWrap.style.display = 'block';
    lineSlateSummary.innerHTML = data.slate_summary ? renderLineSummary(data.slate_summary) : '';
    // Stale-while-revalidate indicator: show subtle "Updating..." when backend is refreshing
    const _staleIndicator = document.getElementById('lineStaleIndicator');
    if (_staleIndicator) {
      _staleIndicator.style.display = (data.is_stale || data.refreshing) ? 'block' : 'none';
    }
    // Auto-correct direction: if the selected direction has no pick but the other does, switch
    if (!LINE_OVER_PICK && LINE_UNDER_PICK && LINE_DIR === 'over') LINE_DIR = 'under';
    else if (!LINE_UNDER_PICK && LINE_OVER_PICK && LINE_DIR === 'under') LINE_DIR = 'over';
    switchLineDir(LINE_DIR);
    lineEmptyEl.style.display = 'none';
    // Ensure history section is visible once picks load successfully
    let histWrap = document.getElementById('lineHistoryWrap');
    if (histWrap) histWrap.style.display = 'block';
  }
}

async function fetchLineOfTheDay(nocache = false, background = false) {
  if (_lineLotdFetchPromise) return _lineLotdFetchPromise;
  const _hasPriorData = LINE_LOTD_STATE && LINE_LOTD_STATE.status === 'success' && LINE_LOTD_STATE.data;
  if (!background) {
    LINE_LOTD_STATE = asyncStateLoading(LINE_LOTD_STATE);
    if (_lineLiveScheduleTimer) { clearTimeout(_lineLiveScheduleTimer); _lineLiveScheduleTimer = null; }
    if (_lineLotdLoadingTimer) { clearTimeout(_lineLotdLoadingTimer); _lineLotdLoadingTimer = null; }
    // Avoid loader flash on fast cache hits; only show skeleton if still loading after a short delay.
    if (!_hasPriorData) {
      _lineLotdLoadingTimer = setTimeout(() => {
        _lineLotdLoadingTimer = null;
        if (LINE_LOTD_STATE && LINE_LOTD_STATE.status === 'loading') _renderLineLOTDFromState();
      }, 700);
    }
  }

  let r;
  _lineLotdFetchPromise = (async function() {
    try {
      const _lotdUrl = nocache ? '/api/line-of-the-day?nocache=1' : '/api/line-of-the-day';
      r = await fetchWithTimeout(_lotdUrl, {}, 30000, _getTabSignal('line'));
      if (!r.ok) throw new Error('line-of-the-day error: ' + r.status);
      let data = await r.json();
      LINE_LOTD_STATE = asyncStateSuccess(LINE_LOTD_STATE, data);
      _renderLineLOTDFromState();
      _lineRotationTriggered = false; // allow next game completion to trigger rotation
      if (data.pick) {
        _startLineLivePoll(LINE_OVER_PICK, LINE_UNDER_PICK);
        if (!background) {
          // Save picks to GitHub as a safety net (backend auto-saves, but this catches failures).
          fetchWithTimeout('/api/save-line', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pick: data.pick, over_pick: data.over_pick, under_pick: data.under_pick }),
          }, 10000).then(function(r) {
            if (!r.ok) console.warn('[save-line] server error:', r.status);
          }).catch(function(e) { console.warn('[save-line] background save failed:', e); });
        }
        // Auto-resolve is cron-driven (0 * * * * — hourly). _startLineLivePoll detects
        // when a game finishes and re-fetches line-of-the-day for per-direction rotation.
      }
    } catch(e) {
      if (!background) {
        console.warn('[line-of-the-day] failed:', classifyFetchError(e, r), e && e.message ? e.message : '');
        if (!_hasPriorData) {
          LINE_LOTD_STATE = asyncStateError(LINE_LOTD_STATE, e);
          _renderLineLOTDFromState();
        } else {
          // Preserve last-good card if refresh fails.
          LINE_LOTD_STATE = asyncStateSuccess(LINE_LOTD_STATE, LINE_LOTD_STATE.data || {});
        }
      }
      _lineRotationTriggered = false; // allow next poll to retry rotation on error
    } finally {
      if (_lineLotdLoadingTimer) { clearTimeout(_lineLotdLoadingTimer); _lineLotdLoadingTimer = null; }
      _lineLotdFetchPromise = null;
    }
  })();
  return _lineLotdFetchPromise;
}

function retryLineOfTheDay() {
  fetchLineOfTheDay();
}


function renderNextSlatePending() {
  return '<div style="text-align:center;padding:24px 16px;color:var(--color-text-muted);font-size:0.82rem">'
    + '<div style="font-size:1.4rem;margin-bottom:8px">🏀</div>'
    + '<div style="margin-bottom:12px">Tomorrow\'s picks are on their way.</div>'
    + '<button type="button" class="secondary-btn" style="font-size:0.72rem" onclick="LINE_LOADED_DATE=\'\';fetchLineOfTheDay()">Check for picks</button>'
    + '</div>';
}

function _renderLineHistoryFromState() {
  const st = LINE_HISTORY_STATE;
  const wrap = document.getElementById('lineHistoryWrap');
  const list = document.getElementById('lineHistoryList');
  const stats = document.getElementById('lineHistoryStats');
  if (st.status === 'initial' || st.status === 'loading') {
    wrap.style.display = 'block';
    if (st.data && st.data.picks && st.data.picks.length) {
      // Stale data available — show it silently while refresh runs in background
      renderLineHistory(st.data);
      filterLineHistory(LINE_DIR);
    } else {
      stats.innerHTML = '';
      list.innerHTML = '<div class="skeleton" style="height:48px;margin-bottom:8px"><div class="skel-block skel-line wide"></div><div class="skel-block skel-line narrow"></div></div>'.repeat(3);
    }
    return;
  }
  if (st.status === 'error') {
    wrap.style.display = 'block';
    stats.innerHTML = '';
    list.innerHTML = '<div class="empty-state" style="padding:16px"><span class="icon">📡</span><p>Couldn\'t load recent picks.</p><button onclick="fetchLineHistory()" style="margin-top:8px;font-size:0.75rem;padding:6px 14px;border-radius:var(--radius-pill);background:var(--surface2);border:1px solid var(--border);color:var(--color-text-muted);cursor:pointer">Try again</button></div>';
    return;
  }
  if (st.status === 'success' && st.data && st.data.picks && st.data.picks.length) {
    renderLineHistory(st.data);
    filterLineHistory(LINE_DIR);
  }
}

async function fetchLineHistory() {
  LINE_HISTORY_STATE = asyncStateLoading(LINE_HISTORY_STATE);
  _renderLineHistoryFromState();
  try {
    const hr = await fetchWithTimeout('/api/line-history', {}, 25000, _getTabSignal('line'));
    if (!hr.ok) throw new Error('line-history HTTP ' + hr.status);
    const hist = await hr.json();
    LINE_HISTORY_STATE = asyncStateSuccess(LINE_HISTORY_STATE, hist);
    _renderLineHistoryFromState();
  } catch(e) {
    LINE_HISTORY_STATE = asyncStateError(LINE_HISTORY_STATE, e);
    _renderLineHistoryFromState();
  }
}

async function initLinePage() {
  const todayET = _etToday();
  const dateChanged = LINE_LOADED_DATE !== todayET;

  // Hydration check: if data freshly loaded (< 30s old), skip fetch entirely
  const _lotdAge = LINE_LOTD_STATE && LINE_LOTD_STATE.loadedAt ? (Date.now() - LINE_LOTD_STATE.loadedAt) : Infinity;
  const _isHydrated = _lotdAge < 30000; // < 30s: freshly hydrated

  if (!dateChanged && _isHydrated) {
    console.log('[line] using hydrated data, skipping fetch');
    _renderLineLOTDFromState();
    const d = LINE_LOTD_STATE && LINE_LOTD_STATE.data;
    if (d && d.pick) _startLineLivePoll(LINE_OVER_PICK, LINE_UNDER_PICK);
    if (!LINE_HIST_DATA) fetchLineHistory();
    return;
  }

  if (!dateChanged) {
    // Same day — stale refresh stays in background to avoid card flashing.
    const _isStale = _lotdAge > 15 * 60 * 1000; // 15 min staleness threshold
    if (_isStale) {
      fetchLineOfTheDay(true, true);
      fetchLineHistory();
      return;
    }
    // Fresh data — DOM preserved by tab toggle, no re-render needed.
    // Live poll restarted by switchTab() (line 1877). Only fetch history if never loaded.
    if (!LINE_HIST_DATA) fetchLineHistory();
    return;
  }
  // Date changed — full reload
  LINE_LOADED_DATE = todayET;
  _lineRotationDoneFor.clear(); // new day: allow rotation for new games
  document.getElementById('lineHistoryWrap').style.display = 'none';
  fetchLineOfTheDay();
  fetchLineHistory();
}

const LINE_STAT_LABEL = { points: 'PTS', rebounds: 'REB', assists: 'AST' };
function _lineStatLabel(statType) { return LINE_STAT_LABEL[statType] || (statType || 'pts').toUpperCase().slice(0, 3); }

function _escapeHtml(s) {
  return String(s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/** Skeleton HTML matching the Line pick card layout (header, play row, 5-col data row, conclusion). */
function renderLinePickCardSkeleton() {
  return `
    <div class="line-pick-card" style="pointer-events:none">
      <div class="line-pick-header">
        <div class="line-pick-header-left">
          <div class="skeleton" style="padding:0;margin:0;border:none;background:transparent;display:block">
            <div class="skel-block skel-line wide" style="height:18px;width:140px;margin-bottom:6px"></div>
            <div class="skel-block skel-line narrow" style="height:12px;width:180px"></div>
          </div>
        </div>
      </div>
      <div class="line-direction-row">
        <div class="skel-block" style="width:52px;height:24px;border-radius:var(--radius-pill)"></div>
        <div class="skel-block skel-line" style="height:14px;width:60px"></div>
      </div>
      <div class="line-pick-data-row">
        <div class="line-pick-data-col"><span class="line-pick-micro-lbl">Baseline</span><div class="skel-block skel-line" style="height:10px;width:100%"></div></div>
        <div class="line-pick-data-col"><span class="line-pick-micro-lbl">Edge</span><div class="skel-block skel-line" style="height:10px;width:100%"></div></div>
        <div class="line-pick-data-col"><span class="line-pick-micro-lbl">PTS</span><div class="skel-block skel-line" style="height:10px;width:100%"></div></div>
        <div class="line-pick-data-col"><span class="line-pick-micro-lbl">Min</span><div class="skel-block skel-line" style="height:10px;width:100%"></div></div>
      </div>
      <div class="line-pick-conclusion-wrap">
        <div class="skel-block skel-line wide" style="height:12px;margin-bottom:8px"></div>
        <div class="skel-block skel-line narrow" style="height:12px;width:80%"></div>
      </div>
    </div>`;
}

function renderLinePickCard(pick, isPrimary) {
  const dir = pick.direction || 'over';
  const statLabel = _lineStatLabel(pick.stat_type);
  const lineNum = parseFloat(pick.line) || 0;
  // Live stat tracking section
  let liveSection = '';
  if (pick._live?.status === 'live') {
    const lv = pick._live;
    const current = lv.stat_current ?? 0;
    const lineVal = parseFloat(pick.line) || 0;
    const isOver = dir === 'over';
    const statLabelLive = (pick.stat_type || 'pts').toLowerCase();
    const pct = Math.min(100, lineVal > 0 ? Math.round(current / lineVal * 100) : 0);
    const winning = isOver
      ? (lv.pace != null ? lv.pace >= lineVal : current >= lineVal)
      : (lv.pace != null ? lv.pace <= lineVal : current < lineVal);
    const barColor = winning ? (isOver ? 'var(--line)' : 'var(--lab)') : 'var(--color-danger)';
    const periodStr = lv.period <= 4 ? `Q${lv.period}` : `OT${lv.period - 4}`;
    liveSection = `
      <div class="line-live-section">
        <div class="line-live-header">
          <span class="line-live-badge">● LIVE ${periodStr} · ${lv.clock}</span>
          ${lv.pace != null ? `<span class="line-live-pace">Pace: ~${lv.pace} ${statLabelLive}</span>` : ''}
        </div>
        <div class="line-live-progress-wrap">
          <div class="line-live-progress-bar" style="width:${pct}%;background:${barColor}"></div>
        </div>
        <div class="line-live-labels">
          <span class="line-live-current">${current} ${statLabelLive}</span>
          <span>Line: ${pick.line}</span>
        </div>
      </div>`;
  }

  const oddsHtml = pick.line_updated_at
    ? (() => {
        const raw = pick.line_updated_at;
        const utcStr = raw.endsWith('Z') ? raw : raw.replace(/([+-]\d{2}:\d{2})?$/, 'Z');
        const t = new Date(utcStr).toLocaleTimeString('en-US', {
          hour: 'numeric', minute: '2-digit', timeZone: 'America/Chicago'
        });
        return `<div class="line-pick-odds-meta">Odds · ${t} CT</div>`;
      })()
    : (() => {
        const oddsVal = pick.direction === 'over' ? pick.odds_over : pick.odds_under;
        return (pick.books_consensus != null && pick.books_consensus > 0 && oddsVal != null)
          ? `<div class="line-pick-odds-meta line-pick-odds-num">${oddsVal}</div><div class="line-pick-odds-meta">${pick.books_consensus} book${pick.books_consensus !== 1 ? 's' : ''}</div>`
          : '<div class="line-pick-odds-meta">MODEL</div>';
      })();

  const subheader = pick.game_time ? `${_escapeHtml(pick.team)} vs ${_escapeHtml(pick.opponent)} · ${_escapeHtml(pick.game_time)}` : `${_escapeHtml(pick.team)} · vs ${_escapeHtml(pick.opponent)}`;
  const edgeClass = (pick.edge ?? 0) > 0 ? 'edge-plus' : (pick.edge ?? 0) < 0 ? 'edge-minus' : '';
  const projStat = pick.projection != null && pick.projection !== '' ? Number(pick.projection).toFixed(1) : '—';
  const seasonStat = pick.season_avg != null && pick.season_avg !== '' ? (typeof pick.season_avg === 'number' ? pick.season_avg.toFixed(1) : String(pick.season_avg)) : '—';
  const projMin = pick.proj_min != null && pick.proj_min !== '' ? (typeof pick.proj_min === 'number' ? pick.proj_min.toFixed(1) : String(pick.proj_min)) : '—';
  const avgMin = pick.avg_min != null && pick.avg_min !== '' ? (typeof pick.avg_min === 'number' ? pick.avg_min.toFixed(1) : String(pick.avg_min)) : '—';
  const tcolor = TEAM_COLORS[pick.team] || '#d4a640';
  const tcolorBorder = _hexToRgba(tcolor, 0.30);
  const tcolorGlow   = _hexToRgba(tcolor, 0.22);

  return `
    <div class="line-pick-card" style="--tcolor:${tcolor};--tcolor-border:${tcolorBorder};--tcolor-glow:${tcolorGlow}">
      <div class="line-pick-header">
        <div class="line-pick-header-left">
          <div class="line-pick-name">${_escapeHtml(pick.player_name)}</div>
          <div class="line-pick-team">${subheader}</div>
        </div>
        ${oddsHtml ? `<div class="line-pick-header-meta">${oddsHtml}</div>` : ''}
      </div>
      <div class="line-direction-row">
        <span class="line-direction-pill ${dir}">${dir.toUpperCase()}</span>
        <div class="line-stat-val">${pick.line ?? '—'} ${statLabel.toLowerCase()}</div>
      </div>
      <div class="line-pick-data-row">
        <div class="line-pick-data-col">
          <span class="line-pick-micro-lbl">Baseline</span>
          <span class="line-pick-micro-val">${pick.line ?? '—'} ${statLabel}</span>
        </div>
        <div class="line-pick-data-col">
          <span class="line-pick-micro-lbl">Edge</span>
          <span class="line-pick-micro-val ${edgeClass}">${(pick.edge ?? 0) > 0 ? '+' : ''}${pick.edge ?? 0}</span>
        </div>
        <div class="line-pick-data-col">
          <span class="line-pick-micro-lbl">${statLabel}</span>
          <span class="line-pick-stat-val">${projStat}</span>
          <span class="line-pick-stat-avg">${seasonStat}</span>
        </div>
        <div class="line-pick-data-col">
          <span class="line-pick-micro-lbl">Min</span>
          <span class="line-pick-stat-val">${projMin}</span>
          <span class="line-pick-stat-avg">${avgMin}</span>
        </div>
      </div>
      ${liveSection}
      ${pick.result && pick.result !== 'pending' ? `
      <div class="line-result-row">
        <span class="line-result-lbl">Result</span>
        <span class="line-result-pill ${pick.result}">${pick.result.toUpperCase()}${pick.actual_stat != null ? ` · ${pick.actual_stat} actual` : ''}</span>
      </div>` : ''}
    </div>`;
}

function renderLineSummary(summary) {
  let timeStr = '';
  if (summary.timestamp) {
    const ts = new Date(summary.timestamp);
    timeStr = `<div class="line-summary-chip">${ts.toLocaleTimeString('en-US',{hour:'numeric',minute:'2-digit'})}</div>`;
  }
  return `
    <div class="line-summary-row">
      <div class="line-summary-chip">${summary.games_evaluated} games</div>
      <div class="line-summary-chip">${summary.props_scanned} props scanned</div>
      <div class="line-summary-chip">${summary.edges_found} edges found</div>
      ${timeStr}
    </div>`;
}

function renderLineHistory(hist) {
  LINE_HIST_DATA = hist;
  const streakNum = hist.streak || 0;
  const streakLabel = hist.streak_type || '';

  let statsHtml = '';
  if (hist.hit_rate !== null || streakNum) {
    const hitRate = hist.hit_rate !== null ? `${hist.hit_rate}%` : '—';
    const streakColor = streakLabel === 'hit' ? 'var(--green)' : 'var(--red)';
    statsHtml = `
    <div style="display:flex;gap:16px;margin-bottom:14px">
      <div style="background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius-card);padding:10px 16px;flex:1;text-align:center">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.3rem;font-weight:900;color:var(--line)">${hitRate}</div>
        <div style="font-size:0.62rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em">Hit Rate</div>
      </div>
      ${streakNum ? `<div style="background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius-card);padding:10px 16px;flex:1;text-align:center;display:flex;flex-direction:column;align-items:center;justify-content:center">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.3rem;font-weight:900;color:${streakColor};line-height:1">${streakNum}× ${streakLabel.toUpperCase()}</div>
        <div style="font-size:0.62rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;margin-top:3px">Streak</div>
      </div>` : ''}
    </div>`;
  }

  document.getElementById('lineHistoryStats').innerHTML = statsHtml;
  document.getElementById('lineHistoryList').innerHTML = renderLineHistoryRows(hist.picks, 'all');
  // Init history filter pill position after render
  setTimeout(() => {
    const histWrap = document.getElementById('lineHistWrap');
    if (histWrap) {
      const activeBtn = histWrap.querySelector('.line-hist-tab.active');
      moveTogglePill(histWrap, activeBtn, '255,255,255');
    }
  }, 30);
}

function renderLineHistoryRows(picks, dirFilter) {
  let html = '';
  const filtered = dirFilter === 'all' ? picks : picks.filter(p => (p.direction || 'over') === dirFilter);
  const _picks = [];
  filtered.forEach(p => {
    const result = p.result || 'pending';
    const dir    = p.direction || 'over';
    const idx    = _picks.push(p) - 1;
    html += `
      <div class="line-history-row" style="cursor:pointer" data-pick-idx="${idx}" onclick="openLinePickModalByIdx(this)" onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();this.click()}" role="button" tabindex="0">
        <div style="display:flex;align-items:center;gap:7px;min-width:0">
          <span class="line-hist-dir-pill ${dir}">${dir.toUpperCase()}</span>
          <div style="min-width:0">
            <div class="line-history-name">${_escapeHtml(p.player_name)}</div>
            <div class="line-history-sub">${_escapeHtml(p.line)} ${_escapeHtml(p.stat_type||'pts')} · ${_escapeHtml(p.date)}</div>
          </div>
        </div>
        <span class="line-result-pill ${result}" style="flex-shrink:0">${result}</span>
      </div>`;
  });
  // Attach picks to rendered nodes after inserting into DOM (deferred via timeout)
  setTimeout(() => {
    document.querySelectorAll('.line-history-row[data-pick-idx]').forEach(el => {
      const _pidx = parseInt(el.dataset.pickIdx);
      if (!isNaN(_pidx) && _pidx < _picks.length) el._linePick = _picks[_pidx];
    });
  }, 0);
  if (!filtered.length) {
    html = `<div style="font-size:0.75rem;color:var(--muted);text-align:center;padding:16px 0">No ${dirFilter === 'all' ? '' : dirFilter + ' '}picks yet</div>`;
  }
  return html;
}

function openLinePickModal(pick) {
  if (!pick) return;
  document.getElementById('linePickModalContent').innerHTML = renderLinePickCard(pick, false);
  const modal = document.getElementById('linePickModal');
  modal.style.display = 'flex';
  document.body.style.overflow = 'hidden';
}

function openLinePickModalByIdx(el) {
  openLinePickModal(el._linePick);
}

function closeLinePickModal() {
  const modal = document.getElementById('linePickModal');
  if (modal) modal.style.display = 'none';
  document.body.style.overflow = '';
}



document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') { document.body.style.overflow = ''; }
});

function filterLineHistory(dir) {
  if (!LINE_HIST_DATA) return;
  // Update active tab styling on the inline All/Over/Under tabs + slide pill
  const histWrap = document.getElementById('lineHistWrap');
  let activeBtn = null;
  document.querySelectorAll('.line-hist-tab').forEach(btn => {
    const isActive = btn.dataset.dir === dir;
    btn.classList.toggle('active', isActive);
    btn.classList.remove('over', 'under');
    if (isActive && dir !== 'all') btn.classList.add(dir);
    if (isActive) activeBtn = btn;
  });
  const rgb = dir === 'over' ? '212,166,64' : dir === 'under' ? '20,184,166' : '255,255,255';
  moveTogglePill(histWrap, activeBtn, rgb);
  document.getElementById('lineHistoryList').innerHTML = renderLineHistoryRows(LINE_HIST_DATA?.picks, dir);

  // Recompute stats for the filtered subset — use same (direction || 'over') default
  // as renderLineHistoryRows so stats always match the rendered pick count.
  const filtered = dir === 'all' ? (LINE_HIST_DATA?.picks || [])
    : (LINE_HIST_DATA?.picks || []).filter(p => (p.direction || 'over') === dir);
  const hits   = filtered.filter(p => p.result === 'hit');
  const misses = filtered.filter(p => p.result === 'miss');
  const total  = hits.length + misses.length;
  const hitRate = total ? Math.round(hits.length / total * 100) : null;
  let streak = 0, streakType = null;
  for (const p of filtered) {
    if (!p.result || p.result === 'pending') continue;
    if (!streakType) { streakType = p.result; streak = 1; }
    else if (p.result === streakType) streak++;
    else break;
  }
  let statsHtml = '';
  if (hitRate !== null || streak) {
    const streakColor = streakType === 'hit' ? 'var(--green)' : 'var(--red)';
    statsHtml = `
    <div style="display:flex;gap:16px;margin-bottom:14px">
      <div style="background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius-card);padding:10px 16px;flex:1;text-align:center">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.3rem;font-weight:900;color:var(--line)">${hitRate !== null ? hitRate + '%' : '—'}</div>
        <div style="font-size:0.62rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em">Hit Rate</div>
      </div>
      ${streak ? `<div style="background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius-card);padding:10px 16px;flex:1;text-align:center;display:flex;flex-direction:column;align-items:center;justify-content:center">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.3rem;font-weight:900;color:${streakColor};line-height:1">${streak}× ${(streakType||'').toUpperCase()}</div>
        <div style="font-size:0.62rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;margin-top:3px">Streak</div>
      </div>` : ''}
    </div>`;
  }
  document.getElementById('lineHistoryStats').innerHTML = statsHtml;
}

function switchLineDir(dir) {
  LINE_DIR = dir;
  // Update sub-nav pill buttons + slide pill
  const lineNav = document.getElementById('lineSubNav');
  let activeBtn = null;
  lineNav.querySelectorAll('.line-dir-btn').forEach(btn => {
    const isActive = btn.dataset.dir === dir;
    btn.classList.toggle('active', isActive);
    if (isActive) activeBtn = btn;
  });
  const rgb = dir === 'over' ? '212,166,64' : '20,184,166';
  moveTogglePill(lineNav, activeBtn, rgb);
  // Render the pick for the selected direction
  const pickCard = document.getElementById('linePickCard');
  const noMsg    = document.getElementById('lineNoPickMsg');
  if (pickCard) {
    const pick = dir === 'over' ? LINE_OVER_PICK : LINE_UNDER_PICK;
    if (pick) {
      pickCard.style.display = '';
      pickCard.innerHTML = renderLinePickCard(pick, true);
      if (noMsg) noMsg.style.display = 'none';
    } else if (!LINE_OVER_PICK && !LINE_UNDER_PICK) {
      // Both null — if still loading, keep skeleton visible; only hide when confirmed no picks
      if (LINE_LOTD_STATE.status === 'loading' || LINE_LOTD_STATE.status === 'initial') {
        pickCard.style.display = '';
      } else {
        pickCard.style.display = 'none';
      }
      if (noMsg) noMsg.style.display = 'none';
    } else {
      // One direction genuinely missing — show "No X pick today" only for that direction
      pickCard.style.display = 'none';
      if (noMsg) { noMsg.style.display = 'block'; noMsg.textContent = `No ${dir.toUpperCase()} pick today.`; }
    }
  }
  // Sync the inline history tabs and re-render history
  filterLineHistory(dir);
}

// ════════════════════════════════════════════════════════
// LAB PAGE — grep: initLabPage, LAB state, labCallClaude, buildLabSystemPrompt
// ════════════════════════════════════════════════════════

const LAB = {
  messages: [],
  system: '',
  briefing: null,
  config: null,
  lineData: null,
  slateData: null,
  parlayData: null,
  initialized: false,
  initAt: 0,
  initDate: '',       // ET date string when Lab was opened (clears messages on rollover)
  _lockPoll: null,
  pendingImage: null,
  cachedStatus: null,   // JSON from /api/lab/status — avoid refetch on quick tab revisit
  statusFetchedAt: 0,   // timestamp; invalidate on Retry or after 10 min or when loadSlate completes
  locked: false,        // mirrors last known lock state; used by line poll event-trigger
};

function _benPersistChatMirror() {
  try {
    const today = _etToday();
    const rows = LAB.messages.filter(m => !m.hidden).map(m => ({ role: m.role, content: m.content }));
    localStorage.setItem('ben_chat_mirror_' + today, JSON.stringify(rows));
  } catch (e) {}
}

function _isLabStatusErrorFallback(status) {
  // Backend returns locked=true with this reason when it hits an exception (ESPN timeout, etc.).
  // We treat that as "status unknown" and never show "BEN IS LOCKED" for it.
  let r = (status.reason || '').toLowerCase();
  return r.indexOf('try again') !== -1 || r.indexOf('temporarily unavailable') !== -1;
}

function _resetLabAndRetry() {
  LAB.cachedStatus = null; LAB.statusFetchedAt = 0; LAB.initialized = false; initLabPage();
}

function _showLabStatusUnknown() {
  const labLockedEl = _el('labLocked');
  const labConnErr = _el('labConnectionError');
  const labConnReason = _el('labConnectionErrorReason');
  const labConnAction = _el('labConnectionErrorAction');
  if (labLockedEl) labLockedEl.style.display = 'none';
  if (labConnErr) labConnErr.style.display = 'block';
  if (labConnReason) labConnReason.textContent = 'Server couldn\'t determine status. Tap Retry.';
  if (labConnAction) labConnAction.innerHTML = '<button type="button" class="secondary-btn" style="margin-top:10px;font-size:0.72rem" onclick="_resetLabAndRetry()">Retry</button>';
}

async function initLabPage() {
  // Re-init Lab if tab revisited after 10+ minutes (refresh context, keep chat open)
  const stale = Date.now() - LAB.initAt > 10 * 60 * 1000;
  if (LAB.initialized && !stale) return;
  if (!_el('labUnlocked')) return; // DOM not ready (e.g. wrong tab markup)
  LAB.initialized = true;
  LAB.initAt = Date.now();
  // Only clear messages when the date has rolled over — not just because we've been
  // away 10+ min. Conversations persist until page refresh or a new day.
  const today = _etToday();
  if (LAB.initDate && LAB.initDate !== today) {
    LAB.messages = [];
    const labMsg = _el('labMessages');
    if (labMsg) {
      // Remove only rendered chat bubbles (keep empty state + anchor).
      labMsg.querySelectorAll('.lab-msg').forEach(el => el.remove());
      labMsg.classList.remove('lab-has-messages');
      // If an older code path removed the anchor/empty-state markup, restore it.
      if (!document.getElementById('labEmptyState') || !document.getElementById('labScrollAnchor')) {
        labMsg.innerHTML =
          '<div id="labEmptyState"><div class="les-icon">🔮</div><div class="les-title">Ask Ben anything</div><div class="les-sub">Today\'s slate is loaded. Ask about players, projections, card boosts, or trends.</div></div>' +
          '<div id="labScrollAnchor" aria-hidden="true" style="height:0;flex-shrink:0;overflow:hidden"></div>';
      }
    }
  }
  LAB.initDate = today;

  // Ben is always available — skip lock status and show unlocked view directly.
  // We still refresh context on stale init to pick up new slates.
  [_el('labLoading'), _el('labLocked'), _el('labConnectionError')].forEach(e => { if (e) e && (e.style.display = 'none'); });
  const labUnlockedEl = _el('labUnlocked');
  if (labUnlockedEl) labUnlockedEl.style.display = 'flex';
  await showLabUnlocked({ locked: false, games_final: 0 });

  // Load persisted daily Ben chat history (server + localStorage mirror if API fails or returns partial data).
  try {
    const r = await fetchWithTimeout('/api/lab/chat-history', {}, 10000, _getTabSignal('lab'));
    let history = [];
    if (r.ok) {
      const j = await r.json();
      if (Array.isArray(j)) history = j;
    }
    // After await: messages sent while fetch was in flight; mirror supplements gaps/offline.
    const _localSnap = LAB.messages.filter(m => !m.hidden).map(m => ({ role: m.role, content: m.content }));
    const _mirror = _safeParseLocalStorage('ben_chat_mirror_' + today, []);
    const labMsg = _el('labMessages');
    if (labMsg) {
      labMsg.querySelectorAll('.lab-msg').forEach(el => el.remove());
      labMsg.classList.remove('lab-has-messages');
    }
    LAB.messages = [];
    const _histKey = (role, content) => role + ':' + String(content);
    const _histSet = new Set();
    const _replayOne = (m) => {
      if (!m || !m.role || m.content == null) return;
      const k = _histKey(m.role, m.content);
      if (_histSet.has(k)) return;
      _histSet.add(k);
      appendLabMessage(m.role, m.content, false);
    };
    history.forEach(_replayOne);
    if (Array.isArray(_mirror)) {
      for (const msg of _mirror) {
        if (!msg || !msg.role || msg.content == null) continue;
        const k = _histKey(msg.role, msg.content);
        if (!_histSet.has(k)) {
          _histSet.add(k);
          appendLabMessage(msg.role, msg.content, false);
        }
      }
    }
    for (const msg of _localSnap) {
      const k = _histKey(msg.role, msg.content);
      if (!_histSet.has(k)) {
        _histSet.add(k);
        appendLabMessage(msg.role, msg.content, false);
      }
    }
    _benPersistChatMirror();
  } catch(e) {
    console.warn('[lab] chat-history load failed:', e && e.message ? e.message : e);
  }
}

function _updateLabLockedStatus(status) {
  document.getElementById('labLockedReason').textContent = status.reason || 'Lab locked';
  let el = document.getElementById('labLockedUnlock');
  if (status.reason && status.reason.indexOf('try again') !== -1) {
    el.innerHTML = '<button type="button" class="secondary-btn" style="margin-top:10px;font-size:0.72rem" onclick="_resetLabAndRetry()">Retry</button>';
    return;
  }
  let tzOpts = {hour:'numeric',minute:'2-digit',timeZone:'America/Chicago'};
  const estUnlock = status.estimated_unlock
    ? 'Estimated unlock: ' + new Date(status.estimated_unlock).toLocaleTimeString('en-US',tzOpts)
    : (status.next_lock_time ? 'Locks at ' + new Date(status.next_lock_time).toLocaleTimeString('en-US',tzOpts) : '');
  el.textContent = estUnlock;
}

async function showLabLocked(status) {
  LAB.locked = true;
  document.getElementById('labConnectionError').style.display = 'none';
  document.getElementById('labLocked').style.display = 'block';
  _updateLabLockedStatus(status);

  // Poll every 3 min while locked: update games remaining + unlock estimate,
  // and automatically flip to unlocked view when all games finish.
  if (LAB._lockPoll) clearInterval(LAB._lockPoll);
  LAB._lockPoll = setInterval(function() {
    const _labLocked = _el('labLocked');
    if (!_labLocked || _labLocked.style.display === 'none') {
      clearInterval(LAB._lockPoll); LAB._lockPoll = null; return;
    }
    _fetchJson('/api/lab/status', 10000).then(async function(s) {
      let labLockedEl = _el('labLocked');
      let labUnlockedEl = _el('labUnlocked');
      let lockedVisible = labLockedEl ? labLockedEl.style.display !== 'none' : false;
      let unlockedVisible = labUnlockedEl ? labUnlockedEl.style.display !== 'none' : false;
      if (!s.locked && lockedVisible) {
        // Locked → unlocked transition (break between games or all final)
        LAB.initialized = false;
        if (labLockedEl) labLockedEl.style.display = 'none';
        let labLoad = _el('labLoading'); if (labLoad) labLoad.style.display = 'block';
        await showLabUnlocked(s);
        if (labLoad) labLoad.style.display = 'none';
        // Keep polling if more games are coming (next_lock_time set)
        if (!s.next_lock_time) { clearInterval(LAB._lockPoll); LAB._lockPoll = null; }
      } else if (s.locked && unlockedVisible) {
        // Break ended → re-locked: transition back to locked view
        if (labUnlockedEl) labUnlockedEl.style.display = 'none';
        await showLabLocked(s);  // Re-shows locked view + starts its own poll
      } else if (s.locked) {
        _updateLabLockedStatus(s);
      }
    }).catch(function(e) {
      console.warn('[lab] Lock poll status check failed:', e && e.message ? e.message : e);
    });
  }, 120000);  // 2 minutes: reduce Railway API calls while keeping unlock detection
}

async function showLabUnlocked(status) {
  LAB.locked = false;
  // Explicitly stop lock polling to prevent memory leak on rapid flickers
  if (LAB._lockPoll) { clearInterval(LAB._lockPoll); LAB._lockPoll = null; }

  [_el('labConnectionError'), _el('labLocked'), _el('labLoading')].forEach(e => { if (e) e.style.display = 'none'; });
  const labUnlockedShow = _el('labUnlocked');
  if (labUnlockedShow) labUnlockedShow.style.display = 'flex';

  // Refresh Predict tab immediately when games finish — next-day slate appears right away
  if (SLATE && SLATE.all_complete) loadSlate().catch(() => {});

  // Show chat UI immediately with whatever context is available
  if (!LAB.system) LAB.system = buildLabSystemPrompt();

  // Load context data in background (non-blocking) — chat is usable while this runs
  let _cachedSlate = SLATE
    ? Promise.resolve(SLATE) : _fetchJson('/api/slate', 10000);
  Promise.allSettled([
    _fetchJson('/api/lab/briefing', 30000),
    _fetchJson('/api/lab/config-history', 10000),
    _cachedSlate,
  ]).then(function([briefingRes, configRes, slateRes]) {
    if (briefingRes.status === 'fulfilled') LAB.briefing  = briefingRes.value;
    if (configRes.status === 'fulfilled')   LAB.config    = configRes.value;
    if (slateRes.status === 'fulfilled')    LAB.slateData = slateRes.value;
    // Rebuild system prompt with full context
    LAB.system = buildLabSystemPrompt();
  });
}

function buildLabSystemPrompt() {
  const briefStr = LAB.briefing ? JSON.stringify(LAB.briefing, null, 2) : 'No briefing available yet.';
  const cfgStr   = LAB.config   ? JSON.stringify(LAB.config.config || LAB.config, null, 2) : 'Config unavailable.';
  const slateStr = (() => {
    const s = LAB.slateData;
    if (!s || !s.lineups) return 'Slate not loaded yet.';
    const chalk  = s.lineups.chalk  || [];
    const upside = s.lineups.upside || [];
    const games  = (s.games || [])
      .filter(g => !g.completed)
      .map(g => `  ${g.away?.abbr || '?'} @ ${g.home?.abbr || '?'}${g.spread != null ? ` (spread ${g.spread > 0 ? '+' : ''}${g.spread}, O/U ${g.total})` : ''}`)
      .join('\n');

    const fmtPlayer = (p, i) => {
      const stats = [
        p.pts  != null ? `${parseFloat(p.pts).toFixed(1)}pts` : null,
        p.reb  != null ? `${parseFloat(p.reb).toFixed(1)}reb` : null,
        p.ast  != null ? `${parseFloat(p.ast).toFixed(1)}ast` : null,
      ].filter(Boolean).join('/');
      const boost = p.est_mult != null ? ` | boost ${parseFloat(p.est_mult).toFixed(2)}x` : '';
      const mins  = p.predMin  != null ? ` | ${parseFloat(p.predMin).toFixed(0)}min` : '';
      return `  ${i+1}. ${p.name} (${p.team}, ${p.pos}) — RS: ${parseFloat(p.rating||0).toFixed(1)}${stats ? `, ${stats}` : ''}${mins}${boost}`;
    };

    return `Date: ${s.date}
Games:
${games || '  (none)'}

CHALK LINEUP (Starting 5 — highest projected value):
${chalk.map(fmtPlayer).join('\n') || '  (none)'}

MOONSHOT LINEUP (High-upside plays):
${upside.map(fmtPlayer).join('\n') || '  (none)'}`;
  })();

  return `You are Ben — the Oracle's autonomous NBA analytics and engineering assistant. Full authority: data, config, and code.

APP STRUCTURE (2 tabs):
- PREDICT: Daily lineup optimizer. Slate-Wide: Starting 5 = chalk (conservative, high-RS + moderate boost, 20-min avg/recent floor). Moonshot = high-boost upside plays (low ownership, 17+ min projected floor, RotoWire cleared). Per-Game: single "THE LINE UP" format (5 players, 20-min projected floor, no card boost — same 2-team pool). Slate locks 5 min before first tip.
- BEN (here): Chat with model context from Predict when loaded.

DATA PIPELINE (this season):
- Historical outcomes for Log/audit: data/top_performers.csv (mega rollup by date) is primary; legacy data/actuals/{date}.csv still supported as fallback.
- Developer-only ingestion: curl/scripts → /api/parse-screenshot + POST /api/save-most-popular, /api/save-most-drafted-3x, /api/save-winning-drafts (see docs/HISTORICAL_DATA.md). Optional INGEST_SECRET on those writers.
- Most popular lists → data/most_popular/{date}.csv (POST /api/save-ownership is an alias of the same path).
- Predictions saved at lock → data/predictions/{date}.csv
- Audit JSON → data/audit/{date}.json (from predictions vs top_performers / actuals)
- Card boost calibration: GET /api/lab/calibrate-boost → fits log_a/log_b from ownership data → propose via update_config action${LAB.briefing?.ownership_calibration_available ? `\n- OWNERSHIP DATA AVAILABLE for dates: ${(LAB.briefing.ownership_dates || []).join(', ')}. Run GET /api/lab/calibrate-boost to fit the boost formula to real data.` : ''}

VISION: You have full image analysis capability. When the user attaches a photo using the 📷 button (left of the input box), you will see the image directly and can read every number, name, and stat visible. Use this for ownership screens, leaderboards, RS results, draft compositions, anything. If asked whether you can process images, say YES and tell them to use the camera button.

CAPABILITIES:
- Analyze any attached screenshot (use read from image content sent with the message)
- Answer questions about today's slate, games, and picks
- Read any file: use read_repo_file (CSVs, audit JSONs, code, config)
- Browse directories: use list_repo_directory to discover available dates/files
- Config changes: <action>update_config</action> for parameter tuning (shows confirm UI), or write_repo_file for data/model-config.json for structural additions
- Validate params: backtest action before applying any config change

WHEN TO USE WHAT:
- Parameter tuning (scalars, weights, thresholds) → <action>backtest</action> first, then <action>update_config</action>
- Algorithmic changes (formulas, logic, filters) → read_repo_file the function → describe the exact change needed → tell the user to implement it via a developer push (you cannot write code files)
- Investigating accuracy → read_repo_file data/audit/{date}.json; actual RS labels: data/top_performers.csv (filter by date) or legacy data/actuals/{date}.csv
- Historical pick analysis → list_repo_directory data/predictions, then read specific files
- Full player projections (card boosts, RS ratings for ALL players, not just the top 10 in context) → read_repo_file data/predictions/{date}.csv (available after slate locks; has all projected players with RS, boost, minutes, pts/reb/ast). Pre-lock: slate context above shows top 10 only — tell user projections save at lock.
- Card boost for a specific player not in the lineups above → read predictions CSV or note the player is outside the model's top picks for today

ACTIONS (show confirm UI — use for config changes requiring user approval):
- backtest: <action>{"action":"backtest","changes":{"dot.path":value},"description":"..."}</action>
- update_config: <action>{"action":"update_config","changes":{"dot.path":value},"description":"..."}</action>
- show_config: <action>{"action":"show_config"}</action>
- show_history: <action>{"action":"show_history"}</action>

TODAY'S SLATE:
${slateStr}

CURRENT MODEL CONFIG:
${cfgStr}

LATEST ACCURACY BRIEFING:
${briefStr}

BEHAVIOR:
- Direct and concise. No preamble. Answer what's asked.
- For lineup/pick questions, use context above before reaching for tools.
- For model changes: backtest param changes; read code before describing changes.
- You cannot modify code files (api/, index.html, .github/). If asked to change code, read the relevant file, describe the exact change needed, and direct the user to implement it via a code push.
- One change per response unless explicitly asked for more.`;
}

function _labSyncParlayContext(data) {
  LAB.parlayData = data && typeof data === 'object' ? data : null;
  if (LAB.initialized) LAB.system = buildLabSystemPrompt();
}

function _labSpinner(statusText) {
  const label = (statusText && String(statusText).trim()) ? String(statusText) : 'Ben is thinking…';
  const el = document.createElement('div');
  el.className = 'lab-msg thinking';
  el.innerHTML = '<div class="lab-thinking-dots"><span></span><span></span><span></span></div>'
    + '<div class="lab-think-status">' + _escapeHtml(label) + '</div>';
  const msgs = document.getElementById('labMessages');
  if (msgs) msgs.classList.add('lab-has-messages');
  msgs.appendChild(el);
  const anchor = document.getElementById('labScrollAnchor');
  if (anchor) msgs.appendChild(anchor); // keep anchor as last child
  _labScrollToBottom();
  return el;
}

async function labCallClaude(attachedImage = null) {
  const labMsgs = document.getElementById('labMessages');
  const thinkEl = _labSpinner('');

  const _sendBtn = document.getElementById('labSendBtn'); if (_sendBtn) _sendBtn.disabled = true;

  try {
    // Build API messages — image only injected into the last user message (not stored in history)
    const rawMsgs = LAB.messages.filter(m => !m.hidden);
    const apiMessages = rawMsgs.map((m, i) => {
      if (attachedImage && i === rawMsgs.length - 1 && m.role === 'user') {
        return {
          role: 'user',
          content: [
            { type: 'image', source: { type: 'base64', media_type: attachedImage.mediaType, data: attachedImage.base64 } },
            { type: 'text', text: m.content },
          ],
        };
      }
      return { role: m.role, content: m.content };
    });

    // SSE streaming requires raw fetch() — fetchWithTimeout doesn't support streaming responses.
    // Manual AbortController provides 60s connection timeout.
    const chatAbort = new AbortController();
    const chatTimeoutId = setTimeout(() => chatAbort.abort(), 60000);
    let r;
    try {
      r = await fetch('/api/lab/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: apiMessages, system: LAB.system }),
        signal: chatAbort.signal,
      });
    } finally {
      clearTimeout(chatTimeoutId); // Always clear — prevents orphaned timer on network errors
    }

    // Read SSE stream — backend emits status events then final content
    if (!r.ok) throw new Error('chat ' + r.status);
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let handled = false;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split('\n\n');
      buf = parts.pop(); // keep any incomplete chunk
      for (const part of parts) {
        if (!part.startsWith('data: ')) continue;
        let evt;
        try { evt = JSON.parse(part.slice(6)); } catch { continue; }
        if (evt.type === 'status') {
          thinkEl.innerHTML = '<div class="lab-thinking-dots"><span></span><span></span><span></span></div><div class="lab-think-status">' + _escapeHtml(evt.text) + '</div>';
          labMsgs.scrollTop = labMsgs.scrollHeight;
        } else if (evt.type === 'content') {
          thinkEl.remove();
          handled = true;
          if (evt.error) { appendLabMessage('assistant', `Error: ${evt.error}`); }
          else { await handleLabResponse(evt.text || ''); }
        }
      }
    }
    if (!handled) { thinkEl.remove(); appendLabMessage('assistant', 'No response received. Try again.'); }

  } catch(e) {
    thinkEl.remove();
    const msg = (e && e.name === 'AbortError') ? 'Request timed out. Please try again.' : 'Network error. Please try again.';
    appendLabMessage('assistant', msg);
  } finally {
    const _sendBtnF = document.getElementById('labSendBtn'); if (_sendBtnF) _sendBtnF.disabled = false;
  }
}

async function handleLabResponse(content) {
  // Extract and handle any action tags
  const actionMatch = content.match(/<action>([\s\S]*?)<\/action>/);
  const displayContent = content.replace(/<action>[\s\S]*?<\/action>/g, '').trim();

  appendLabMessage('assistant', displayContent);

  if (actionMatch) {
    try {
      const action = JSON.parse(actionMatch[1]);
      await executeLabAction(action);
    } catch(e) {
      // Not valid JSON in action tag — ignore
    }
  }
}

async function executeLabAction(action) {
  const type = action.action;

  if (type === 'backtest') {
    const btEl = _labSpinner('Running backtest on historical slates...');
    appendLabMessage('user', `Backtest result: running...`, true); // placeholder
    try {
      const r = await fetchWithTimeout('/api/lab/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ proposed_changes: action.changes, description: action.description }),
      }, 120000);
      if (!r.ok) throw new Error('Backtest failed: HTTP ' + r.status);
      const result = await r.json();
      btEl.remove();
      appendLabMessage('user', `Backtest result: ${JSON.stringify(result)}`, true);
      await labCallClaude();
    } catch(e) {
      btEl.remove();
      appendLabMessage('assistant', 'Backtest failed. Try again.');
    }

  } else if (type === 'show_config') {
    const cfgEl = _labSpinner('Loading config...');
    try {
      const r = await fetchWithTimeout('/api/lab/config-history', {}, 10000);
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const data = await r.json();
      cfgEl.remove();
      appendLabMessage('user', `Config data: ${JSON.stringify(data)}`, true);
      await labCallClaude();
    } catch(e) { cfgEl.remove(); }

  } else if (type === 'show_history') {
    try {
      const r = await fetchWithTimeout('/api/lab/config-history', {}, 10000);
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const data = await r.json();
      const changelogStr = (data.changelog || []).map(c => `v${c.version} (${c.date}): ${c.change}`).join('\n');
      appendLabMessage('assistant', `**Config History:**\n${changelogStr}`);
    } catch(e) {}

  } else if (type === 'update_config') {
    // Don't auto-apply — show confirm button
    const msgEl = document.createElement('div');
    msgEl.className = 'lab-msg assistant';
    const descText = action.description || JSON.stringify(action.changes);
    msgEl.innerHTML = `
      <div class="lab-msg-label">Ben</div>
      <div data-config-status>Ready to apply: <b></b></div>
      <div style="margin-top:10px;display:flex;gap:8px">
        <button class="lab-apply-btn"
          style="background:var(--lab);border:none;border-radius:var(--radius-card);color:var(--color-text-primary);padding:6px 14px;font-family:'Barlow Condensed',sans-serif;font-weight:800;font-size:0.8rem;text-transform:uppercase;cursor:pointer">
          Apply
        </button>
        <button class="lab-decline-btn"
          style="background:transparent;border:1px solid var(--border);border-radius:8px;color:var(--muted);padding:6px 12px;font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:0.8rem;text-transform:uppercase;cursor:pointer">
          Decline
        </button>
      </div>`;
    msgEl.querySelector('div b').textContent = descText;
    msgEl.querySelector('.lab-apply-btn').addEventListener('click', () => applyLabConfig(action.changes, action.description || '', msgEl));
    msgEl.querySelector('.lab-decline-btn').addEventListener('click', () => {
      const st = msgEl.querySelector('[data-config-status]');
      if (st) st.textContent = 'Change declined.';
      msgEl.querySelector('.lab-apply-btn')?.closest('div')?.remove();
    });
    const _lm = document.getElementById('labMessages');
    if (_lm) { _lm.classList.add('lab-has-messages'); _lm.appendChild(msgEl); }
    msgEl.scrollIntoView({behavior:'smooth'});
  }
}

async function applyLabConfig(changes, description, msgEl) {
  const statusDiv = msgEl.querySelector('[data-config-status]');
  const btnDiv    = msgEl.querySelector('.lab-apply-btn')?.parentElement;
  try {
    const r = await fetchWithTimeout('/api/lab/update-config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ changes, change_description: description }),
    }, 15000);
    if (!r.ok) {
      const errData = await r.json().catch(() => ({}));
      if (statusDiv) statusDiv.textContent = errData.error || ('HTTP ' + r.status);
      return;
    }
    const result = await r.json().catch(() => ({}));
    if (result.error) {
      if (statusDiv) statusDiv.textContent = `Error: ${result.error}`;
      return;
    }
    if (statusDiv) statusDiv.textContent = `Applied — now config v${result.version}.`;
    if (btnDiv) btnDiv.remove();
    appendLabMessage('user', `Config updated to v${result.version}: ${description}`, true);
    await labCallClaude();
  } catch(e) {
    if (statusDiv) statusDiv.textContent = 'Failed to apply config change.';
  }
}

function formatLabTables(text) {
  const escape = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const rowRe = /^\|.+\|$/;
  const sepRe = /^\|[\s\-:]+(\|[\s\-:]+)*\|$/;
  const lines = text.split(/\r?\n/);
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const start = i;
    while (i < lines.length && rowRe.test(lines[i])) i++;
    const run = lines.slice(start, i);
    if (run.length >= 1) {
      const headerCells = run[0].split('|').slice(1, -1).map(c => escape(c.trim()));
      let bodyStart = 1;
      if (run.length > 1 && sepRe.test(run[1])) bodyStart = 2;
      const bodyRows = run.slice(bodyStart).map(row => row.split('|').slice(1, -1).map(c => escape(c.trim())));
      const colCount = headerCells.length;
      const valid = bodyRows.every(row => row.length === colCount);
      if (valid && colCount > 0) {
        const thRow = '<tr>' + headerCells.map(c => `<th>${c}</th>`).join('') + '</tr>';
        const tdRows = bodyRows.map(row => '<tr>' + row.map(c => `<td>${c}</td>`).join('') + '</tr>').join('');
        out.push('<table class="lab-table"><thead>' + thRow + '</thead><tbody>' + tdRows + '</tbody></table>');
      } else {
        run.forEach(l => out.push(l));
      }
    }
    if (i < lines.length) { out.push(lines[i]); i++; }
  }
  return out.join('\n');
}

function appendLabMessage(role, content, hidden = false, isStatus = false, imageSrc = null) {
  LAB.messages.push({ role, content, hidden });
  if (hidden) return;

  const el = document.createElement('div');
  el.className = `lab-msg ${role}`;
  const msgsRoot = document.getElementById('labMessages');
  if (msgsRoot) msgsRoot.classList.add('lab-has-messages');

  if (role === 'assistant') {
    // Convert basic markdown: code blocks, tables, bold, italic, newlines
    const afterCode = content
      .replace(/```[\w]*\n?([\s\S]*?)```/g, (_,c) =>
        `<pre style="background:rgba(0,0,0,0.35);border:1px solid var(--border);border-radius:6px;padding:8px 10px;font-size:0.74rem;overflow-x:auto;white-space:pre-wrap;margin:6px 0">${c.trim()}</pre>`);
    const afterTables = formatLabTables(afterCode);
    const formatted = afterTables
      .replace(/`([^`]+)`/g, '<code style="background:rgba(0,0,0,0.3);border-radius:3px;padding:1px 4px;font-size:0.8em">$1</code>')
      .replace(/\*\*(.*?)\*\*/g, '<b>$1</b>')
      .replace(/\*(.*?)\*/g, '<i>$1</i>')
      .replace(/\n/g, '<br>');
    el.innerHTML = `<div class="lab-msg-label">Ben</div><div>${formatted}</div>`;
  } else if (imageSrc) {
    // User message with attached image — show thumbnail + text
    el.innerHTML = `<img src="${imageSrc}" style="width:100%;max-width:220px;border-radius:8px;display:block;margin-bottom:6px;opacity:0.9">${content ? `<div style="margin-top:2px">${content}</div>` : ''}`;
  } else {
    el.textContent = content;
  }

  const msgs = msgsRoot || document.getElementById('labMessages');
  msgs.appendChild(el);
  const anchor3 = document.getElementById('labScrollAnchor');
  if (anchor3) msgs.appendChild(anchor3);
  _labScrollToBottom();
  _benPersistChatMirror();
}

// ── Ben scroll helper ──
// Uses a dummy anchor element and rAF so viewport reflow settles first.
function _labScrollToBottom() {
  const anchor = document.getElementById('labScrollAnchor');
  if (!anchor) return;
  requestAnimationFrame(() => {
    anchor.scrollIntoView({ behavior: 'smooth', block: 'end' });
  });
}

// ── Ben keyboard / nav management ──
// On mobile: when the virtual keyboard opens, the fixed bottom nav floats between
// the input and the keyboard — hide it while Ben's input is focused, restore on blur.
// On desktop: skip entirely (no virtual keyboard, no nav hiding needed).
(function setupBenKeyboard() {
  const isMobile = () => window.matchMedia('(hover: none) and (pointer: coarse)').matches;
  const input  = document.getElementById('labInput');
  const nav    = document.getElementById('bottomNav');
  const tabLab = document.getElementById('tab-lab');
  if (!input || !nav || !tabLab) return;
  const root = document.documentElement;

  // Keep layout CSS-driven: write viewport vars, avoid direct element height math.
  const _syncLabViewportVars = () => {
    const vv = window.visualViewport;
    const vhPx = vv ? Math.round(vv.height) : window.innerHeight;
    root.style.setProperty('--lab-vh', `${vhPx}px`);
    const kbHeight = vv ? Math.max(0, Math.round(window.innerHeight - vv.height - vv.offsetTop)) : 0;
    root.style.setProperty('--keyboard-height', `${kbHeight}px`);
  };
  _syncLabViewportVars();

  if (window.visualViewport) {
    window.visualViewport.addEventListener('resize', _syncLabViewportVars);
    window.visualViewport.addEventListener('scroll', _syncLabViewportVars);
  }
  if (navigator.virtualKeyboard) {
    try { navigator.virtualKeyboard.overlaysContent = false; } catch (_) {}
    navigator.virtualKeyboard.addEventListener('geometrychange', () => {
      const rect = navigator.virtualKeyboard.boundingRect;
      root.style.setProperty('--keyboard-height', `${Math.max(0, Math.round(rect?.height || 0))}px`);
      _syncLabViewportVars();
    });
  }

  // P4: MutationObserver — auto-scroll #labMessages whenever new content is added
  // (covers Claude streaming tokens and new messages). Only scrolls if user is already
  // near the bottom (within 120px), so manually scrolling up to read history isn't hijacked.
  // Disconnect any existing observer before re-attaching (prevents accumulation on re-open).
  const msgs = document.getElementById('labMessages');
  if (msgs && typeof MutationObserver !== 'undefined') {
    if (window._labMsgObserver) { window._labMsgObserver.disconnect(); }
    window._labMsgObserver = new MutationObserver(() => {
      const distFromBottom = msgs.scrollHeight - msgs.scrollTop - msgs.clientHeight;
      if (distFromBottom < 200) _labScrollToBottom();
    });
    window._labMsgObserver.observe(msgs, { childList: true, subtree: true, characterData: true });
  }

  input.addEventListener('focus', () => {
    if (!isMobile()) return;
    tabLab.classList.add('lab-kb-open');
    nav.style.transition = 'opacity 0.15s ease, transform 0.15s ease';
    nav.style.opacity = '0';
    nav.style.transform = 'translateX(-50%) translateY(16px)';
    nav.style.pointerEvents = 'none';
    window.scrollTo(0, 0);
    _syncLabViewportVars();
    _labScrollToBottom(); // pin to bottom as keyboard slides up
  });
  input.addEventListener('blur', () => {
    if (!isMobile()) return;
    tabLab.classList.remove('lab-kb-open');
    nav.style.opacity = '';
    nav.style.transform = '';
    nav.style.pointerEvents = '';
    window.scrollTo(0, 0);
    _syncLabViewportVars();
    _labScrollToBottom();
  });
})();

function _labAttachPhoto(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    const dataUrl = e.target.result;
    LAB.pendingImage = { base64: dataUrl.split(',')[1], mediaType: file.type || 'image/jpeg', dataUrl };
    document.getElementById('labPhotoThumb').src = dataUrl;
    document.getElementById('labPhotoPreview').style.display = 'flex';
    document.getElementById('labPhotoBtn').style.background = 'rgba(20,184,166,0.18)';
  };
  reader.readAsDataURL(file);
}

function _labClearPhoto() {
  LAB.pendingImage = null;
  document.getElementById('labPhotoPreview').style.display = 'none';
  document.getElementById('labPhotoBtn').style.background = '';
}

async function labSend() {
  const input = document.getElementById('labInput');
  const text  = input.value.trim();
  const img   = LAB.pendingImage;
  if (!text && !img) return;
  input.value = '';
  const sendBtn = document.getElementById('labSendBtn');
  if (sendBtn) { sendBtn.disabled = true; sendBtn.style.opacity = '0.4'; }
  input.focus();
  if (img) _labClearPhoto();

  const displayText = text || 'What do you see in this screenshot?';
  appendLabMessage('user', displayText, false, false, img?.dataUrl);
  await labCallClaude(img);
}

