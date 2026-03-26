"""
Playwright session management for Real Sports ingestion.

Handles login, session persistence, and page setup.
Session state is saved to /tmp/real_sports_session.json so subsequent
runs within the same day skip re-authentication.

Usage:
    from playwright.sync_api import sync_playwright
    from scripts.real_ingest.session import get_page

    with sync_playwright() as pw:
        page = get_page(pw)
        # page is authenticated and ready
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

SESSION_FILE = Path("/tmp/real_sports_session.json")
BASE_URL = os.environ.get("REAL_SPORTS_BASE_URL", "https://www.realsports.io").rstrip("/")


def _load_env() -> tuple[str, str]:
    """Return (username, password) from environment. Raise if missing."""
    username = os.environ.get("REAL_SPORTS_USERNAME", "").strip()
    password = os.environ.get("REAL_SPORTS_PASSWORD", "").strip()
    if not username or not password:
        raise EnvironmentError(
            "REAL_SPORTS_USERNAME and REAL_SPORTS_PASSWORD must be set in the environment. "
            "Copy .env.example → .env and fill in your credentials."
        )
    return username, password


def _kill_switch_check() -> None:
    """Abort if the automation kill switch is set to false."""
    enabled = os.environ.get("INGEST_AUTOMATION_ENABLED", "true").strip().lower()
    if enabled in ("false", "0", "no", "off"):
        raise RuntimeError(
            "Ingestion automation is disabled (INGEST_AUTOMATION_ENABLED=false). "
            "Set it to 'true' in your .env to enable."
        )


def _is_authenticated(page) -> bool:
    """
    Quick heuristic to check if we're already logged in.
    Looks for common authenticated-state indicators.
    We check the URL (not on login page) and presence of app content.
    """
    url = page.url
    if "login" in url.lower() or "signin" in url.lower():
        return False
    # Check for common nav elements present when logged in
    for selector in [
        "[data-testid='user-menu']",
        "[data-testid='nav-profile']",
        ".user-avatar",
        ".profile-menu",
        "[aria-label='profile']",
        "button:has-text('Draft')",
        "button:has-text('Leaderboard')",
        "nav",
    ]:
        try:
            if page.locator(selector).first.is_visible(timeout=500):
                return True
        except Exception:
            continue
    return False


def _login(page, username: str, password: str) -> None:
    """
    Navigate to the login page, fill credentials, and submit.
    Waits for the authenticated state before returning.
    Tries multiple common login form selector patterns.
    """
    logger.info("Navigating to login page: %s", BASE_URL)
    page.goto(BASE_URL, wait_until="domcontentloaded", timeout=30_000)
    time.sleep(2)

    # If already on an authenticated screen, skip
    if _is_authenticated(page):
        logger.info("Already authenticated (session cookie active).")
        return

    # Navigate to login if we're on a landing page
    login_url = f"{BASE_URL}/login"
    current_url = page.url
    if "login" not in current_url.lower():
        # Try clicking a login button first
        for sel in [
            "a:has-text('Log in')",
            "a:has-text('Login')",
            "a:has-text('Sign in')",
            "button:has-text('Log in')",
            "button:has-text('Sign in')",
        ]:
            try:
                btn = page.locator(sel).first
                if btn.is_visible(timeout=1000):
                    logger.info("Clicking login button: %s", sel)
                    btn.click()
                    page.wait_for_load_state("domcontentloaded", timeout=10_000)
                    time.sleep(1)
                    break
            except Exception:
                continue
        else:
            # Direct navigate to /login
            logger.info("Navigating directly to %s", login_url)
            page.goto(login_url, wait_until="domcontentloaded", timeout=30_000)
            time.sleep(2)

    # Fill username
    username_selectors = [
        "input[type='email']",
        "input[type='text'][name*='user']",
        "input[name='email']",
        "input[name='username']",
        "input[placeholder*='email' i]",
        "input[placeholder*='username' i]",
        "input[autocomplete='username']",
        "input[autocomplete='email']",
    ]
    filled_user = False
    for sel in username_selectors:
        try:
            field = page.locator(sel).first
            if field.is_visible(timeout=1000):
                field.fill(username)
                logger.info("Filled username into: %s", sel)
                filled_user = True
                break
        except Exception:
            continue

    if not filled_user:
        # Screenshot for debugging
        _save_debug_screenshot(page, "login_page_no_username_field")
        raise RuntimeError(
            "Could not find username/email input on the login page. "
            "Check discovery_report/login_page_no_username_field.png for the current page state."
        )

    # Fill password
    password_selectors = [
        "input[type='password']",
        "input[name='password']",
        "input[autocomplete='current-password']",
    ]
    filled_pass = False
    for sel in password_selectors:
        try:
            field = page.locator(sel).first
            if field.is_visible(timeout=1000):
                field.fill(password)
                logger.info("Filled password into: %s", sel)
                filled_pass = True
                break
        except Exception:
            continue

    if not filled_pass:
        _save_debug_screenshot(page, "login_page_no_password_field")
        raise RuntimeError(
            "Could not find password input on the login page. "
            "Check discovery_report/login_page_no_password_field.png"
        )

    # Submit
    submit_selectors = [
        "button[type='submit']",
        "input[type='submit']",
        "button:has-text('Log in')",
        "button:has-text('Sign in')",
        "button:has-text('Login')",
        "button:has-text('Continue')",
    ]
    submitted = False
    for sel in submit_selectors:
        try:
            btn = page.locator(sel).first
            if btn.is_visible(timeout=1000):
                logger.info("Clicking submit: %s", sel)
                btn.click()
                submitted = True
                break
        except Exception:
            continue

    if not submitted:
        # Try pressing Enter on the password field
        page.keyboard.press("Enter")
        submitted = True

    # Wait for redirect away from login
    logger.info("Waiting for post-login redirect...")
    try:
        page.wait_for_url(lambda url: "login" not in url.lower(), timeout=20_000)
    except Exception:
        pass
    page.wait_for_load_state("domcontentloaded", timeout=15_000)
    time.sleep(3)

    if not _is_authenticated(page):
        _save_debug_screenshot(page, "login_failed")
        raise RuntimeError(
            "Login appears to have failed — still on login page or no authenticated elements found. "
            "Check discovery_report/login_failed.png for the page state."
        )

    logger.info("Login successful. Current URL: %s", page.url)


def _save_debug_screenshot(page, name: str) -> None:
    out_dir = Path("scripts/real_ingest/discovery_report/screenshots")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    try:
        page.screenshot(path=str(path))
        logger.warning("Debug screenshot saved: %s", path)
    except Exception as e:
        logger.warning("Could not save debug screenshot: %s", e)


def get_page(playwright, headless: bool = False):
    """
    Return an authenticated Playwright Page.

    Strategy:
    1. Try to restore saved session state (storage_state) from SESSION_FILE.
    2. If restoration fails or session is expired, perform a fresh login.
    3. Save the new session state for subsequent runs.

    Args:
        playwright: The sync Playwright instance.
        headless:   Run Chromium headlessly. Default False for dev/debug.
                    Set to True for automated/CI runs.

    Returns:
        playwright Page object, already authenticated.
    """
    _kill_switch_check()
    username, password = _load_env()

    # Use explicit executable path if Playwright's managed binary is missing
    # (common in sandboxed / restricted-download environments)
    _FALLBACK_CHROME_PATHS = [
        "/root/.cache/ms-playwright/chromium-1194/chrome-linux/chrome",
        "/root/.cache/ms-playwright/chromium-1208/chrome-linux/chrome",
        "/usr/bin/chromium-browser",
        "/usr/bin/chromium",
        "/usr/bin/google-chrome",
    ]
    import os as _os
    _exe = next((p for p in _FALLBACK_CHROME_PATHS if _os.path.exists(p)), None)

    browser_args: dict = {
        "headless": headless,
        "args": ["--no-sandbox", "--disable-dev-shm-usage"],
    }
    if _exe:
        browser_args["executable_path"] = _exe
        logger.info("Using Chromium executable: %s", _exe)

    # Try restoring saved session
    if SESSION_FILE.exists():
        logger.info("Attempting to restore session from %s", SESSION_FILE)
        try:
            browser = playwright.chromium.launch(**browser_args)
            context = browser.new_context(
                storage_state=str(SESSION_FILE),
                viewport={"width": 1280, "height": 900},
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
            )
            page = context.new_page()
            page.goto(BASE_URL, wait_until="domcontentloaded", timeout=30_000)
            time.sleep(2)
            if _is_authenticated(page):
                logger.info("Session restored successfully.")
                return page
            logger.info("Saved session expired, performing fresh login.")
            context.close()
            browser.close()
        except Exception as e:
            logger.warning("Session restore failed: %s — performing fresh login.", e)

    # Fresh login
    browser = playwright.chromium.launch(**browser_args)
    context = browser.new_context(
        viewport={"width": 1280, "height": 900},
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    )
    page = context.new_page()

    _login(page, username, password)

    # Persist session
    try:
        SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        context.storage_state(path=str(SESSION_FILE))
        logger.info("Session saved to %s", SESSION_FILE)
    except Exception as e:
        logger.warning("Could not save session state: %s", e)

    return page
