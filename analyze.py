import asyncio
import json
import math
import os
import re
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from evaluation_store import (
    ENTRY_WAIT_HOURS,
    TRADE_MAX_HOLD_HOURS,
    cleanup_evaluation_data,
    clear_evaluation_data,
    prompt_hash,
    save_evaluation_case,
)

load_dotenv()

# Bot analyzes and tracks USDⓈ-M perpetual futures, not spot — klines/price/funding/OI must all
# come from the futures market so structure, current price, and outcome tracking stay consistent
# with what the user actually trades. Response schema is identical to spot (same 12 kline fields).
BINANCE_FUTURES_API_BASE = "https://fapi.binance.com"
BINANCE_API_URL   = f"{BINANCE_FUTURES_API_BASE}/fapi/v1/klines"
# Quote asset every symbol is resolved/quoted against. Binance Futures lists far fewer USDC pairs
# (~38) than USDT pairs (~680) — switching this away from USDT only works for symbols that actually
# have a <BASE>USDC contract; resolve_binance_symbol/add_symbol's existing price-check already fails
# clearly for anything that doesn't.
BINANCE_QUOTE_ASSET = (os.getenv("BINANCE_QUOTE_ASSET", "USDT") or "USDT").strip().upper()

# Some tokens were rebased 1000x when Binance listed their perpetual futures contract — the Futures
# symbol differs from the Spot/common name (e.g. spot SHIBUSDT vs futures 1000SHIBUSDT). Verified live
# against /fapi/v1/ticker/price: the bare name returns HTTP 400 "Invalid symbol" on futures, only the
# 1000x-prefixed name resolves. Without this map, admins typing the common name could never analyze
# these coins even though they're genuinely listed.
BINANCE_FUTURES_SYMBOL_ALIASES = {
    "SHIB": "1000SHIB", "PEPE": "1000PEPE", "BONK": "1000BONK", "FLOKI": "1000FLOKI",
    "LUNC": "1000LUNC", "RATS": "1000RATS", "XEC": "1000XEC", "SATS": "1000SATS",
}


def resolve_binance_symbol(raw: str) -> str:
    """Normalize a user-typed symbol into the actual Binance Futures symbol string."""
    s = (raw or "").strip().lstrip("/").upper()
    if not s:
        return ""
    for quote in ("USDT", "USDC"):
        if s.endswith(quote):
            s = s[:-len(quote)]
            break
    s = BINANCE_FUTURES_SYMBOL_ALIASES.get(s, s)
    return f"{s}{BINANCE_QUOTE_ASSET}"


def _binance_get_with_retry(
    url: str, params: dict, max_retries: int = 2, timeout: int = 15
) -> "requests.Response | None":
    """GET with retry+backoff; a transient network blip must not silently drop a timeframe.

    429/418 (rate limit / IP ban) get a longer backoff since Binance explicitly asks callers to
    slow down; other errors (timeout, DNS, 5xx) get a shorter linear backoff. 4xx other than
    429/418 (e.g. 400 invalid symbol) is a permanent client error, not transient — retrying just
    wastes time and requests, so it fails fast instead.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.exceptions.HTTPError as exc:
            last_exc = exc
            status = exc.response.status_code if exc.response is not None else None
            if status is not None and 400 <= status < 500 and status not in (429, 418):
                break
            if attempt < max_retries:
                time.sleep((3.0 if status in (429, 418) else 1.5) * (attempt + 1))
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
    print(f"Binance API lỗi: {url} params={params} error={last_exc}", flush=True)
    return None




def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


# Reasoning effort for the Planner call (now routed through OpenRouter; see PLANNER_MODEL).
# Defaults to "high" since the main goal is cost control at scale. Can be changed to "max" on Railway.
PLANNER_REASONING_EFFORT = os.getenv("PLANNER_REASONING_EFFORT", "max").strip()
PLANNER_RETRY_REASONING_EFFORT = os.getenv(
    "PLANNER_RETRY_REASONING_EFFORT", PLANNER_REASONING_EFFORT or "max"
).strip()

# Max reasoning shares the same completion token budget as the final answer.
# The cap must be large enough that after reasoning, the model still has room to output a parseable format.
PLANNER_MAX_OUTPUT_TOKENS = int(os.getenv("PLANNER_MAX_OUTPUT_TOKENS", "12000"))
PLANNER_OUTPUT_TOKEN_CAP = int(os.getenv("PLANNER_OUTPUT_TOKEN_CAP", "12000"))
# Main analysis has no continuation: output is short, and continuation would just turn one request into multiple rounds that can hang.
PLANNER_MAX_CONTINUATIONS = int(os.getenv("PLANNER_MAX_CONTINUATIONS", "0"))
# Timeout/retry settings for the AI provider.
# GLM uses max reasoning for both the first attempt and the retry; the retry is still capped at one attempt.
PLANNER_TIMEOUT_SECONDS = int(os.getenv("PLANNER_TIMEOUT_SECONDS", "240"))
PLANNER_RETRY_TIMEOUT_SECONDS = int(os.getenv("PLANNER_RETRY_TIMEOUT_SECONDS", "150"))
PLANNER_API_RETRIES = int(os.getenv("PLANNER_API_RETRIES", "1"))
PLANNER_RETRY_LIMIT = int(os.getenv("PLANNER_RETRY_LIMIT", "1"))
PLANNER_RETRY_SLEEP_SECONDS = float(os.getenv("PLANNER_RETRY_SLEEP_SECONDS", "2"))

# ─── Auto Scan mode config ──────────────────────────────────────────────────
# Auto Scan calls Planner directly on a fixed schedule (hourly, aligned to the 1H candle close).
# There is no separate filter/review stage: a NO_TRADE label is discarded, anything else is sent as-is.
AUTOSCAN_INTERVAL_SECONDS = int(os.getenv("AUTOSCAN_INTERVAL_SECONDS", "3600"))
AUTOSCAN_MODES = [m.strip().lower() for m in os.getenv("AUTOSCAN_MODES", "short").split(",") if m.strip()]
AUTO_SCAN_MAX_SYMBOLS_PER_RUN = 1  # Auto Scan only allows 1 symbol per user to avoid wasting resources.
AUTOSCAN_SEND_NO_TRADE = os.getenv("AUTOSCAN_SEND_NO_TRADE", "0").strip().lower() in {"1", "true", "yes", "on"}
AUTOSCAN_CANDLE_CLOSE_DELAY_SECONDS = int(os.getenv("AUTOSCAN_CANDLE_CLOSE_DELAY_SECONDS", "5"))
# Job scheduler only wakes up to check whether a candle-close slot is due.
# It does NOT call Binance/LLM unless should_run_auto_scan_now() returns true.
AUTOSCAN_SCHEDULER_TICK_SECONDS = max(30, int(os.getenv("AUTOSCAN_SCHEDULER_TICK_SECONDS", "60") or "60"))
# The user-facing log only ever keeps the 5 most recent entries. This is fixed in code so the old
# Railway variable AUTO_SCAN_LOG_LIMIT=20 doesn't accidentally make the DB/Telegram log long again.
AUTO_SCAN_LOG_LIMIT = 5  # number of rows shown to the user
AUTOSCAN_LOG_RETENTION_DAYS = max(1, int(os.getenv("AUTOSCAN_LOG_RETENTION_DAYS", "14")))
AUTOSCAN_DEBUG = os.getenv("AUTOSCAN_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}

# Prevent overlapping Auto Scan cycles. If a candle-close slot arrives while a cycle
# is still running, the active cycle performs at most one catch-up pass for the
# newest closed slot after it finishes. Older missed slots are intentionally skipped.
_AUTO_SCAN_RUN_LOCK = asyncio.Lock()
_AUTO_SCAN_CATCH_UP_MAX_PASSES = 1
# Auto Scan sleep window in Vietnam time: 00:00-07:00.
AUTOSCAN_SLEEP_HOUR_VN = int(os.getenv("AUTOSCAN_SLEEP_HOUR_VN", "0"))
AUTOSCAN_WAKE_HOUR_VN = int(os.getenv("AUTOSCAN_WAKE_HOUR_VN", "7"))
# Each user can call Planner at most N times per Auto Scan day (07:00 VN to 06:59 the next day).
# AUTOSCAN_MAX_PLANNER_CALLS_PER_DAY is the name to set; the older names still work as fallbacks so
# an existing Railway config keeps behaving the same after this rename.
AUTOSCAN_MAX_PLANNER_CALLS_PER_DAY = max(
    1,
    int(os.getenv(
        "AUTOSCAN_MAX_PLANNER_CALLS_PER_DAY",
        os.getenv("AUTO_SCAN_MAX_FINAL_AI_CALLS_PER_DAY", os.getenv("AUTO_SCAN_MAX_GLM_CALLS_PER_DAY", "5")),
    )),
)

# OpenRouter — single provider for Planner, the only AI stage left in the pipeline.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
PLANNER_MODEL = os.getenv("PLANNER_MODEL", os.getenv("OPENROUTER_PLANNER_MODEL", "deepseek/deepseek-v4-flash-0731"))

DB_PATH           = os.getenv("DB_PATH", "bot.db")

# All four timeframes in each mode get identical treatment in the prompt (see
# build_feature_engineering_block) — Python does not assign any one of them a role like
# "decides direction" or "designs Entry/SL/TP". Which frame matters for what is entirely
# the model's own judgment call. The numbers below are just each frame's raw-candle window.
SHORT_TERM_TIMEFRAMES = {
    "15M": ("15m", 480),   # ~5 days
    "1H":  ("1h",  360),   # ~15 days
    "4H":  ("4h",  360),   # ~60 days
    "1D":  ("1d",  365),   # ~1 year
}

LONG_TERM_TIMEFRAMES = {
    "4H": ("4h",  360),   # ~60 days
    "1D": ("1d",  365),   # ~1 year
    "1W": ("1w",  208),   # ~4 years
    # Fetch limit is much larger than the ~6 candles actually shown (see _v50_raw_limit) because
    # EMA50/RSI24/MACD/ADX/vol_ratio each need their own warm-up period (up to 50 candles) before
    # producing a value at all. Requesting 150 is harmless even though Binance Futures has only
    # existed since Sept 2019 (~85 months of real 1M history as of 2026, so it always returns
    # fewer than 150 today) — this just means the limit won't be the bottleneck again as more
    # history accumulates year over year. Ichimoku's Senkou Span B needs 52+26=78 candles AFTER
    # that 50-candle warm-up (~128 total) to produce a value on 1M specifically — real history
    # doesn't clear that yet for any coin including BTC, so 1M's Ichimoku line is correctly
    # omitted for now (see _v50_ichimoku_block); this isn't fixable by raising the limit further,
    # only by the exchange's own history getting longer.
    "1M": ("1M",  150),
}

# Lifecycle by mode: short = SCALP, long = SWING
# (ENTRY_WAIT_HOURS / TRADE_MAX_HOLD_HOURS are imported from evaluation_store.py above -
# the single source of truth, to avoid the hour mismatch between the two modules that happened before.)

CHECK_INTERVAL_HOURS = {
    "short": 0.5,     # Scalp: check every 30min, matching job_check_predictions' own 30min interval
    "long": 12,       # Swing: check every 12h
}

RESULT_CHECK_INTERVAL = {
    "short": "15m",   # Scalp: score the outcome using 15-minute candles
    "long": "1h",     # Swing: score the outcome using 1-hour candles
}


def get_result_check_interval(mode: str) -> str:
    return RESULT_CHECK_INTERVAL.get(mode, "15m")

VISIBLE_PREDICTION_RETENTION_LIMIT = 5
HIDDEN_LEARNING_RETENTION_LIMIT = 5
# REJECTED_PLAN/NO_TRADE are no longer saved into predictions after every analysis.
# This variable is kept only to filter legacy data from older DB versions.
HIDDEN_LEARNING_RESULTS = ("REJECTED_PLAN", "NO_TRADE")
VN_TZ = timezone(timedelta(hours=7))


# ─── DB ───────────────────────────────────────────────────────────────────────

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


_prediction_db_initialized = False


def init_prediction_db() -> None:
    global _prediction_db_initialized
    if _prediction_db_initialized:
        return
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id             INTEGER,
                chat_id             INTEGER,
                symbol              TEXT NOT NULL,
                mode                TEXT NOT NULL,
                created_at          TEXT NOT NULL,
                check_after_hours   INTEGER NOT NULL DEFAULT 12,
                entry_wait_hours    INTEGER NOT NULL DEFAULT 12,
                max_hold_hours      INTEGER NOT NULL DEFAULT 72,
                next_check_at       TEXT,
                direction           TEXT NOT NULL,
                entry_low           REAL,
                entry_high          REAL,
                sl                  REAL,
                tp1                 REAL,
                tp2                 REAL,
                entry_status        TEXT NOT NULL DEFAULT 'PENDING_ENTRY',
                entry_filled_at     TEXT,
                entry_price         REAL,
                trade_closed_at     TEXT,
                rr_result           REAL,
                hold_hours          REAL,
                market_snapshot     TEXT,
                feature_snapshot    TEXT,
                reasoning_summary   TEXT,
                full_response       TEXT,
                result              TEXT NOT NULL DEFAULT 'PENDING_ENTRY',
                result_price        REAL,
                result_reason       TEXT,
                result_checked_at   TEXT
            )
        """)
        for col, definition in [
            ("user_id", "INTEGER"),
            ("chat_id", "INTEGER"),
            ("check_after_hours", "INTEGER NOT NULL DEFAULT 12"),
            ("entry_wait_hours", "INTEGER NOT NULL DEFAULT 12"),
            ("max_hold_hours", "INTEGER NOT NULL DEFAULT 72"),
            ("next_check_at", "TEXT"),
            ("entry_status", "TEXT NOT NULL DEFAULT 'PENDING_ENTRY'"),
            ("entry_filled_at", "TEXT"),
            ("entry_price", "REAL"),
            ("trade_closed_at", "TEXT"),
            ("rr_result", "REAL"),
            ("hold_hours", "REAL"),
            ("reasoning_summary", "TEXT"),
            ("full_response", "TEXT"),
            ("result_reason", "TEXT"),
            ("market_snapshot", "TEXT"),
            ("feature_snapshot", "TEXT"),
            ("setup_status", "TEXT"),
            ("lifecycle_status", "TEXT"),
            ("mae", "REAL"),
            ("mfe", "REAL"),
        ]:
            try:
                conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {definition}")
            except sqlite3.OperationalError:
                pass

        # Migrate old PENDING rows to lifecycle naming.
        try:
            conn.execute("UPDATE predictions SET result='PENDING_ENTRY' WHERE result='PENDING'")
            conn.execute("UPDATE predictions SET entry_status='PENDING_ENTRY' WHERE entry_status IS NULL OR entry_status='' ")
        except sqlite3.OperationalError:
            pass

        # Lightweight index for history/stats/learning/auto-check as the DB grows.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_user_id_id ON predictions(user_id, id DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_user_symbol_mode_id ON predictions(user_id, symbol, mode, id DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_result_next_check ON predictions(result, next_check_at)")

        # Migration/cleanup: right after deploy, also keep only the 5 most recent predictions per user
        # for both the visible group and the hidden-learning group, without waiting for the next save.
        hidden_a, hidden_b = HIDDEN_LEARNING_RESULTS
        conn.execute(
            """
            DELETE FROM predictions
            WHERE id IN (
                SELECT id FROM (
                    SELECT
                        id,
                        ROW_NUMBER() OVER (
                            PARTITION BY user_id,
                                CASE WHEN result IN (?, ?) THEN 1 ELSE 0 END
                            ORDER BY id DESC
                        ) AS keep_rank
                    FROM predictions
                    WHERE user_id IS NOT NULL
                ) ranked
                WHERE keep_rank > ?
            )
            """,
            (hidden_a, hidden_b, VISIBLE_PREDICTION_RETENTION_LIMIT),
        )
        conn.commit()
    _prediction_db_initialized = True


def prune_prediction_history(user_id: int | None) -> None:
    """Keep the DB lean: each user only keeps the 5 most recent visible trades.

    - /history only uses the visible group, so that group is kept at exactly the 5 newest rows.
    - NO_TRADE/REJECTED_PLAN are hidden learning records, not shown in /history; they're still
      limited separately so the DB doesn't grow unbounded over time.
    - The learning prompt pulls the most recent rows per PREDICTION_HISTORY_COUNT (default 3) for the matching user/symbol/mode.
    """
    if user_id is None:
        return

    hidden_a, hidden_b = HIDDEN_LEARNING_RESULTS
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            DELETE FROM predictions
            WHERE user_id=?
              AND result NOT IN (?, ?)
              AND id NOT IN (
                  SELECT id
                  FROM predictions
                  WHERE user_id=?
                    AND result NOT IN (?, ?)
                  ORDER BY id DESC
                  LIMIT ?
              )
            """,
            (user_id, hidden_a, hidden_b, user_id, hidden_a, hidden_b, VISIBLE_PREDICTION_RETENTION_LIMIT),
        )
        conn.execute(
            """
            DELETE FROM predictions
            WHERE user_id=?
              AND result IN (?, ?)
              AND id NOT IN (
                  SELECT id
                  FROM predictions
                  WHERE user_id=?
                    AND result IN (?, ?)
                  ORDER BY id DESC
                  LIMIT ?
              )
            """,
            (user_id, hidden_a, hidden_b, user_id, hidden_a, hidden_b, HIDDEN_LEARNING_RETENTION_LIMIT),
        )
        conn.commit()


def save_prediction(
    symbol: str,
    mode: str,
    direction: str,
    entry_low: float | None,
    entry_high: float | None,
    sl: float | None,
    tp1: float | None,
    tp2: float | None,
    market_snapshot: str | None,
    feature_snapshot: str | None,
    reasoning_summary: str | None,
    full_response: str | None,
    user_id: int | None = None,
    chat_id: int | None = None,
    setup_status: str | None = None,
) -> int:
    """setup_status is stored so a finished trade can be traced back to what the Planner itself
    labeled it at creation time (READY_TO_ENTER vs SETUP_WAITING_TRIGGER)."""
    now = utc_now()
    entry_wait = ENTRY_WAIT_HOURS.get(mode, 24)
    max_hold = TRADE_MAX_HOLD_HOURS.get(mode, 72)
    next_check = now + timedelta(hours=CHECK_INTERVAL_HOURS.get(mode, 1))

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions
                (user_id, chat_id, symbol, mode, created_at, check_after_hours, entry_wait_hours, max_hold_hours,
                 next_check_at, direction, entry_low, entry_high, sl, tp1, tp2,
                 entry_status, market_snapshot, feature_snapshot, reasoning_summary, full_response, result,
                 setup_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING_ENTRY', ?, ?, ?, ?, 'PENDING_ENTRY',
                    ?)
            """,
            (user_id, chat_id, symbol, mode, iso(now), CHECK_INTERVAL_HOURS.get(mode, 1), entry_wait, max_hold,
             iso(next_check), direction, entry_low, entry_high, sl, tp1, tp2,
             market_snapshot, feature_snapshot, reasoning_summary, full_response,
             setup_status),
        )
        prediction_id = cursor.lastrowid
        conn.commit()
    prune_prediction_history(user_id)
    return prediction_id






def _row_to_pred(row) -> dict:
    keys = [
        "id", "user_id", "chat_id", "symbol", "mode", "created_at",
        "entry_wait_hours", "max_hold_hours", "next_check_at", "direction",
        "entry_low", "entry_high", "sl", "tp1", "tp2", "entry_status",
        "entry_filled_at", "entry_price", "result"
    ]
    return dict(zip(keys, row))


def get_due_predictions(force: bool = False) -> list[dict]:
    """
    Get open predictions for auto-check.

    - force=False: only fetches predictions due per next_check_at; used by the periodic job.
    - force=True: fetches all PENDING_ENTRY/ENTRY_FILLED rows; used by /checknow to force an immediate check.
    """
    now_s = iso(utc_now())
    where_due = "" if force else "AND (next_check_at IS NULL OR next_check_at <= ?)"
    params = () if force else (now_s,)
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            f"""
            SELECT id, user_id, chat_id, symbol, mode, created_at,
                   entry_wait_hours, max_hold_hours, next_check_at, direction,
                   entry_low, entry_high, sl, tp1, tp2, entry_status,
                   entry_filled_at, entry_price, result
            FROM predictions
            WHERE result IN ('PENDING_ENTRY', 'ENTRY_FILLED')
              {where_due}
            ORDER BY id ASC
            LIMIT 200
            """,
            params,
        ).fetchall()
    return [_row_to_pred(row) for row in rows]


def schedule_next_check(pid: int, mode: str) -> None:
    next_at = utc_now() + timedelta(hours=CHECK_INTERVAL_HOURS.get(mode, 1))
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE predictions SET next_check_at=?, result_checked_at=? WHERE id=?",
            (iso(next_at), iso(utc_now()), pid),
        )
        conn.commit()


def mark_entry_filled(pid: int, entry_price: float, filled_at: datetime, mode: str) -> None:
    next_at = utc_now() + timedelta(hours=CHECK_INTERVAL_HOURS.get(mode, 1))
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            UPDATE predictions
            SET result='ENTRY_FILLED', entry_status='ENTRY_FILLED', entry_price=?,
                entry_filled_at=?, next_check_at=?, result_checked_at=?
            WHERE id=?
            """,
            (entry_price, iso(filled_at), iso(next_at), iso(utc_now()), pid),
        )
        conn.commit()


def _calc_rr(direction: str, entry_price: float | None, sl: float | None, outcome_price: float | None, result: str) -> float | None:
    if entry_price is None or sl is None or outcome_price is None:
        return None
    risk = abs(entry_price - sl)
    if risk <= 0:
        return None
    if result == "LOSS":
        return -1.0
    if direction == "LONG":
        return (outcome_price - entry_price) / risk
    if direction == "SHORT":
        return (entry_price - outcome_price) / risk
    return None


def update_prediction_result(
    pid: int,
    result: str,
    result_price: float,
    result_reason: str | None = None,
    trade_closed_at: datetime | None = None,
    entry_price: float | None = None,
    direction: str | None = None,
    sl: float | None = None,
    entry_filled_at: datetime | None = None,
) -> None:
    now = utc_now()
    closed = trade_closed_at or now
    hold_hours = None
    if entry_filled_at is not None:
        hold_hours = max(0.0, (closed - entry_filled_at).total_seconds() / 3600)
    rr_result = _calc_rr(direction or "", entry_price, sl, result_price, result)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            UPDATE predictions
            SET result=?, result_price=?, result_reason=?, result_checked_at=?,
                trade_closed_at=?, hold_hours=?, rr_result=?, next_check_at=NULL
            WHERE id=?
            """,
            (result, result_price, result_reason, iso(now), iso(closed), hold_hours, rr_result, pid),
        )
        conn.commit()








# ─── Auto WIN/LOSS check ──────────────────────────────────────────────────────

def get_current_price_raw(symbol: str) -> float | None:
    r = _binance_get_with_retry(
        f"{BINANCE_FUTURES_API_BASE}/fapi/v1/ticker/price", {"symbol": symbol}, max_retries=1, timeout=10
    )
    if r is None:
        return None
    try:
        return float(r.json()["price"])
    except Exception:
        return None


def parse_utc_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_vn_datetime(value: str | datetime | None) -> str:
    if not value:
        return "-"
    dt = value if isinstance(value, datetime) else parse_utc_datetime(value)
    if dt is None:
        return "-"
    local = dt.astimezone(VN_TZ)
    return local.strftime("%H:%M ngày %d/%m/%Y")


def get_binance_klines_since(
    symbol: str,
    interval: str,
    start: datetime,
    limit: int = 1000,
) -> pd.DataFrame | None:
    r = _binance_get_with_retry(
        BINANCE_API_URL,
        {
            "symbol": symbol,
            "interval": interval,
            "startTime": int(start.timestamp() * 1000),
            "limit": limit,
        },
        timeout=20,
    )
    if r is None:
        return None
    try:
        data = r.json()
        if not data:
            return None
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "count",
            "taker_buy_volume", "taker_buy_quote_volume", "ignore",
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df
    except Exception as exc:
        print(f"Historical Binance error {symbol} {interval}: {exc}", flush=True)
        return None


def get_funding_rate_context(symbol: str) -> dict | None:
    """Last 3 funding settlements (8h apart) for the USDT-M perpetual contract.

    Extreme positive funding = crowded/over-leveraged longs (squeeze-down risk); extreme negative =
    crowded shorts (squeeze-up risk). Returns None if the symbol has no futures market (spot-only
    coin) or the request fails — callers must treat this as optional context, not a hard dependency.
    """
    r = _binance_get_with_retry(
        f"{BINANCE_FUTURES_API_BASE}/fapi/v1/fundingRate",
        {"symbol": symbol, "limit": 3},
        max_retries=1, timeout=10,
    )
    if r is None:
        return None
    try:
        data = r.json()
        if not data:
            return None
        rates_pct = [float(x["fundingRate"]) * 100 for x in data]
        return {"latest_pct": rates_pct[-1], "history_pct": rates_pct}
    except Exception:
        return None


def get_open_interest_context(symbol: str) -> dict | None:
    """Open interest change over the last ~6h (1h buckets) for the USDT-M perpetual contract.

    Rising OI + rising price = fresh money confirming the trend; falling OI + rising price = short
    covering (weaker trend). Returns None if unavailable — optional context only.
    """
    r = _binance_get_with_retry(
        f"{BINANCE_FUTURES_API_BASE}/futures/data/openInterestHist",
        {"symbol": symbol, "period": "1h", "limit": 6},
        max_retries=1, timeout=10,
    )
    if r is None:
        return None
    try:
        data = r.json()
        if not data or len(data) < 2:
            return None
        first_oi = float(data[0]["sumOpenInterest"])
        last_oi = float(data[-1]["sumOpenInterest"])
        change_pct = ((last_oi - first_oi) / first_oi * 100) if first_oi else 0.0
        return {"current": last_oi, "change_pct_6h": change_pct}
    except Exception:
        return None


def get_long_short_ratio_context(symbol: str) -> dict | None:
    """Compare top-trader (large account) long/short positioning vs the broader retail crowd.

    Divergence between the two is a contrarian signal professional futures traders watch — e.g.
    top traders net short while the retail crowd is heavily long often precedes a squeeze down.
    Returns None if either leg is unavailable — optional context only.
    """
    top = _binance_get_with_retry(
        f"{BINANCE_FUTURES_API_BASE}/futures/data/topLongShortAccountRatio",
        {"symbol": symbol, "period": "1h", "limit": 1}, max_retries=1, timeout=10,
    )
    glob = _binance_get_with_retry(
        f"{BINANCE_FUTURES_API_BASE}/futures/data/globalLongShortAccountRatio",
        {"symbol": symbol, "period": "1h", "limit": 1}, max_retries=1, timeout=10,
    )
    if top is None or glob is None:
        return None
    try:
        top_data = top.json()
        glob_data = glob.json()
        if not top_data or not glob_data:
            return None
        top_ratio = float(top_data[-1]["longShortRatio"])
        global_ratio = float(glob_data[-1]["longShortRatio"])
        return {"top_ratio": top_ratio, "global_ratio": global_ratio}
    except Exception:
        return None


def build_futures_context_block(
    symbol: str, funding: dict | None, oi: dict | None, long_short: dict | None = None
) -> str | None:
    """Short objective text block for funding rate + open interest + long/short ratio; None if all
    three are unavailable."""
    if funding is None and oi is None and long_short is None:
        return None
    lines = [f"FUTURES_CONTEXT ({symbol} perpetual — bối cảnh khách quan, không phải tín hiệu bắt buộc):"]
    if funding is not None:
        history_text = " → ".join(f"{v:+.4f}%" for v in funding["history_pct"])
        lines.append(
            f"- Funding rate hiện tại: {funding['latest_pct']:+.4f}% (mỗi 8h); "
            f"3 lần gần nhất: {history_text}."
        )
    else:
        lines.append("- Funding rate: không có dữ liệu.")
    if oi is not None:
        lines.append(
            f"- Open interest thay đổi ~6h gần nhất: {oi['change_pct_6h']:+.2f}%."
        )
    else:
        lines.append("- Open interest: không có dữ liệu.")
    if long_short is not None:
        lines.append(
            f"- Long/Short ratio: top trader={long_short['top_ratio']:.2f}, "
            f"retail={long_short['global_ratio']:.2f}."
        )
    else:
        lines.append("- Long/Short ratio: không có dữ liệu.")
    return "\n".join(lines)


def get_btc_correlation_snapshot() -> dict | None:
    """BTC 4H/1D EMA alignment + recent price action, for context when analyzing a non-BTC symbol.

    Altcoins routinely get pulled by BTC moves within minutes, especially on lower timeframes; a
    trader always checks BTC before taking an alt trade. Reuses the same candle limits as the main
    pipeline so indicators have proper warm-up. Returns None on fetch failure — optional context.
    """
    try:
        btc_symbol = f"BTC{BINANCE_QUOTE_ASSET}"
        df_4h = load_timeframe_data(btc_symbol, "4h", 360)
        df_1d = load_timeframe_data(btc_symbol, "1d", 365)
    except Exception:
        return None
    if df_4h is None or df_4h.empty or df_1d is None or df_1d.empty:
        return None

    def _frame_summary(df: pd.DataFrame) -> dict | None:
        row = _analysis_row(df)
        if row is None:
            return None
        closed = _v50_closed_df(df)
        change_pct = None
        if closed is not None and len(closed) >= 6:
            first_c = _safe_float(closed.iloc[-6]["close"])
            last_c = _safe_float(closed.iloc[-1]["close"])
            if first_c:
                change_pct = (last_c - first_c) / first_c * 100
        return {
            "ema_7": _safe_float(row.get("ema_7")),
            "ema_25": _safe_float(row.get("ema_25")),
            "ema_50": _safe_float(row.get("ema_50")),
            "change_pct_6candles": change_pct,
            "rsi_12": _safe_float(row.get("rsi_12")),
        }

    return {"4h": _frame_summary(df_4h), "1d": _frame_summary(df_1d)}


def build_btc_correlation_block(btc_ctx: dict | None) -> str | None:
    """Short objective text block summarizing BTC's own trend, for correlation context."""
    if not btc_ctx:
        return None
    parts = ["BTC_CONTEXT (bối cảnh tương quan BTC — không áp đặt hướng cho altcoin):"]
    for label, key in (("4H", "4h"), ("1D", "1d")):
        info = btc_ctx.get(key)
        if not info:
            continue
        change = info.get("change_pct_6candles")
        change_text = f"{change:+.2f}%/6 nến" if change is not None else "N/A"
        parts.append(
            f"- BTC {label}: EMA7={fmt(info.get('ema_7'))}, EMA25={fmt(info.get('ema_25'))}, "
            f"EMA50={fmt(info.get('ema_50'))}, biến động gần đây={change_text}, "
            f"RSI12={fmt(info.get('rsi_12'), 1)}."
        )
    return "\n".join(parts) if len(parts) > 1 else None


def _btc_eth_strength_index(candle_count: int = 3) -> float | None:
    """Average % change of BTC's N most recently closed 1H candles minus the same average for
    ETH (N=3 by default: each candle's own (close-open)/open%, then averaged — not one span
    computed from the oldest open to the newest close). Positive = BTC relatively stronger over
    that window (fell less or rose more than ETH); negative = BTC relatively weaker.

    Independent of whichever symbol/mode is actually being analyzed, and deliberately never sent
    to the model — this is a Python-only number, added to a message only after the model has
    already decided, purely for the human reading the sent signal. Returns None on fetch failure
    so a Binance hiccup never blocks sending the actual trade plan.
    """
    def _avg_closed_pct_change(symbol: str) -> float | None:
        # Deliberately a single fast attempt with no retry/backoff (unlike get_binance_klines) —
        # this is best-effort supplementary context, not critical analysis data, so a slow/failing
        # Binance response must fail fast rather than hold up sending an already-decided,
        # time-sensitive trade plan.
        try:
            r = requests.get(
                BINANCE_API_URL,
                params={"symbol": symbol, "interval": "1h", "limit": candle_count + 1},
                timeout=5,
            )
            r.raise_for_status()
            data = r.json()
        except Exception:
            return None
        if not isinstance(data, list) or len(data) < candle_count + 1:
            return None
        closed = data[:-1]  # drop the still-forming last row
        pct_changes = []
        for row in closed[-candle_count:]:
            try:
                open_price, close_price = float(row[1]), float(row[4])
            except Exception:
                continue
            if open_price:
                pct_changes.append((close_price - open_price) / open_price * 100.0)
        if len(pct_changes) < candle_count:
            return None
        return sum(pct_changes) / len(pct_changes)

    btc_pct = _avg_closed_pct_change(f"BTC{BINANCE_QUOTE_ASSET}")
    eth_pct = _avg_closed_pct_change(f"ETH{BINANCE_QUOTE_ASSET}")
    if btc_pct is None or eth_pct is None:
        return None
    return btc_pct - eth_pct


def _insert_btc_strength_line(output: str, strength_index: float | None) -> str:
    """Insert 'Chỉ số sức mạnh BTC: +x.xx%' right below the Giá hiện tại line of a message that's
    actually being sent to the user. No-op if the index couldn't be computed."""
    if strength_index is None:
        return output
    text = output or ""
    strength_line = f"Chỉ số sức mạnh BTC: {strength_index:+.2f}%"
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.search(r"^\s*Giá\s+hiện\s+tại\s*:", line, flags=re.IGNORECASE):
            lines.insert(i + 1, strength_line)
            return "\n".join(lines)
    return text + "\n" + strength_line


def _interval_to_timedelta(interval: str) -> timedelta:
    """Duration of a Binance candle, used to fetch one extra candle back so overlapping candles aren't missed when creating a signal."""
    m = re.fullmatch(r"(\d+)([mhdw])", interval.strip().lower())
    if not m:
        return timedelta(minutes=15)
    n = int(m.group(1))
    unit = m.group(2)
    if unit == "m":
        return timedelta(minutes=n)
    if unit == "h":
        return timedelta(hours=n)
    if unit == "d":
        return timedelta(days=n)
    if unit == "w":
        return timedelta(weeks=n)
    return timedelta(minutes=15)


def _range_low_high(a: float | None, b: float | None) -> tuple[float | None, float | None]:
    if a is None or b is None:
        return None, None
    low = min(float(a), float(b))
    high = max(float(a), float(b))
    return low, high


def _entry_touched(direction: str, entry_low: float | None, entry_high: float | None, high: float, low: float) -> bool:
    low_zone, high_zone = _range_low_high(entry_low, entry_high)
    if low_zone is None or high_zone is None:
        return False
    # A candle touches the Entry zone when its [low, high] range intersects the Entry range.
    return low <= high_zone and high >= low_zone


def _price_in_entry_range(price: float | None, entry_low: float | None, entry_high: float | None) -> bool:
    if price is None:
        return False
    low_zone, high_zone = _range_low_high(entry_low, entry_high)
    if low_zone is None or high_zone is None:
        return False
    return low_zone <= float(price) <= high_zone


def _entry_price(direction: str, entry_low: float | None, entry_high: float | None, fill_price: float | None = None) -> float | None:
    if fill_price is not None:
        return float(fill_price)
    low_zone, high_zone = _range_low_high(entry_low, entry_high)
    if low_zone is None or high_zone is None:
        return None
    return (low_zone + high_zone) / 2


def _tp_sl_result(pred: dict, candles: pd.DataFrame) -> tuple[str, float | None, str, datetime | None]:
    direction, sl, tp1 = pred["direction"], pred["sl"], pred["tp1"]
    candle_label = "15M" if pred.get("mode") == "short" else "1H"
    if not sl or not tp1:
        return "UNKNOWN", None, "Thiếu SL hoặc TP1 nên không thể chấm kết quả.", None
    for _, row in candles.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        closed_at_ts = row["close_time"]
        closed_at = closed_at_ts.to_pydatetime() if hasattr(closed_at_ts, "to_pydatetime") else None
        text_time = str(row["close_time"])[:16]

        if direction == "LONG":
            hit_tp = high >= tp1
            hit_sl = low <= sl
            if hit_tp and hit_sl:
                return "AMBIGUOUS", close, f"TP1 và SL cùng bị chạm trong một nến {candle_label} lúc {text_time}.", closed_at
            if hit_tp:
                return "WIN", tp1, f"TP1 chạm trước SL lúc {text_time}.", closed_at
            if hit_sl:
                return "LOSS", sl, f"SL chạm trước TP1 lúc {text_time}.", closed_at
        elif direction == "SHORT":
            hit_tp = low <= tp1
            hit_sl = high >= sl
            if hit_tp and hit_sl:
                return "AMBIGUOUS", close, f"TP1 và SL cùng bị chạm trong một nến {candle_label} lúc {text_time}.", closed_at
            if hit_tp:
                return "WIN", tp1, f"TP1 chạm trước SL lúc {text_time}.", closed_at
            if hit_sl:
                return "LOSS", sl, f"SL chạm trước TP1 lúc {text_time}.", closed_at
    return "RUNNING", float(candles.iloc[-1]["close"]), "Đã khớp Entry nhưng chưa chạm TP1 hoặc SL.", None


def evaluate_prediction_lifecycle(
    pred: dict,
    candles: pd.DataFrame | None,
    current_price: float | None = None,
) -> dict:
    """
    Score the prediction's lifecycle.

    Key rules:
    - A signal created at T only considers data with close_time after T.
    - PENDING_ENTRY is filled if the current price is inside the Entry zone.
    - The Entry range is a price band: entry_low <= price <= entry_high, regardless of LONG/SHORT.
    - Once Entry has been filled, TP/SL are only evaluated from entry_filled_at onward.
    """
    now = utc_now()
    created = parse_utc_datetime(pred.get("created_at"))
    entry_filled_at = parse_utc_datetime(pred.get("entry_filled_at"))
    if created is None:
        return {"action": "skip", "reason": "Không đọc được thời gian tạo prediction."}

    status = pred.get("result") or pred.get("entry_status") or "PENDING_ENTRY"

    if status == "PENDING_ENTRY":
        entry_deadline = created + timedelta(hours=int(pred.get("entry_wait_hours") or 24))

        # Check the live price first so we don't miss the case where the current price is already inside the Entry zone.
        # Example: Entry 50000-50500, current price 50300 => ENTRY_FILLED immediately.
        if now <= entry_deadline and _price_in_entry_range(current_price, pred.get("entry_low"), pred.get("entry_high")):
            return {
                "action": "fill",
                "price": _entry_price(pred["direction"], pred.get("entry_low"), pred.get("entry_high"), current_price),
                "filled_at": now,
                "reason": f"Giá hiện tại {current_price} đang nằm trong vùng Entry.",
            }

        if candles is None or candles.empty:
            if now >= entry_deadline:
                return {
                    "action": "close",
                    "result": "NOT_FILLED",
                    "price": current_price,
                    "reason": f"Hết thời gian chờ Entry {pred.get('entry_wait_hours')}h nhưng không có dữ liệu nến để xác nhận giá đã chạm Entry.",
                    "closed_at": now,
                }
            return {"action": "reschedule", "reason": "Không có dữ liệu nến."}

        # The fetch may look back one extra candle to catch overlap, but only closed candles after the signal's creation time are considered.
        # Match on the candle's OPEN time, not its close: the fetch deliberately starts one interval
        # early, so filtering by close_time would keep the candle that was already running when the
        # signal was created. Its high/low include ticks from before the plan existed, which can mark
        # an Entry as filled — and then score SL/TP — on price action that predates the signal.
        pending_candles = candles[candles["timestamp"] >= pd.Timestamp(created)]
        # Do not fill Entry using a candle that closed after the entry-wait deadline.
        pending_candles = pending_candles[pending_candles["close_time"] <= pd.Timestamp(entry_deadline)]

        for _, row in pending_candles.iterrows():
            high = float(row["high"])
            low = float(row["low"])
            if _entry_touched(pred["direction"], pred.get("entry_low"), pred.get("entry_high"), high, low):
                filled_at_ts = row["close_time"]
                filled_at = filled_at_ts.to_pydatetime() if hasattr(filled_at_ts, "to_pydatetime") else now
                entry_price = _entry_price(pred["direction"], pred.get("entry_low"), pred.get("entry_high"))
                post = candles[candles["close_time"] >= row["close_time"]]
                result, price, reason, closed_at = _tp_sl_result({**pred, "entry_price": entry_price}, post)
                if result in ("WIN", "LOSS", "AMBIGUOUS"):
                    return {
                        "action": "close",
                        "result": result,
                        "price": price,
                        "reason": f"Entry khớp rồi {reason}",
                        "closed_at": closed_at or filled_at,
                        "entry_price": entry_price,
                        "entry_filled_at": filled_at,
                    }
                return {
                    "action": "fill",
                    "price": entry_price,
                    "filled_at": filled_at,
                    "reason": f"Entry đã khớp trong nến đóng lúc {str(row['close_time'])[:16]}.",
                }

        if now >= entry_deadline:
            fallback_price = current_price
            if fallback_price is None and candles is not None and not candles.empty:
                fallback_price = float(candles.iloc[-1]["close"])
            return {
                "action": "close",
                "result": "NOT_FILLED",
                "price": fallback_price,
                "reason": f"Hết thời gian chờ Entry {pred.get('entry_wait_hours')}h nhưng giá chưa chạm vùng Entry.",
                "closed_at": now,
            }
        return {"action": "reschedule", "reason": "Chưa chạm Entry, tiếp tục chờ."}

    if status == "ENTRY_FILLED":
        if entry_filled_at is None:
            return {"action": "reschedule", "reason": "Thiếu entry_filled_at."}
        if candles is None or candles.empty:
            return {"action": "reschedule", "reason": "Không có dữ liệu nến."}
        filled_candles = candles[candles["close_time"] > pd.Timestamp(entry_filled_at)]
        if filled_candles.empty:
            return {"action": "reschedule", "reason": "Chưa có nến đóng sau thời điểm khớp Entry."}
        result, price, reason, closed_at = _tp_sl_result(pred, filled_candles)
        if result in ("WIN", "LOSS", "AMBIGUOUS"):
            return {
                "action": "close",
                "result": result,
                "price": price,
                "reason": reason,
                "closed_at": closed_at or now,
                "entry_price": pred.get("entry_price"),
                "entry_filled_at": entry_filled_at,
            }
        hold_deadline = entry_filled_at + timedelta(hours=int(pred.get("max_hold_hours") or 72))
        if now >= hold_deadline:
            return {
                "action": "close",
                "result": "EXPIRED",
                "price": price or current_price,
                "reason": f"Đã khớp Entry nhưng quá thời gian giữ lệnh {pred.get('max_hold_hours')}h mà chưa chạm TP1/SL.",
                "closed_at": now,
                "entry_price": pred.get("entry_price"),
                "entry_filled_at": entry_filled_at,
            }
        return {"action": "reschedule", "reason": reason}

    return {"action": "skip", "reason": f"Trạng thái {status} không cần kiểm tra."}


def _calculate_mae_mfe(pred: dict, candles: pd.DataFrame | None, entry_price: float | None) -> tuple[float | None, float | None]:
    if candles is None or candles.empty or entry_price is None:
        return None, None
    try:
        highs = pd.to_numeric(candles["high"], errors="coerce")
        lows = pd.to_numeric(candles["low"], errors="coerce")
        direction = str(pred.get("direction") or "").upper()
        if direction == "LONG":
            mae = max(0.0, float(entry_price) - float(lows.min()))
            mfe = max(0.0, float(highs.max()) - float(entry_price))
        elif direction == "SHORT":
            mae = max(0.0, float(highs.max()) - float(entry_price))
            mfe = max(0.0, float(entry_price) - float(lows.min()))
        else:
            return None, None
        return mae, mfe
    except Exception:
        return None, None


def _update_prediction_lifecycle_metrics(prediction_id: int, lifecycle_status: str, mae: float | None = None, mfe: float | None = None) -> None:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "UPDATE predictions SET lifecycle_status=?, mae=COALESCE(?,mae), mfe=COALESCE(?,mfe) WHERE id=?",
                (lifecycle_status, mae, mfe, prediction_id),
            )
    except Exception:
        pass


def _compat_lifecycle_status(result: str | None, action: str | None = None) -> str:
    mapping = {
        "WIN": "TP1_HIT",
        "LOSS": "SL_HIT",
        "AMBIGUOUS": "AMBIGUOUS_TP_SL",
        "NOT_FILLED": "EXPIRED_NOT_FILLED",
        "EXPIRED": "EXPIRED_AFTER_ENTRY",
        "PENDING_ENTRY": "WAITING_TRIGGER",
        "ENTRY_FILLED": "ENTRY_FILLED",
    }
    if action == "fill":
        return "ENTRY_FILLED"
    return mapping.get(str(result or "").upper(), str(result or action or "SETUP_CREATED").upper())


async def auto_check_pending_predictions(force: bool = False) -> dict:
    """Check open predictions, only updating the DB and returning a summary.

    This function intentionally no longer creates a notification for the user/admin.
    Users who want to see results actively use /history, /stats, or /dashboard.
    """
    init_prediction_db()
    due = get_due_predictions(force=force)
    entry_filled_count = 0
    closed_count = 0
    rescheduled_count = 0
    skipped_count = 0

    check_label = "all active predictions" if force else "due predictions"
    print(f"[AUTO_CHECK] Checking {len(due)} {check_label} at {iso(utc_now())}", flush=True)

    for pred in due:
        try:
            start_dt = parse_utc_datetime(pred.get("entry_filled_at")) or parse_utc_datetime(pred.get("created_at"))
            if start_dt is None:
                skipped_count += 1
                continue
            result_interval = get_result_check_interval(pred.get("mode", "short"))
            fetch_start = start_dt - _interval_to_timedelta(result_interval)
            current_price = None
            if (pred.get("result") or pred.get("entry_status")) == "PENDING_ENTRY":
                current_price = await asyncio.to_thread(get_current_price_raw, pred["symbol"])
            candles = await asyncio.to_thread(get_binance_klines_since, pred["symbol"], result_interval, fetch_start)
            decision = evaluate_prediction_lifecycle(pred, candles, current_price=current_price)
            action = decision.get("action")

            if action == "fill":
                mark_entry_filled(pred["id"], decision["price"], decision["filled_at"], pred["mode"])
                _update_prediction_lifecycle_metrics(pred["id"], "ENTRY_FILLED")
                entry_filled_count += 1
                # No message is sent when Entry fills; it's only logged to Railway and saved to the DB.
                print(f"[AUTO_CHECK] #{pred['id']} ENTRY_FILLED {pred['symbol']} {decision.get('reason')}", flush=True)
                continue

            if action == "close":
                result = decision["result"]
                price = decision.get("price")
                if price is None:
                    price = await asyncio.to_thread(get_current_price_raw, pred["symbol"])
                if price is None:
                    schedule_next_check(pred["id"], pred["mode"])
                    rescheduled_count += 1
                    continue
                entry_price = decision.get("entry_price") or pred.get("entry_price")
                entry_filled_at = decision.get("entry_filled_at") or parse_utc_datetime(pred.get("entry_filled_at"))
                update_prediction_result(
                    pred["id"], result, float(price), decision.get("reason"),
                    trade_closed_at=decision.get("closed_at"), entry_price=entry_price,
                    direction=pred.get("direction"), sl=pred.get("sl"), entry_filled_at=entry_filled_at,
                )
                metric_candles = candles
                if entry_filled_at is not None and candles is not None and not candles.empty:
                    metric_candles = candles[candles["close_time"] >= pd.Timestamp(entry_filled_at)]
                mae, mfe = _calculate_mae_mfe(pred, metric_candles, entry_price)
                _update_prediction_lifecycle_metrics(pred["id"], _compat_lifecycle_status(result), mae, mfe)
                closed_count += 1
                print(
                    f"[AUTO_CHECK] #{pred['id']} CLOSED {pred['symbol']} {result} "
                    f"price={price} reason={decision.get('reason')}",
                    flush=True,
                )
                continue

            if action == "reschedule":
                schedule_next_check(pred["id"], pred["mode"])
                rescheduled_count += 1
                continue

            skipped_count += 1
        except Exception as exc:
            print(f"[AUTO_CHECK] #{pred.get('id')} ERROR {exc}", flush=True)
            skipped_count += 1

    return {
        "due_count": len(due),
        "force": force,
        "entry_filled_count": entry_filled_count,
        "closed_count": closed_count,
        "rescheduled_count": rescheduled_count,
        "skipped_count": skipped_count,
        # Old key kept so older code doesn't crash if it still references it, but it's always left empty.
        "admin_messages": [],
        "user_messages": [],
    }


# ─── Stats / History helpers ─────────────────────────────────────────────────

def build_prediction_where(
    symbol: str | None = None,
    user_id: int | None = None,
    include_rejected: bool = False,
) -> tuple[str, list]:
    clauses: list[str] = []
    params: list = []
    if symbol:
        normalized_symbol = resolve_binance_symbol(symbol)
        clauses.append("symbol=?")
        params.append(normalized_symbol)
    if user_id is not None:
        clauses.append("user_id=?")
        params.append(user_id)
    # REJECTED_PLAN and NO_TRADE are internal learning records, not shown in /history, /stats, /dashboard
    # so users/admins don't mistake them for real signals. get_recent_predictions() can still read
    # these records so the model can learn from validator errors or from times it should have stayed out.
    if not include_rejected:
        clauses.append("result NOT IN ('REJECTED_PLAN', 'NO_TRADE')")
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    return where, params


def format_scope_label(symbol: str | None = None, user_id: int | None = None) -> str:
    symbol_label = symbol.upper() if symbol else None
    if user_id is None:
        return f"{symbol_label}" if symbol_label else "Teopard"
    return f"của bạn - {symbol_label}" if symbol_label else "của bạn"


def format_stats(symbol: str | None = None, user_id: int | None = None) -> str:
    init_prediction_db()
    where, params = build_prediction_where(symbol=symbol, user_id=user_id)
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            f"SELECT result, direction, mode, rr_result FROM predictions {where}",
            params,
        ).fetchall()
    if not rows:
        return "Chưa có lịch sử dự đoán."
    total = len(rows)
    counts = {}
    for result, *_ in rows:
        counts[result] = counts.get(result, 0) + 1
    closed = [r for r in rows if r[0] in ("WIN", "LOSS")]
    wins = sum(1 for r in closed if r[0] == "WIN")
    losses = sum(1 for r in closed if r[0] == "LOSS")
    win_rate = wins / len(closed) * 100 if closed else 0
    rr_values = [r[3] for r in rows if r[3] is not None]
    avg_rr = sum(rr_values) / len(rr_values) if rr_values else 0
    title = f"📊 Thống kê {format_scope_label(symbol, user_id)}"
    return "\n".join([
        title,
        f"Tổng lệnh đã trade theo bot: {total}",
        f"WIN/LOSS: {wins}/{losses} | Win rate: {win_rate:.1f}%",
        f"PENDING_ENTRY: {counts.get('PENDING_ENTRY', 0)}",
        f"ENTRY_FILLED: {counts.get('ENTRY_FILLED', 0)}",
        f"NOT_FILLED: {counts.get('NOT_FILLED', 0)}",
        f"EXPIRED: {counts.get('EXPIRED', 0)}",
        f"AMBIGUOUS: {counts.get('AMBIGUOUS', 0)}",
        f"RR trung bình: {avg_rr:.2f}R" if rr_values else "RR trung bình: chưa có dữ liệu",
    ])


def format_history(symbol: str | None = None, limit: int = 5, user_id: int | None = None) -> str:
    init_prediction_db()
    limit = max(1, min(5, int(limit or 5)))
    where, params = build_prediction_where(symbol=symbol, user_id=user_id)
    params.append(limit)
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            f"""
            SELECT id, user_id, chat_id, symbol, mode, direction, entry_low, entry_high, sl, tp1, tp2,
                   result, result_price, created_at, result_reason
            FROM predictions
            {where}
            ORDER BY id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    if not rows:
        return "Chưa có lịch sử dự đoán."

    # /history shows a stable index number over a rolling window of the 5 most recent trades: oldest -> newest.
    # When a 6th trade is saved, the oldest one is pruned and the list stays #1..#5.
    # The DB id stays the same inside the database, but it isn't used as the display number for the user.
    rows = list(reversed(rows))

    # user_id=None is only used for admin, so admin sees which user each trade belongs to.
    is_admin_scope = user_id is None
    lines = [f"🧾 {limit} lệnh đã trade theo bot gần nhất {format_scope_label(symbol, user_id)}"]
    for display_idx, row in enumerate(rows, 1):
        pid, owner_user_id, owner_chat_id, sym, mode, direction, entry_low, entry_high, sl, tp1, tp2, result, result_price, created_at, result_reason = row
        mode_label = "SCALP" if mode == "short" else "SWING"
        created_label = format_vn_datetime(created_at) if created_at else "không rõ"
        owner_line = ""
        if is_admin_scope:
            owner_label = str(owner_user_id) if owner_user_id is not None else "không rõ"
            chat_label = str(owner_chat_id) if owner_chat_id is not None else "không rõ"
            owner_line = f"User ID: {owner_label} | Chat ID: {chat_label}\n"
        reason_line = ""
        if result == "REJECTED_PLAN" and result_reason:
            short_reason = str(result_reason)[:260] + ("..." if len(str(result_reason)) > 260 else "")
            reason_line = f"\nLý do không auto-check: {short_reason}"
        lines.append(
            f"#{display_idx} {sym} {mode_label} {direction} → {result}\n"
            f"{owner_line}"
            f"Thời gian phân tích: {created_label}\n"
            f"Entry {fmt(entry_low)}–{fmt(entry_high)} | SL {fmt(sl)} | TP1 {fmt(tp1)} | TP2 {fmt(tp2)}"
            + (f" | Giá check {fmt(result_price)}" if result_price else "")
            + reason_line
        )
    return "\n\n".join(lines)


def clear_prediction_history() -> dict:
    """Wipe every history/tracking table (predictions, evaluation_cases, auto_scan signal/log/trend
    state) so stats start fresh from this point. Never touches whitelist, allowed_symbols, or the
    user's current auto_scan_settings (on/off, chosen symbol) — those are configuration, not history."""
    init_prediction_db()
    init_auto_scan_db()
    with sqlite3.connect(DB_PATH) as conn:
        visible_count = int(conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE result NOT IN ('REJECTED_PLAN', 'NO_TRADE')"
        ).fetchone()[0])
        total_prediction_count = int(conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
        conn.execute("DELETE FROM predictions")
        conn.execute("DELETE FROM auto_scan_signals")
        conn.execute("DELETE FROM auto_scan_logs")
        try:
            conn.execute("DELETE FROM auto_scan_trend_state")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("DELETE FROM analysis_snapshots")
        except sqlite3.OperationalError:
            pass
        for table in ("predictions", "auto_scan_signals", "auto_scan_logs", "analysis_snapshots"):
            try:
                conn.execute("DELETE FROM sqlite_sequence WHERE name=?", (table,))
            except sqlite3.Error:
                pass
        conn.commit()
    evaluation_count = clear_evaluation_data()
    return {
        "visible_count": visible_count,
        "total_prediction_count": total_prediction_count,
        "evaluation_count": evaluation_count,
    }


# ─── Binance + Indicators ─────────────────────────────────────────────────────

def get_binance_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame | None:
    r = _binance_get_with_retry(
        BINANCE_API_URL, {"symbol": symbol, "interval": interval, "limit": limit}, timeout=20
    )
    if r is None:
        return None
    try:
        data = r.json()
        if not data:
            return None
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "count",
            "taker_buy_volume", "taker_buy_quote_volume", "ignore",
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume",
                    "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]:
            df[col] = df[col].astype(float)
        return df
    except Exception as exc:
        print(f"Lỗi Binance {symbol} {interval}: {exc}")
        return None


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    return data.ewm(span=period, adjust=False, min_periods=period).mean()


def calculate_rsi(data: pd.Series, period: int) -> pd.Series:
    delta = data.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = pd.Series(np.nan, index=data.index, dtype="float64")
    avg_loss = pd.Series(np.nan, index=data.index, dtype="float64")
    if len(data) <= period:
        return pd.Series(np.nan, index=data.index, dtype="float64")
    avg_gain.iloc[period] = gain.iloc[1: period + 1].mean()
    avg_loss.iloc[period] = loss.iloc[1: period + 1].mean()
    for i in range(period + 1, len(data)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period
    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100)
    rsi = rsi.where(avg_gain != 0, 0)
    flat = (avg_gain == 0) & (avg_loss == 0)
    rsi = rsi.where(~flat, 50)
    return rsi


def calculate_macd(data: pd.Series, fast=12, slow=26, signal=9):
    ema_fast    = calculate_ema(data, fast)
    ema_slow    = calculate_ema(data, slow)
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line, signal_line, macd_line - signal_line


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    # Wilder's smoothing (RMA), matching TradingView/Binance chart ATR — not a plain SMA of TR.
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ADX: objective trend-strength cross-check for the qualitative continuation-vs-chop
    reading already done in the prompts. High ADX = trending (structure-break continuation more
    reliable); low ADX = ranging (breakouts more prone to fail). Direction is NOT read from this —
    only strength; +DI/-DI are intermediate and not exposed."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_high, prev_low, prev_close = high.shift(1), low.shift(1), close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    tr = pd.concat([
        (high - low), (high - prev_close).abs(), (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    smoothed_tr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / smoothed_tr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / smoothed_tr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def calculate_ichimoku(
    df: pd.DataFrame, tenkan_period: int = 9, kijun_period: int = 26,
    senkou_b_period: int = 52, displacement: int = 26,
):
    """Standard Ichimoku Kinko Hyo (Hosoda), fixed periods 9/26/52/26 — same on every charting
    platform, no tunable sensitivity parameter. Senkou Span A/B are shifted forward by
    `displacement` so the returned value at each row is the cloud edge actually overlapping that
    candle on a real chart, not the raw same-day computation."""
    high, low = df["high"], df["low"]
    tenkan = (high.rolling(tenkan_period).max() + low.rolling(tenkan_period).min()) / 2
    kijun = (high.rolling(kijun_period).max() + low.rolling(kijun_period).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(displacement)
    senkou_b = ((high.rolling(senkou_b_period).max() + low.rolling(senkou_b_period).min()) / 2).shift(displacement)
    return tenkan, kijun, senkou_a, senkou_b


def add_indicators(df: pd.DataFrame | None) -> pd.DataFrame | None:
    # No artificial minimum history here beyond what each indicator itself needs to produce a real
    # (non-NaN) value — every calculate_* function below already uses min_periods=<its own period>,
    # so dropna() at the end naturally trims exactly the leading rows that don't have enough history
    # yet. A coin too new to clear an indicator's own min_periods ends up with a short or empty
    # result, which the timeframe-omission logic downstream (see _v50_closed_df, _analysis_row,
    # _missing_critical_timeframes) already handles — Python must not additionally reject a coin's
    # real, available history just because it is short.
    if df is None:
        return None
    r = df.copy()
    r["ema_7"],  r["ema_25"], r["ema_50"] = (
        calculate_ema(r["close"], 7),
        calculate_ema(r["close"], 25),
        calculate_ema(r["close"], 50),
    )
    r["rsi_6"], r["rsi_12"], r["rsi_24"] = (
        calculate_rsi(r["close"], 6),
        calculate_rsi(r["close"], 12),
        calculate_rsi(r["close"], 24),
    )
    r["macd_line"], r["macd_signal"], r["macd_hist"] = calculate_macd(r["close"])
    r["atr_14"] = calculate_atr(r, 14)
    r["atr_pct"] = (r["atr_14"] / r["close"]) * 100
    r["adx_14"] = calculate_adx(r, 14)
    # Baseline excludes the current candle so a real spike isn't diluted by itself.
    r["vol_ma20"]  = r["volume"].shift(1).rolling(20).mean()
    r["vol_ratio"] = r["volume"] / r["vol_ma20"]
    # A fully flat candle run (zero true range) or a zero-volume baseline divides to inf, which
    # dropna() alone doesn't catch — turn those into NaN too so they're dropped like any other
    # incomplete row instead of leaking a literal "inf" into the LLM prompt.
    r = r.replace([np.inf, -np.inf], np.nan)
    return r.dropna().reset_index(drop=True)


# ─── Feature engineering: ATR / Structure / Fibonacci / Liquidity ────────────

def _safe_float(v, default: float | None = None) -> float | None:
    try:
        if v is None or pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _last_close_from_data(timeframe_data: dict[str, pd.DataFrame | None]) -> float | None:
    for df in timeframe_data.values():
        if df is not None and not df.empty:
            return _safe_float(df.iloc[-1]["close"])
    return None
















def _analysis_row(df: pd.DataFrame | None):
    """
    Use the most recently closed candle to read indicators/volume.

    Binance usually also returns the currently running candle; its volume is very low right after
    the candle opens, which can make the model mistakenly read it as weak liquidity and choose NO TRADE.
    So indicator/regime/snapshot logic uses candle -2 whenever there's enough data.
    """
    if df is None or df.empty or len(df) < 2:
        return None
    return df.iloc[-2]






def _pct_delta(new_value, old_value) -> float | None:
    new_num = _safe_float(new_value)
    old_num = _safe_float(old_value)
    if new_num is None or old_num is None or abs(old_num) <= 1e-12:
        return None
    return (new_num - old_num) / abs(old_num) * 100.0








def _taker_buy_ratio(row) -> float | None:
    if row is None:
        return None
    volume = _safe_float(row.get("volume"))
    taker = _safe_float(row.get("taker_buy_volume"))
    if volume is None or taker is None or volume <= 0:
        return None
    return taker / volume * 100.0


def _candle_delta(row) -> float:
    """Net taker aggression for one candle: taker buy volume minus taker sell volume.
    Missing data contributes 0 so a running CVD sum doesn't break on a gap."""
    volume = _safe_float(row.get("volume"))
    taker_buy = _safe_float(row.get("taker_buy_volume"))
    if volume is None or taker_buy is None:
        return 0.0
    return 2 * taker_buy - volume










def _live_candle_progress(row) -> float | None:
    if row is None:
        return None
    start = row.get("timestamp")
    end = row.get("close_time")
    try:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        now_ts = pd.Timestamp.now(tz="UTC")
        duration = max((end_ts - start_ts).total_seconds(), 1.0)
        return max(0.0, min(1.0, (now_ts - start_ts).total_seconds() / duration))
    except Exception:
        # Genuinely unknown, not "just opened" — a fabricated 0.0% would look like a real computed
        # value once formatted, when what actually happened is the timestamp couldn't be read at all.
        return None






# ─── Format helpers ───────────────────────────────────────────────────────────

def fmt(v, decimals: int | None = None) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    # An explicit decimals request (RSI, CVD, vol_ratio, ADX...) is always honored exactly — these
    # aren't prices, so a fixed decimal count is the actual intent, not the number's magnitude.
    if decimals is not None:
        return f"{v:,.{decimals}f}"
    # No decimals given: adaptive precision for price display, where magnitude is what actually
    # determines meaningful precision — a coin priced at 0.00001234 needs 8 decimals to be
    # meaningful, one priced at 65,000 only needs 2.
    if abs(v) >= 100:
        return f"{v:,.2f}"
    if abs(v) >= 1:
        return f"{v:,.4f}"
    return f"{v:,.8f}"


def macd_momentum_text(macd_hist: float | None, decimals: int = 4) -> str:
    """Describe the MACD histogram in plain wording so the raw `Hist` jargon doesn't leak into the output."""
    if macd_hist is None or (isinstance(macd_hist, float) and np.isnan(macd_hist)):
        return "động lượng MACD N/A"
    value = fmt(macd_hist, decimals)
    if macd_hist > 0:
        return f"động lượng MACD dương {value}"
    if macd_hist < 0:
        return f"động lượng MACD âm {value}"
    return "động lượng MACD trung tính 0"




# ─── Fear & Greed ─────────────────────────────────────────────────────────────

def build_market_snapshot(
    timeframe_data: dict[str, pd.DataFrame | None],
    fear_greed_info: str,
    current_price_str: str,
) -> str:
    lines = [current_price_str]
    for label, df in timeframe_data.items():
        if df is None or df.empty:
            lines.append(f"{label}: no data")
            continue

        last = _analysis_row(df)
        if last is None:
            lines.append(f"{label}: no data")
            continue
        e7  = _safe_float(last.get("ema_7"))
        e25 = _safe_float(last.get("ema_25"))
        e50 = _safe_float(last.get("ema_50"))

        lines.append(
            f"{label}: close={fmt(_safe_float(last.get('close')))}, "
            f"EMA(7={fmt(e7)},25={fmt(e25)},50={fmt(e50)}), "
            f"RSI6={fmt(_safe_float(last.get('rsi_6')),1)}/RSI12={fmt(_safe_float(last.get('rsi_12')),1)}/RSI24={fmt(_safe_float(last.get('rsi_24')),1)}, "
            f"{macd_momentum_text(_safe_float(last.get('macd_hist')))}, "
            f"vol={fmt(_safe_float(last.get('vol_ratio')), 2)}x"
        )

    return " | ".join(lines)


def get_current_price_str(symbol: str) -> tuple[str, float | None]:
    price = get_current_price_raw(symbol)
    if price is None:
        return "Giá hiện tại: không có dữ liệu", None
    return f"Giá hiện tại: {fmt(price)} {BINANCE_QUOTE_ASSET}", price






def _truncate_text(text: str | None, limit: int = 600) -> str | None:
    if not text:
        return None
    text = str(text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


# ─── Select current AI provider/API key/model (multi-provider config) ───



def get_ai_model_name() -> str:
    return PLANNER_MODEL


def get_ai_provider_label() -> str:
    return "openrouter"


def ensure_ai_config() -> None:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Missing OpenRouter API key. Set OPENROUTER_API_KEY in Railway variables.")


def _openrouter_create_once(
    system: str | None,
    messages: list,
    model: str,
    max_tokens: int,
    timeout: int | None = None,
    temperature: float | None = None,
    response_format: dict | None = None,
    reasoning_effort: str | None = None,
) -> dict:
    """Call OpenRouter's OpenAI-compatible Chat Completions API for the Planner call."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Missing OPENROUTER_API_KEY. Set it in Railway variables.")

    payload_messages = []
    if system:
        payload_messages.append({"role": "system", "content": system})
    payload_messages.extend(messages or [])

    effective_model = (model or "").strip()
    if not effective_model:
        raise RuntimeError("Missing OpenRouter model id.")

    payload: dict = {
        "model": effective_model,
        "messages": payload_messages,
        "max_tokens": int(max_tokens),
        # Route to the cheapest provider serving this model, price above all else. This disables
        # OpenRouter's default load-balancing (which weights by price but also mixes in reliability),
        # so a request can land on a slower/less-reliable provider if it's the cheapest at that moment.
        "provider": {"sort": "price"},
    }
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if response_format:
        payload["response_format"] = response_format

    effort_norm = (reasoning_effort or "").strip().lower()
    # OpenRouter's unified reasoning param (docs: openrouter.ai/docs/use-cases/reasoning-tokens)
    # natively supports 7 tiers: none/minimal/low/medium/high/xhigh/max. Forward the configured
    # value as-is instead of collapsing it — "max" must reach the API as "max", not get downgraded.
    valid_tiers = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
    if effort_norm in {"", "off", "false", "0", "disabled"}:
        payload["reasoning"] = {"effort": "none"}
    elif effort_norm in valid_tiers:
        payload["reasoning"] = {"effort": effort_norm}
    else:
        payload["reasoning"] = {"effort": "high"}

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    request_timeout = int(timeout or PLANNER_TIMEOUT_SECONDS)
    r = requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=request_timeout,
    )
    try:
        r.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"OpenRouter API error: {r.status_code} - {r.text[:1000]}") from exc

    data = r.json()
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning_content = message.get("reasoning") or message.get("reasoning_content") or ""
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    if isinstance(reasoning_content, list):
        reasoning_content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in reasoning_content
        )
    return {
        "text": str(content or ""),
        "reasoning_text": str(reasoning_content or ""),
        "stop_reason": choice.get("finish_reason"),
        "usage": data.get("usage"),
        "model": effective_model,
    }


def llm_create_once(
    system: str | None,
    messages: list,
    max_tokens: int,
    timeout: int,
    reasoning_effort: str | None = None,
) -> dict:
    ensure_ai_config()
    effective_reasoning_effort = (
        reasoning_effort if reasoning_effort is not None else PLANNER_REASONING_EFFORT
    )
    return _openrouter_create_once(
        system, messages, model=PLANNER_MODEL, max_tokens=max_tokens, timeout=timeout,
        reasoning_effort=effective_reasoning_effort,
    )


def _is_length_stop(stop_reason) -> bool:
    if stop_reason is None:
        return False
    return str(stop_reason).lower() in ("max_tokens", "length", "token_limit", "output_limit")


def _is_transient_llm_error(exc: Exception) -> bool:
    text = str(exc).lower()
    transient_markers = (
        "timeout", "timed out", "read timed out", "connection aborted",
        "connection reset", "temporarily unavailable", "bad gateway",
        "gateway timeout", "502", "503", "504",
        "empty final content", "retryable",
    )
    return isinstance(exc, requests.exceptions.RequestException) or any(m in text for m in transient_markers)


def create_with_continuation(
    *,
    system: str | None,
    messages: list,
    max_tokens: int = PLANNER_MAX_OUTPUT_TOKENS,
    timeout: int = PLANNER_TIMEOUT_SECONDS,
    allow_continuation: bool = True,
    reasoning_effort: str | None = None,
    call_type: str = "main",
) -> str:
    """
    Call the current model; if the provider reports a max-token cutoff, call again to continue the output.
    Includes retry for transient network/timeout errors from the AI provider.
    Python never edits the strategic content, it only asks the model to continue the cut-off part.
    """
    convo = list(messages)
    full_text = ""
    max_attempts = PLANNER_MAX_CONTINUATIONS + 1 if allow_continuation else 1
    retry_count = max(0, PLANNER_API_RETRIES)
    if call_type in ("main", "main_json"):
        # Don't let the old PLANNER_API_RETRIES=2/3 Railway variable make manual analysis hang for 9-20 minutes.
        retry_count = min(retry_count, max(0, PLANNER_RETRY_LIMIT))
    elif call_type == "summary":
        # Summary is just secondary metadata; not worth making the user wait longer for a retry.
        retry_count = 0

    for attempt in range(max_attempts):
        result = None
        last_exc: Exception | None = None
        for retry_idx in range(retry_count + 1):
            effective_timeout = timeout
            effective_reasoning_effort = reasoning_effort
            if retry_idx > 0 and call_type in ("main", "main_json"):
                effective_timeout = max(30, min(timeout, PLANNER_RETRY_TIMEOUT_SECONDS))
                effective_reasoning_effort = PLANNER_RETRY_REASONING_EFFORT or "max"
            try:
                effort_for_log = effective_reasoning_effort or PLANNER_REASONING_EFFORT or "max"
                print(
                    f"[LLM_CALL] call_type={call_type} provider={get_ai_provider_label()} "
                    f"model={get_ai_model_name()} attempt={attempt + 1} try={retry_idx + 1}/{retry_count + 1} "
                    f"timeout={effective_timeout}s max_tokens={max_tokens} "
                    f"effort={effort_for_log or 'default'}",
                    flush=True,
                )
                result = llm_create_once(
                    system,
                    convo,
                    max_tokens=max_tokens,
                    timeout=effective_timeout,
                    reasoning_effort=effective_reasoning_effort,
                )
                if not (result.get("text") or "").strip() and not full_text.strip() and not _is_length_stop(result.get("stop_reason")):
                    # Provider returned 200 OK with a genuinely empty final answer (not a length
                    # cutoff to continue from) — observed live: a couple of calls finished in ~24s
                    # (far faster than a normal analysis) with nothing usable, silently burning the
                    # whole scan as a parse error. _is_transient_llm_error already had a marker for
                    # exactly this ("empty final content") but nothing ever raised it — wire it up
                    # so this routes through the same retry path as a network failure.
                    raise RuntimeError(f"Empty final content from provider (stop_reason={result.get('stop_reason')}).")
                break
            except Exception as exc:
                last_exc = exc
                if retry_idx >= retry_count or not _is_transient_llm_error(exc):
                    raise
                try:
                    print(
                        f"[LLM_RETRY] call_type={call_type} provider={get_ai_provider_label()} "
                        f"model={get_ai_model_name()} attempt={attempt + 1} retry={retry_idx + 1}/{retry_count} "
                        f"error={exc}",
                        flush=True,
                    )
                except Exception:
                    pass
                try:
                    import time
                    time.sleep(max(0.0, PLANNER_RETRY_SLEEP_SECONDS) * (retry_idx + 1))
                except Exception:
                    pass
        if result is None:
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("LLM call failed without response.")

        chunk = result.get("text") or ""
        full_text += chunk
        stop_reason = result.get("stop_reason")
        try:
            print(
                f"[LLM_RESPONSE] call_type={call_type} provider={get_ai_provider_label()} model={get_ai_model_name()} "
                f"effort={result.get('effort')} attempt={attempt + 1} stop_reason={stop_reason} usage={result.get('usage')}",
                flush=True,
            )
        except Exception:
            pass
        if not _is_length_stop(stop_reason):
            break
        if not allow_continuation:
            print(
                "[LLM_LENGTH_NO_CONTINUE] Model trả stop_reason=length nhưng call này không continuation.",
                flush=True,
            )
            break
        print("[LLM_TRUNCATED] Model trả stop_reason=length, gọi tiếp để nối phần còn lại...", flush=True)
        convo = convo + [
            {"role": "assistant", "content": chunk},
            {
                "role": "user",
                "content": (
                    "Tiếp tục viết nốt phần còn lại ngay từ chỗ bị ngắt, "
                    "không lặp lại nội dung đã viết, không giải thích gì thêm."
                ),
            },
        ]
    return full_text.strip()


def build_local_reasoning_summary(full_response: str, limit: int = 420) -> str:
    """Build a short metadata summary from Activation/Risk, without needing the public Reason section."""
    text = sanitize_user_output(full_response or "").strip()
    if not text:
        return ""
    parts: list[str] = []
    for pattern in (
        r"(?:^|\n)\s*Kích\s*hoạt\s*:\s*(.*?)(?=\n|\Z)",
        r"(?:^|\n)\s*⚠️\s*Rủi\s*ro\s*:\s*(.*?)(?=\n\s*\[\[TEOPARD_|\Z)",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            value = re.sub(r"\s+", " ", match.group(1)).strip(" -")
            if value:
                parts.append(value)
    summary = " | ".join(parts) if parts else text
    summary = re.sub(r"\s+", " ", summary).strip()
    return _truncate_text(summary, limit)


def _extract_json_object(text: str) -> dict | None:
    """Extract a JSON object from the model output, even if the model wrapped it in ```json."""
    if not text:
        return None
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _num_or_none(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        if isinstance(value, str):
            value = value.replace(",", "").strip()
        return float(value)
    except Exception:
        return None


def _extract_legacy_confidence(output: str | None) -> float | None:
    """Compatibility parser: accepts old confidence labels as well as the new Signal Score."""
    text = output or ""
    patterns = [
        r"(?:Điểm\s+tín\s+hiệu|Diem\s+tin\s+hieu|Signal\s+score|Độ\s+chắc\s+chắn|Điểm\s+chắc\s+chắn|Điểm\s+tin\s+cậy\s+AI)\s*:\s*([0-9]+(?:\.[0-9]+)?)(?:\s*(?:%|/\s*100))?",
        r"QUYẾT\s+ĐỊNH[:\s]+(?:LONG|SHORT|NO[_\s-]?TRADE|KHÔNG\s+VÀO\s+LỆNH|KHONG\s+VAO\s+LENH)\s*[—\-]\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        r"(?:📈|📉)?\s*(?:LONG|SHORT|NO[_\s-]?TRADE)\s*[—\-]\s*([0-9]+(?:\.[0-9]+)?)\s*%",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            try:
                return min(max(float(m.group(1)), 0.0), 100.0)
            except Exception:
                pass
    return None


# ─── Parse prediction from output ───

def parse_prediction_from_output(output: str) -> dict:
    def find_price(patterns: list[str], text: str | None = None) -> float | None:
        haystack = output if text is None else text
        for pat in patterns:
            m = re.search(pat, haystack, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1).replace(",", ""))
                except Exception:
                    pass
        return None

    # Direction: prefers the QUYET DINH (DECISION) line, falls back to emoji
    direction = "WAIT"
    # [*_]* tolerates the model wrapping the decision in markdown emphasis, e.g. "QUYẾT ĐỊNH: **SHORT**" —
    # without it the value never matches at all and direction silently falls through to "WAIT".
    m = re.search(r"QUYẾT ĐỊNH[:\s]+[*_]*(LONG|SHORT|NO[_\s-]?TRADE|KHÔNG\s+VÀO\s+LỆNH|KHONG\s+VAO\s+LENH)", output, re.IGNORECASE)
    if m:
        raw_direction = m.group(1).upper().replace("-", "_").replace(" ", "_")
        direction = "NO_TRADE" if raw_direction in ("NO_TRADE", "NO__TRADE", "KHÔNG_VÀO_LỆNH", "KHONG_VAO_LENH") else raw_direction
    elif re.search(r"📈\s*LONG", output):
        direction = "LONG"
    elif re.search(r"📉\s*SHORT", output):
        direction = "SHORT"

    selected_output = output
    if direction in ("LONG", "SHORT"):
        section_match = re.search(
            rf"(?m)^\s*(?:📈|📉)?\s*{direction}\s*[—\-]",
            output,
            re.IGNORECASE,
        )
        if section_match:
            selected_output = output[section_match.start():]
            other_direction = "SHORT" if direction == "LONG" else "LONG"
            next_match = re.search(
                rf"(?m)^\s*(?:📈|📉)?\s*{other_direction}\s*[—\-]",
                selected_output[1:],
                re.IGNORECASE,
            )
            risk_match = re.search(r"\n\s*(?:⚠️|📊|Lưu ý|Rủi ro)", selected_output[1:], re.IGNORECASE)
            cut_points = [
                match.start() + 1
                for match in [next_match, risk_match]
                if match is not None
            ]
            if cut_points:
                selected_output = selected_output[:min(cut_points)]

    # Entry - can be a range like "95,000-95,500" or a single value like "95,000"
    entry_low = entry_high = None
    # Accept every separator a model realistically puts between the two Entry bounds. Matching only
    # "-" and "–" meant an em dash or the word "đến" silently dropped the upper bound, collapsing the
    # Entry zone to a single price that the tracker then almost never sees touched.
    em = re.search(
        r"Entry[:\s]+[*_]*\s*([0-9,\.]+)(?:\s*(?:[-–—‒−~]|đến|tới|to)\s*[*_]*\s*([0-9,\.]+))?",
        selected_output,
        re.IGNORECASE,
    )
    if em:
        try:
            entry_low  = float(em.group(1).replace(",", ""))
            entry_high = float(em.group(2).replace(",", "")) if em.group(2) else entry_low
        except Exception:
            pass

    # [*_]* here too — same markdown-emphasis gap as Entry/direction/status; a bolded "SL: **64,050**"
    # would otherwise silently parse as no SL at all and get the whole plan rejected for a missing field.
    sl  = find_price([r"SL[:\s]+[*_]*\s*([0-9,\.]+)"], selected_output)
    tp1 = find_price([r"TP1[:\s]+[*_]*\s*([0-9,\.]+)"], selected_output)
    tp2 = find_price([r"TP2[:\s]+[*_]*\s*([0-9,\.]+)"], selected_output)

    setup_strength = None
    setup_match = re.search(
        r"(?:Độ\s+mạnh\s+setup|Chất\s+lượng\s+kế\s+hoạch)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*(?:/\s*100)?",
        output,
        flags=re.IGNORECASE,
    )
    if setup_match:
        try:
            setup_strength = min(max(float(setup_match.group(1)), 0.0), 100.0)
        except Exception:
            setup_strength = None

    signal_score = _extract_legacy_confidence(output)
    confidence = signal_score

    return {
        "direction":  direction,
        "signal_score": signal_score,
        "confidence": confidence,
        "setup_strength": setup_strength,
        "entry_low":  entry_low,
        "entry_high": entry_high,
        "sl":         sl,
        "tp1":        tp1,
        "tp2":        tp2,
    }


# ─── Trade-plan guard before saving to auto-check ───


def _validate_actionable_trade_plan(
    pred: dict,
    timeframe_data: dict[str, pd.DataFrame | None],
    mode: str,
    current_price: float | None,
    output: str | None = None,
) -> list[str]:
    """Validate only technical completeness; never score market quality."""
    direction = (pred.get("direction") or "").upper()
    if direction not in ("LONG", "SHORT"):
        return []

    errors: list[str] = []
    status = _extract_setup_status(output)
    if status not in {"READY_TO_ENTER", "SETUP_WAITING_TRIGGER"}:
        errors.append("Planner thiếu hoặc trả sai nhãn Trạng thái bắt buộc.")

    required = {
        "entry_low": "Entry thấp",
        "entry_high": "Entry cao",
        "sl": "SL",
        "tp1": "TP1",
    }
    values: dict[str, float] = {}
    for key, label in required.items():
        value = _num_or_none(pred.get(key))
        if value is None or not math.isfinite(float(value)):
            errors.append(f"Không đọc được {label} hợp lệ.")
        else:
            values[key] = float(value)

    # Deliberately nothing beyond this point. Whether the levels make sense as a trade — SL on the
    # right side, TP worth taking, structure sound — is entirely the Planner's own judgment call, not
    # Python's. Python only confirms it could READ the model's decision well enough to store and
    # track it; it never judges the decision. _range_low_high() already sorts the Entry bounds, so
    # even a reversed Entry range stores and tracks correctly without Python second-guessing the model.
    return errors


async def _repair_planner_format(
    system_prompt: str,
    planner_clean: str,
    guard_errors: list[str],
    timeframe_data: dict[str, pd.DataFrame | None],
    mode: str,
    current_price: float | None,
) -> tuple[str, dict, list[str]]:
    """One retry asking Planner to fix ONLY the flagged output-format issues (missing/malformed
    Entry/SL/TP1 number, wrong status label), reusing its existing analysis/evidence instead of
    re-running the full analysis. Guard failures are almost always pure formatting slips,
    not market-quality judgment — discarding an already-completed analysis over a technicality
    is pure waste.

    Returns (repaired_planner_clean, repaired_pred, remaining_errors) — remaining_errors is empty
    on success. On any failure to even get a response, returns the original unchanged.
    """
    errors_text = "\n".join(f"- {e}" for e in guard_errors)
    repair_prompt = (
        "Plan bên dưới đã phân tích xong nhưng phần OUTPUT PUBLIC bị lỗi định dạng kỹ thuật, không phải lỗi phán đoán thị trường.\n"
        f"Lỗi cụ thể cần sửa:\n{errors_text}\n\n"
        "Giữ nguyên toàn bộ nội dung phân tích, hướng, Entry/SL/TP/trigger/bằng chứng/rủi ro đã có trong plan gốc — "
        "chỉ sửa đúng phần bị lỗi định dạng nêu trên cho khớp đúng template OUTPUT PUBLIC. "
        "Không phân tích lại từ đầu, không đổi hướng, và TUYỆT ĐỐI không đổi bất kỳ con số Entry/SL/TP nào — "
        "đây chỉ là bước sửa lỗi trình bày, không phải cơ hội để phân tích lại giá. "
        "Chỉ được sửa phần trình bày: thêm nhãn Trạng thái còn thiếu, ghi lại đúng định dạng dòng Entry/SL/TP đã có, bổ sung mục còn thiếu của template. "
        "Nếu lỗi không thể sửa mà không đổi mức giá, hãy trả lại nguyên văn plan gốc.\n"
        "Trả lại toàn bộ output đầy đủ đúng template, không thêm giải thích ngoài template.\n\n"
        "=== PLAN GỐC ===\n"
        f"{planner_clean}"
    )
    try:
        repaired_raw = await asyncio.to_thread(request_claude_analysis, system_prompt, repair_prompt)
    except Exception as exc:
        print(f"[PLANNER_FORMAT_REPAIR_ERROR] {exc}", flush=True)
        return planner_clean, parse_prediction_from_output(planner_clean), guard_errors
    repaired_clean = (repaired_raw or "").strip()
    repaired_pred = parse_prediction_from_output(repaired_clean)
    remaining_errors = _validate_actionable_trade_plan(repaired_pred, timeframe_data, mode, current_price, repaired_clean)
    return repaired_clean, repaired_pred, remaining_errors


def _guarded_no_trade_output(
    symbol: str,
    mode: str,
    current_price: float | None,
    errors: list[str],
    pred: dict | None = None,
    timeframe_data: dict[str, pd.DataFrame | None] | None = None,
) -> str:
    """Render a NO TRADE caused by the Python guard, while still keeping the direction the model preferred.

    The DECISION is still NO TRADE because the trade failed the guard. The user should still see
    whether the original plan leaned LONG or SHORT — but that comes straight from the model's own
    rejected output, never from a Python-computed trend classification.
    """
    mode_label = "SCALP" if mode == "short" else "SWING"
    price_text = f" Giá hiện tại {fmt(current_price)} {BINANCE_QUOTE_ASSET}." if current_price is not None else ""
    reason = errors[0] if errors else "Kế hoạch LONG/SHORT bị bộ lọc rủi ro từ chối."
    pred_data = pred or {}

    rejected_direction = str(pred_data.get("direction") or "").upper()
    direction_line = ""
    if rejected_direction in ("LONG", "SHORT"):
        direction_emoji = "📈" if rejected_direction == "LONG" else "📉"
        direction_line = f"Hướng ưu tiên bị từ chối: {rejected_direction} {direction_emoji}\n"

    return sanitize_user_output(
        f"🎯 {symbol} — {mode_label}\n"
        f"🏆 QUYẾT ĐỊNH: NO TRADE\n"
        f"{direction_line}"
        f"Giá hiện tại: {fmt(current_price)} {BINANCE_QUOTE_ASSET}\n"
        f"⚠️ Rủi ro: {reason}{price_text} Bot không lưu tín hiệu này; nếu cố vào lệnh, nguy cơ bị nhiễu hoặc quét SL ngắn hạn còn cao."
    )


# ─── Hybrid AI validator ─────────────────────────────────────────────────────


def _remove_hidden_liquidity_sections(text: str) -> str:
    """Hide liquidity sections/zones from the user-facing output; this data is for internal use only."""
    if not text:
        return text

    # Remove the block starting with the liquidity emoji/heading, up to the next section.
    text = re.sub(
        r"\n?💧\s*(?:Thanh khoản|Vùng thanh khoản|Heatmap|Vùng thanh lý)[\s\S]*?(?=\n\s*(?:🏆|📈|📉|Entry:|Lý do:|📊|⚠️)|\Z)",
        "\n",
        text,
        flags=re.IGNORECASE,
    )

    # Remove liquidity/liquidation zone list lines in case the model still scattered them elsewhere.
    text = re.sub(
        r"^\s*(?:Vùng\s+)?(?:thanh khoản|thanh lý|heatmap|vùng quét)[^\n]*\n?",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    text = re.sub(
        r"^\s*Vùng\s+thanh\s+khoản\s+(?:dưới|trên)[^\n]*\n?",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    # Clean up extra whitespace after removing the block.
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

# Required technical label for the "Trang thai: ..." (Status) line. The underscore must be kept
# (NO_TRADE, not "NO TRADE") because _extract_setup_status() reads this line verbatim
# to determine the plan status. The wording sanitizer below must not touch this line.
_STATUS_LABEL_RE = re.compile(
    r"(Trạng\s*thái\s*:\s*)(READY_TO_ENTER|SETUP_WAITING_TRIGGER|NO_TRADE)",
    flags=re.IGNORECASE,
)


def sanitize_user_output(output: str) -> str:
    """Clean up confusing wording and internal technical labels before sending to the user / saving full_response."""
    replacements = {
        "swing gần": "đỉnh/đáy gần",
        "Swing gần": "Đỉnh/đáy gần",
        "swing lớn": "biên lớn",
        "Swing lớn": "Biên lớn",
        "MARKET_REGIME_DO_PYTHON_PHAN_LOAI": "phân loại thị trường do Python",
        "FEATURE_ENGINEERING_DO_PYTHON_TINH_SAN": "dữ liệu kỹ thuật do Python tính sẵn",
        "REGIME_CHINH": "xu hướng chính",
        "BULL_TREND": "xu hướng tăng",
        "BEAR_TREND": "xu hướng giảm",
        "RANGE_CHOPPY": "đi ngang/nhiễu",
        "MIXED_UNCLEAR": "chưa rõ xu hướng",
        "MIXED_TRANSITION": "trạng thái chuyển pha",
        "TRENDING_UP": "xu hướng tăng rõ",
        "TRENDING_DOWN": "xu hướng giảm rõ",
        "HIGH_VOLATILITY_RISK": "rủi ro biến động mạnh",
        "LOW_LIQUIDITY_RISK": "rủi ro thanh khoản thấp",
        "LOWER_TIMEFRAME_PULLBACK_AGAINST_STRUCTURE": "khung nhỏ đang hồi ngược cấu trúc lớn",
        "HIGH_VOLATILITY": "biến động mạnh",
        "LOW_VOLATILITY": "biến động thấp",
        "NORMAL_VOLATILITY": "biến động bình thường",
        "HIGH_VOLUME": "khối lượng cao",
        "LOW_VOLUME": "khối lượng thấp",
        "NORMAL_VOLUME": "khối lượng bình thường",
        "EMA_TANG": "EMA nghiêng tăng",
        "EMA_GIAM": "EMA nghiêng giảm",
        "EMA_DAN_XEN": "EMA đan xen",
        "modifier": "ghi chú",
    }
    text = output or ""

    # BUGFIX: protect the "Trang thai: NO_TRADE/READY_TO_ENTER/SETUP_WAITING_TRIGGER" (Status) line
    # before running the wording-cleanup regexes below. Otherwise the
    # "NO_TRADE -> NO TRADE" rule below would turn "Trang thai: NO_TRADE" (correct)
    # into "Trang thai: NO TRADE" (wrong format), which makes _extract_setup_status()
    # fail to match, and a perfectly valid NO_TRADE decision gets misrecorded
    # as STATUS_PARSE_ERROR in evaluation_cases.
    _status_placeholder = "\x00STATUS_LABEL_PLACEHOLDER\x00"
    _status_match = _STATUS_LABEL_RE.search(text)
    if _status_match:
        _status_label = _status_match.group(2).upper()
        text = _STATUS_LABEL_RE.sub(
            lambda m: f"{m.group(1)}{_status_placeholder}", text, count=1
        )

    # Clean up typos/English labels the model sometimes slips into the user-facing output.
    text = re.sub(r"\bNO[_\s-]?TRADE\b", "NO TRADE", text, flags=re.IGNORECASE)
    text = re.sub(r"\bREJECTED[_\s-]?PLAN\b", "kế hoạch bị từ chối", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsweep\b", "quét thanh khoản", text, flags=re.IGNORECASE)
    text = re.sub(r"\breclaim\b", "lấy lại vùng", text, flags=re.IGNORECASE)
    text = re.sub(r"\brisk\s*/\s*reward\b", "tỷ lệ lời/lỗ", text, flags=re.IGNORECASE)
    text = re.sub(r"\brisk\s*-\s*reward\b", "tỷ lệ lời/lỗ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<![a-zA-Z\-])\brisk\b(?![\-a-zA-Z])", "rủi ro", text, flags=re.IGNORECASE)
    text = re.sub(r"\breward\b", "lợi nhuận kỳ vọng", text, flags=re.IGNORECASE)
    text = re.sub(r"\bNếuu\b", "Nếu", text, flags=re.IGNORECASE)
    text = re.sub(r"\bNếuuu+\b", "Nếu", text, flags=re.IGNORECASE)
    # Replace longer internal labels first so overlapping terms do not leave fragments.
    for old in sorted(replacements, key=len, reverse=True):
        text = text.replace(old, replacements[old])

    # Clean MACD histogram labels with a dedicated regex so words like "history" aren't accidentally mangled.
    text = re.sub(r"\bMACD[_\s-]*hist(?:ogram)?\b", "động lượng MACD", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhist(?:ogram)?\b", "động lượng MACD", text, flags=re.IGNORECASE)

    # Keep the public output minimal: don't show extra metadata or legacy sections.
    text = re.sub(r"^\s*Xu hướng:[^\n]*\n?", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"^\s*Giá:[^\n]*\n?", "", text, flags=re.IGNORECASE | re.MULTILINE)
    # For LONG/SHORT, older explanation blocks may be hidden to keep the public output concise.
    # For NO TRADE, the explanation is the main content that helps the user understand why the planner stayed out;
    # never remove the Reason/Main Scenario block just because there's no Risk section after it.
    is_no_trade_output = bool(
        re.search(
            r"(?:QUYẾT\s+ĐỊNH|Trạng\s+thái)\s*:\s*NO\s+TRADE\b",
            text,
            flags=re.IGNORECASE,
        )
    )
    if not is_no_trade_output:
        text = re.sub(
            r"\n?\s*(?:📊\s*)?(?:Lý\s*do|Kịch\s*bản\s*chính)\s*:[\s\S]*?(?=\n\s*⚠️\s*Rủi\s*ro\s*:|\n\s*\[\[TEOPARD_|\Z)",
            "\n",
            text,
            flags=re.IGNORECASE,
        )
    text = _remove_hidden_liquidity_sections(text)

    # Restore the verbatim Status label that was protected above (keeping the underscore intact).
    if _status_match:
        text = text.replace(_status_placeholder, _status_label)

    return text


def ensure_current_price_line(output: str, current_price: float | None) -> str:
    """Insert the Current Price line below the DECISION line if the older model text doesn't already have it."""
    text = output or ""
    if re.search(r"^\s*Giá\s+hiện\s+tại\s*:", text, flags=re.IGNORECASE | re.MULTILINE):
        return text
    price_line = f"Giá hiện tại: {fmt(current_price)} {BINANCE_QUOTE_ASSET}" if current_price is not None else "Giá hiện tại: N/A"
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.search(r"QUYẾT\s+ĐỊNH\s*:", line, flags=re.IGNORECASE):
            insert_at = i + 1
            while insert_at < len(lines) and re.search(
                r"^\s*Độ\s+(?:mạnh\s+setup|chắc\s+chắn)\s*:",
                lines[insert_at],
                flags=re.IGNORECASE,
            ):
                insert_at += 1
            lines.insert(insert_at, price_line)
            return "\n".join(lines)
    return price_line + "\n" + text




def log_hidden_rejection(symbol: str, mode: str, pred: dict, validation_errors: list[str], output: str) -> None:
    """Log only technical parse/format rejection details; no Python market scoring."""
    try:
        print("[TEOPARD_TECHNICAL_REJECT]", flush=True)
        print(f"symbol={symbol} mode={mode} direction={pred.get('direction')}", flush=True)
        print("errors=" + " | ".join(str(e) for e in (validation_errors or [])), flush=True)
        print("output_preview=" + (output or "")[:1500].replace("\n", " "), flush=True)
    except Exception:
        pass


def _load_prompt_file(*filenames: str) -> str:
    """Load a prompt reliably from cwd or beside analyze.py."""
    bases = [Path.cwd(), Path(__file__).resolve().parent]
    checked = []
    for base in bases:
        for filename in filenames:
            path = base / filename
            checked.append(str(path))
            if path.exists():
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    return text
    raise FileNotFoundError(f"Không tìm thấy prompt: {', '.join(filenames)}; checked={checked}")


def load_system_prompt() -> str:
    return _load_prompt_file("analyze_system_prompt.txt", "analysis_system_prompt.txt")


def load_timeframe_data(binance_symbol: str, interval: str, limit: int) -> pd.DataFrame | None:
    """Sync helper: fetch Binance candles then calculate indicators."""
    return add_indicators(get_binance_klines(binance_symbol, interval, limit))


def request_claude_analysis(system_prompt: str, user_prompt: str) -> str:
    """Sync helper: calls the main model; output is short so there's no continuation."""
    max_tokens = max(800, min(PLANNER_MAX_OUTPUT_TOKENS, PLANNER_OUTPUT_TOKEN_CAP))
    return create_with_continuation(
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=max_tokens,
        timeout=PLANNER_TIMEOUT_SECONDS,
        allow_continuation=False,
        call_type="main",
    )


# ─── Objective market packet ──────────────────────────────────────────────

def _mode_frame_roles(mode: str) -> tuple[str, str, str, str]:
    """Return this mode's 4 timeframe labels, smallest to largest. Purely an iteration order —
    none of the 4 is treated as more important than another anywhere downstream."""
    if mode == "short":
        return "15M", "1H", "4H", "1D"
    return "4H", "1D", "1W", "1M"


def _missing_critical_timeframes(timeframe_data: dict, mode: str) -> list[str]:
    """No timeframe is presumed less important than another, so all four are required to have been
    fetched at all — if any is missing, the model must not be trusted to notice a "không có dữ liệu"
    text line and quietly work around it; Python forces NO_TRADE instead of letting a partial
    packet reach the planner.

    This is about a real fetch failure only (network/API error -> load_timeframe_data returns None).
    It is deliberately NOT triggered by an empty-but-not-None DataFrame: that shape means the fetch
    itself succeeded but this coin doesn't have enough closed history yet for any indicator to
    produce a real value at this interval (e.g. 1M for a coin listed 2 weeks ago, or 4H for a coin
    listed a few days ago) — add_indicators' own dropna() already produces that empty frame
    naturally. That case is handled separately, downstream, by omitting just that one timeframe's
    section from the packet instead of failing the whole analysis.
    """
    critical = list(_mode_frame_roles(mode))
    return [label for label in critical if timeframe_data.get(label) is None]

def _v50_timestamp_value(row) -> pd.Timestamp | None:
    """Get the UTC timestamp for the correct candle for internal use; not the display string, which shouldn't be used for calculations."""
    for key in ("open_time", "timestamp", "time", "datetime"):
        try:
            value = row.get(key)
        except Exception:
            value = None
        if value is not None and str(value) not in {"", "nan", "NaT"}:
            try:
                if isinstance(value, (int, float, np.integer, np.floating)):
                    unit = "ms" if float(value) > 10_000_000_000 else "s"
                    return pd.to_datetime(value, unit=unit, utc=True)
                return pd.to_datetime(value, utc=True)
            except Exception:
                pass
    try:
        return pd.to_datetime(row.name, utc=True)
    except Exception:
        return None


def _v50_time_value(row) -> str:
    """Display the full market-packet timestamp in Vietnam time (UTC+7)."""
    ts = _v50_timestamp_value(row)
    if ts is None or pd.isna(ts):
        return str(getattr(row, "name", "N/A"))
    return ts.tz_convert("Asia/Ho_Chi_Minh").strftime("%Y-%m-%d %H:%M VN")


def _v50_closed_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    # The last Binance row is usually the still-running candle. With only 1 row total, that row
    # IS the still-running candle (e.g. a coin too new to have even one closed candle yet at this
    # interval) — there are zero closed candles, not one, so this must not fall through to
    # returning that single unclosed row as if it were confirmed data.
    if len(df) < 2:
        return df.iloc[0:0].copy()
    return df.iloc[:-1].copy()


def _v50_raw_limit(mode: str, label: str) -> int:
    # This is the DISPLAY window only — how many closed candles get printed row-by-row. It is
    # deliberately much smaller than the FETCH window (see SHORT_TERM_TIMEFRAMES/LONG_TERM_TIMEFRAMES)
    # that add_indicators uses for EMA/RSI/MACD/ADX warm-up: showing hundreds of raw rows doesn't
    # help the model (long, repetitive numeric tables are unreliable to read in full — the earlier
    # single-snapshot + trailing indicator series already carry the "how has this been trending"
    # signal), it just adds noise and cost. Fetching stays wide for indicator accuracy either way.
    # SCALP counts in days (1/2/3/4 days of 15M/1H/4H/1D). SWING: 1 week of 4H, 2 weeks of 1D,
    # 6 weeks of 1W, 6 months of 1M. If a coin doesn't have this many closed candles yet, the
    # caller (_v50_raw_candles' .tail()) just returns however many actually exist — this is an
    # upper bound, never a forced/padded count.
    limits = {
        "short": {"15M": 96, "1H": 48, "4H": 18, "1D": 4},
        "long": {"4H": 42, "1D": 14, "1W": 6, "1M": 6},
    }
    return limits.get(mode, {}).get(label, 16)


def _v50_raw_candles(label: str, df: pd.DataFrame | None, mode: str) -> str:
    closed = _v50_closed_df(df)
    if closed is None or closed.empty:
        return ""
    rows = closed.tail(_v50_raw_limit(mode, label))
    # vol_ratio replaces raw volume (raw volume is meaningless without context).
    # takerBuy% shows buy-side pressure per candle, enabling accumulation/distribution reading.
    # CVD is cumulative delta starting from 0 at the first row shown here — only its shape/trend
    # across this window matters (compare against price shape), not the absolute number.
    out = [f"{label} — {len(rows)} nến đã đóng gần nhất (time,O,H,L,C,vol_ratio,takerBuy%,CVD):"]
    cvd = 0.0
    for _, row in rows.iterrows():
        taker = _taker_buy_ratio(row)
        cvd += _candle_delta(row)
        out.append(
            f"{_v50_time_value(row)} | "
            f"{fmt(_safe_float(row.get('open')))} | {fmt(_safe_float(row.get('high')))} | "
            f"{fmt(_safe_float(row.get('low')))} | {fmt(_safe_float(row.get('close')))} | "
            f"{fmt(_safe_float(row.get('vol_ratio')), 2)}x | "
            f"{fmt(taker, 1) if taker is not None else 'N/A'}% | "
            f"{fmt(cvd, 0)}"
        )
    return "\n".join(out)




def _v50_live_line(label: str, df: pd.DataFrame | None) -> str:
    # Requires at least one closed candle to exist too (len >= 2), not just the live one — a coin
    # too new to have any closed candle at this interval gets this whole frame omitted, same as
    # everywhere else, instead of showing a live candle with no closed history behind it.
    if df is None or df.empty or len(df) < 2:
        return ""
    row = df.iloc[-1]
    raw_progress = _live_candle_progress(row)
    progress = raw_progress * 100.0 if raw_progress is not None else None
    return (
        f"{label} live ({fmt(progress, 1) if progress is not None else 'N/A'}%): "
        f"time={_v50_time_value(row)}, O={fmt(_safe_float(row.get('open')))}, "
        f"H={fmt(_safe_float(row.get('high')))}, L={fmt(_safe_float(row.get('low')))}, "
        f"C={fmt(_safe_float(row.get('close')))}, V={fmt(_safe_float(row.get('volume')))}. "
        "Đây là nến đang chạy, không phải xác nhận đóng nến."
    )


def _v50_indicator_series(label: str, df: pd.DataFrame | None, count: int = 8) -> str:
    """Trailing EMA7/EMA25/EMA50, RSI12, and MACD-histogram values, most recent last — pure numbers,
    no trend label attached (same "give the series, not a conclusion" treatment CVD already gets in
    _v50_raw_candles).

    The single-candle indicator line elsewhere in the packet only ever shows the latest value, so
    there is no data at all backing a judgment like "EMA7 vừa cắt lên EMA25" or "momentum is fading"
    — and unlike a plain close price, these are recursive smoothed values (same ewm mechanism as
    RSI), not something a model can reconstruct by eyeballing raw OHLC. This does not decide whether
    a cross happened or momentum is diverging; it only gives the numbers a model would need to make
    that call itself. vol_ratio/CVD are skipped here — those already appear as a per-row series in
    the raw OHLCV block, so repeating them would be redundant.
    """
    closed = _v50_closed_df(df)
    if closed is None or len(closed) < count:
        return ""
    tail = closed.tail(count)
    fields = [
        ("EMA7", "ema_7", 4), ("EMA25", "ema_25", 4), ("EMA50", "ema_50", 4),
        ("RSI12", "rsi_12", 1), ("MACD histogram", "macd_hist", 4),
    ]
    out = []
    for display_name, col, decimals in fields:
        vals = [_safe_float(row.get(col)) for _, row in tail.iterrows()]
        if any(v is None for v in vals):
            return ""
        out.append(f"{label} {display_name} {count} nến đã đóng gần nhất: " + ", ".join(fmt(v, decimals) for v in vals))
    return "\n".join(out)


def _v50_ichimoku_block(label: str, df: pd.DataFrame | None, chikou_lag: int = 26) -> str:
    """Tenkan-sen, Kijun-sen, and the Senkou Span A/B values overlapping the latest closed candle
    (i.e. already shifted forward, same as what a real chart displays there) — plus the close price
    from `chikou_lag` candles ago for a Chikou-style comparison against the current close (already
    given elsewhere in the packet). Pure numbers; Python does not state where price sits relative
    to the cloud or any other conclusion — that comparison is left entirely to the model.
    """
    closed = _v50_closed_df(df)
    if closed is None or len(closed) < 2:
        return ""
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(closed)
    row_idx = len(closed) - 1
    t, k, sa, sb = tenkan.iloc[row_idx], kijun.iloc[row_idx], senkou_a.iloc[row_idx], senkou_b.iloc[row_idx]
    if any(pd.isna(v) for v in (t, k, sa, sb)):
        return ""
    lines = [
        f"{label} Ichimoku (Tenkan9/Kijun26/SenkouB52, đã dịch tới {chikou_lag} nến): "
        f"Tenkan-sen={fmt(_safe_float(t))}, Kijun-sen={fmt(_safe_float(k))}, "
        f"Senkou Span A={fmt(_safe_float(sa))}, Senkou Span B={fmt(_safe_float(sb))}"
    ]
    if row_idx - chikou_lag >= 0:
        chikou_close = _safe_float(closed.iloc[row_idx - chikou_lag].get("close"))
        if chikou_close is not None:
            lines.append(f"{label} giá đóng cửa {chikou_lag} nến trước (đối chiếu Chikou): {fmt(chikou_close)}")
    return "\n".join(lines)


def build_feature_engineering_block(
    timeframe_data: dict[str, pd.DataFrame | None],
    mode: str,
    current_price: float | None,
) -> str:
    """Build the objective packet used by both Manual and Auto Scan planner.

    All four timeframes receive identical treatment — same indicator set, same series depth.
    Python doesn't pre-assign which frame matters more for direction vs. entry vs. timing; that
    judgment is left entirely to the model. Only standard, formula-defined indicators (EMA/RSI/
    MACD/ADX/ATR/Ichimoku) appear here — no Python-invented pattern detector (swing/pivot finder,
    "noise profile", range-position stat) with its own tunable sensitivity parameter.
    """
    trigger, setup, trend, big = _mode_frame_roles(mode)
    labels = [trigger, setup, trend, big]
    lines = [
        "OBJECTIVE_MARKET_PACKET",
        "Múi giờ của mọi timestamp trong packet: giờ Việt Nam (UTC+7), hậu tố VN.",
        f"Giá hiện tại: {fmt(current_price)}",
        "Python chỉ chuẩn bị dữ kiện khách quan; không kết luận hướng và không dựng Entry/SL/TP.",
        "Packet có atr_pct (ngữ cảnh biên độ khách quan); không có Fibonacci, market-regime label hay trend label.",
    ]
    for label in labels:
        df = timeframe_data.get(label)
        row = _analysis_row(df)
        if row is None:
            # No closed candle at all for this timeframe (e.g. a coin too new to have one yet at
            # this interval) — omit the whole section instead of a placeholder line. Nothing told
            # the model up front how many timeframes to expect, so a frame simply not appearing
            # here needs no explanation.
            continue
        lines.append(
            f"{label} chỉ báo nến đóng gần nhất: close={fmt(_safe_float(row.get('close')))}, "
            f"EMA7={fmt(_safe_float(row.get('ema_7')))}, EMA25={fmt(_safe_float(row.get('ema_25')))}, "
            f"EMA50={fmt(_safe_float(row.get('ema_50')))}, "
            f"RSI6={fmt(_safe_float(row.get('rsi_6')),1)},RSI12={fmt(_safe_float(row.get('rsi_12')),1)},RSI24={fmt(_safe_float(row.get('rsi_24')),1)}, "
            f"MACD line={fmt(_safe_float(row.get('macd_line')))}, signal={fmt(_safe_float(row.get('macd_signal')))}, "
            f"histogram={fmt(_safe_float(row.get('macd_hist')))}, vol_ratio={fmt(_safe_float(row.get('vol_ratio')),2)}x, "
            f"atr_pct={fmt(_safe_float(row.get('atr_pct')),3)}%, adx14={fmt(_safe_float(row.get('adx_14')),1)}, "
            f"takerBuy={fmt(_taker_buy_ratio(row),1)}%."
        )
        indicator_series = _v50_indicator_series(label, df)
        if indicator_series:
            lines.append(indicator_series)
        # SCALP 1D and SWING 1M are excluded from Ichimoku by design (not a data check) — 1D/1M
        # here is a large-scale background frame with only a handful of raw candles shown, and
        # SWING 1M additionally can never satisfy Ichimoku's own data requirement anyway (see
        # LONG_TERM_TIMEFRAMES' 1M comment).
        skip_ichimoku = (mode == "short" and label == "1D") or (mode == "long" and label == "1M")
        if not skip_ichimoku:
            ichimoku_block = _v50_ichimoku_block(label, df)
            if ichimoku_block:
                lines.append(ichimoku_block)
    return "\n".join(lines)


def build_feature_snapshot(
    timeframe_data: dict[str, pd.DataFrame | None],
    mode: str,
    current_price: float | None,
) -> str:
    """Compact packet stored to `predictions.feature_snapshot` for later inspection — never sent to
    the model. Left over from the removed Prefilter stage this used to feed; kept only as a DB record
    of standard-indicator values + recent closed candles at analysis time, same data shape as the
    Planner packet but smaller.
    """
    trigger, setup, trend, big = _mode_frame_roles(mode)
    lines = [
        f"Mode={'SCALP' if mode == 'short' else 'SWING'}; price={fmt(current_price)}",
    ]
    # SCALP: timing 12, setup 24, trend 16, macro 6. SWING uses the same allocation, mapped to
    # the corresponding roles.
    recent_counts = {trigger: 12, setup: 24, trend: 16, big: 6}
    for label in (trigger, setup, trend, big):
        df = timeframe_data.get(label)
        closed = _v50_closed_df(df)
        row = _analysis_row(df) if df is not None and not df.empty else None
        if row is None:
            lines.append(f"{label}: N/A")
            continue
        lines.append(
            f"{label} latest: O={fmt(_safe_float(row.get('open')))},H={fmt(_safe_float(row.get('high')))},"
            f"L={fmt(_safe_float(row.get('low')))},C={fmt(_safe_float(row.get('close')))},"
            f"EMA7/25/50={fmt(_safe_float(row.get('ema_7')))}/{fmt(_safe_float(row.get('ema_25')))}/{fmt(_safe_float(row.get('ema_50')))},"
            f"RSI6={fmt(_safe_float(row.get('rsi_6')),1)},RSI12={fmt(_safe_float(row.get('rsi_12')),1)},RSI24={fmt(_safe_float(row.get('rsi_24')),1)},"
            f"MACDline={fmt(_safe_float(row.get('macd_line')))},"
            f"signal={fmt(_safe_float(row.get('macd_signal')))},hist={fmt(_safe_float(row.get('macd_hist')))},"
            f"vol_ratio={fmt(_safe_float(row.get('vol_ratio')),2)}x,"
            f"atr_pct={fmt(_safe_float(row.get('atr_pct')),3)}%,adx14={fmt(_safe_float(row.get('adx_14')),1)},"
            f"takerBuy={fmt(_taker_buy_ratio(row),1)}%"
        )
        if closed is not None and not closed.empty:
            compact=[]
            cvd = 0.0
            for _, candle in closed.tail(recent_counts[label]).iterrows():
                taker = _taker_buy_ratio(candle)
                cvd += _candle_delta(candle)
                compact.append(
                    f"{_v50_time_value(candle)} O={fmt(_safe_float(candle.get('open')))} "
                    f"H={fmt(_safe_float(candle.get('high')))} L={fmt(_safe_float(candle.get('low')))} "
                    f"C={fmt(_safe_float(candle.get('close')))} "
                    f"vol={fmt(_safe_float(candle.get('vol_ratio')),2)}x "
                    f"macd_h={fmt(_safe_float(candle.get('macd_hist')),4)} "
                    f"tb={fmt(taker,1) if taker is not None else 'N/A'}% "
                    f"cvd={fmt(cvd,0)}"
                )
            lines.append(f"{label} recent closed ({len(compact)}): " + " || ".join(compact))
    return "\n".join(lines)


def build_synchronized_decision_snapshot(
    timeframe_data: dict[str, pd.DataFrame | None],
    mode: str,
    current_price: float | None,
) -> str:
    trigger, setup, trend, big = _mode_frame_roles(mode)
    lines = ["SYNCHRONIZED_DECISION_SNAPSHOT", "Mọi timestamp bên dưới dùng giờ Việt Nam (UTC+7), hậu tố VN."]
    lines += [
        line for label in (trigger, setup, trend, big)
        if (line := _v50_live_line(label, timeframe_data.get(label)))
    ]
    return "\n".join(lines)


def build_user_prompt(
    symbol: str,
    mode: str,
    timeframe_data: dict[str, pd.DataFrame | None],
    fear_greed_info: str,
    current_price_str: str,
    feature_block: str | None = None,
    open_signal_context: str | None = None,
    decision_snapshot: str | None = None,
    direction_scorecard: str | None = None,
    market_context_block: str | None = None,
) -> str:
    """Data-first planner prompt; analytical rules live only in system prompt."""
    mode_label = "SCALP" if mode == "short" else "SWING"
    trigger, setup, trend, big = _mode_frame_roles(mode)
    raw_sections = [
        section for label in (trigger, setup, trend, big)
        if (section := _v50_raw_candles(label, timeframe_data.get(label), mode))
    ]
    return "\n".join([
        f"PHÂN TÍCH {symbol} — {mode_label}",
        f"Thời điểm tạo packet: {utc_now().astimezone(VN_TZ).strftime('%Y-%m-%d %H:%M:%S VN')}",
        current_price_str,
        "Không có kế hoạch đang mở, Fear & Greed, Fibonacci hoặc hướng ưu tiên. RAW OHLCV bên dưới dùng vol_ratio và takerBuy% thay volume thô.",
        "",
        "RAW OHLCV:",
        "\n\n".join(raw_sections),
        "",
        feature_block or "OBJECTIVE_MARKET_PACKET: N/A",
        "",
        market_context_block or "",
        "",
        decision_snapshot or "LIVE SNAPSHOT: N/A",
        "",
        "Tuân thủ toàn bộ quy trình phân tích và tự phản biện trong system prompt, nhưng phần bạn xuất ra chỉ được bắt đầu thẳng từ dòng 🎯 bên dưới — không in bất kỳ nhãn hay khối nào kiểu 'DECISION ENGINE', ghi chú xác nhận, hay bước suy luận trung gian nào trước dòng đó.",
        "",
        "OUTPUT PUBLIC:",
        f"🎯 {symbol} — {mode_label}",
        "🏆 QUYẾT ĐỊNH: [CHỌN MỘT: LONG / SHORT / NO TRADE]",
        "Trạng thái: READY_TO_ENTER | SETUP_WAITING_TRIGGER | NO_TRADE",
        f"Giá hiện tại: ... {BINANCE_QUOTE_ASSET}",
        "Nếu NO TRADE:",
        "Lý do: (1–2 câu ngắn gọn nêu đúng lý do bạn không vào lệnh)",
        "Nếu LONG/SHORT:",
        "Entry: low–high",
        "SL: ...",
        "TP1: ...",
        "TP2: ... hoặc N/A",
        "Kích hoạt: ...",
        "Bằng chứng Entry: ...",
        "Bằng chứng SL: ...",
        "Bằng chứng TP1: ...",
        "Bằng chứng TP2: ... hoặc N/A",
        "⚠️ Rủi ro:",
        "- ...",
    ])


def _extract_setup_status(output: str | None) -> str:
    """Read an explicit planner status; never infer READY_TO_ENTER from prose."""
    text = output or ""
    # Same markdown-emphasis tolerance as the direction parser (see parse_prediction_from_output) —
    # confirmed live: a real response bolding "Trạng thái: **SETUP_WAITING_TRIGGER**" hit this exact
    # gap and got wrongly discarded as STATUS_PARSE_ERROR even though the label was there and correct.
    m = re.search(r"Trạng\s*thái\s*:\s*[*_]*(READY_TO_ENTER|SETUP_WAITING_TRIGGER|NO_TRADE)", text, flags=re.I)
    if m:
        return m.group(1).upper()
    return "STATUS_PARSE_ERROR"


def _ensure_v50_tables() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                user_id INTEGER,
                chat_id INTEGER,
                symbol TEXT NOT NULL,
                mode TEXT NOT NULL,
                source TEXT NOT NULL,
                model TEXT,
                data_variant TEXT,
                planner_input TEXT,
                planner_output TEXT,
                setup_status TEXT,
                current_price REAL,
                outcome TEXT DEFAULT 'SETUP_CREATED',
                mae REAL,
                mfe REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS auto_scan_trend_state (
                user_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                mode TEXT NOT NULL,
                last_direction TEXT,
                skip_remaining INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(user_id, symbol, mode)
            )
        """)
        for table in ("predictions",):
            for col, definition in [
                ("setup_status", "TEXT"),
                ("mae", "REAL"),
                ("mfe", "REAL"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {definition}")
                except sqlite3.OperationalError:
                    pass


def _auto_scan_consume_trend_skip(user_id: int, symbol: str, mode: str) -> int | None:
    """If this symbol/mode is in a trend-confirmed skip window, consume one skip and return the
    remaining count. Returns None when there's nothing to skip (normal scan should proceed)."""
    _ensure_v50_tables()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT skip_remaining FROM auto_scan_trend_state WHERE user_id=? AND symbol=? AND mode=?",
            (user_id, symbol, mode),
        ).fetchone()
        remaining = int(row[0]) if row and row[0] else 0
        if remaining <= 0:
            return None
        remaining -= 1
        conn.execute(
            "UPDATE auto_scan_trend_state SET skip_remaining=?, updated_at=? WHERE user_id=? AND symbol=? AND mode=?",
            (remaining, iso(utc_now()), user_id, symbol, mode),
        )
        conn.commit()
    return remaining


def _auto_scan_update_trend_state(user_id: int, symbol: str, mode: str, direction: str) -> int:
    """Compare this scan's direction to the previous one. Two consecutive LONG/LONG or SHORT/SHORT
    scans mean the trend is already confirmed, so the next 2 scan cycles are skipped to save cost
    (e.g. 13h VN LONG, 14h LONG -> skip 15h/16h, resume 17h). A trigger resets the memory so the
    scan right after resuming needs a fresh pair before it can trigger again, instead of
    immediately re-triggering off the stale pre-skip direction. Returns the number of scans just
    scheduled to be skipped (0 if this scan didn't trigger one)."""
    _ensure_v50_tables()
    direction = str(direction or "").upper()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT last_direction FROM auto_scan_trend_state WHERE user_id=? AND symbol=? AND mode=?",
            (user_id, symbol, mode),
        ).fetchone()
        last_direction = row[0] if row else None
        triggered = bool(last_direction and direction in {"LONG", "SHORT"} and direction == last_direction)
        new_last_direction = None if triggered else direction
        new_skip_remaining = 2 if triggered else 0
        conn.execute(
            """INSERT INTO auto_scan_trend_state(user_id, symbol, mode, last_direction, skip_remaining, updated_at)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(user_id, symbol, mode) DO UPDATE SET
               last_direction=excluded.last_direction, skip_remaining=excluded.skip_remaining, updated_at=excluded.updated_at""",
            (user_id, symbol, mode, new_last_direction, new_skip_remaining, iso(utc_now())),
        )
        conn.commit()
    return new_skip_remaining


def _save_analysis_snapshot(**kwargs) -> None:
    """Save the full case whenever Planner is called; the older history and Auto log still serve their own separate UI."""
    try:
        _ensure_v50_tables()
        planner_output = kwargs.get("planner_output") or ""
        public_output = kwargs.get("public_output") or planner_output
        parsed = parse_prediction_from_output(public_output)
        direction = (parsed.get("direction") or "NO_TRADE").upper()
        status = kwargs.get("setup_status") or _extract_setup_status(public_output)
        if direction == "NO_TRADE":
            phase, final_result = "PLANNER_NO_TRADE", "NO_TRADE"
        elif status in ("SETUP_WAITING_TRIGGER", "READY_TO_ENTER"):
            phase, final_result = "PLANNER_APPROVED", direction
        else:
            phase, final_result = "PLANNER_PARSE_ERROR", "PARSE_ERROR"

        funding_ctx = kwargs.get("funding_context") or {}
        save_evaluation_case(
            user_id=kwargs.get("user_id"), chat_id=kwargs.get("chat_id"), source=kwargs.get("source") or "unknown",
            symbol=kwargs.get("symbol"), mode=kwargs.get("mode"), pipeline_phase=phase, final_result=final_result,
            current_price=kwargs.get("current_price"),
            planner_direction=direction, planner_status=status,
            entry_low=parsed.get("entry_low"),
            entry_high=parsed.get("entry_high"), sl=parsed.get("sl"), tp1=parsed.get("tp1"), tp2=parsed.get("tp2"),
            market_packet=kwargs.get("planner_input"), planner_output=planner_output,
            public_output=public_output, planner_prompt_hash=prompt_hash(load_system_prompt()),
            funding_rate_pct=funding_ctx.get("latest_pct"),
            btc_context_text=kwargs.get("btc_context_text"),
        )
        cleanup_evaluation_data()
    except Exception as exc:
        print(f"[SNAPSHOT_SAVE_ERROR] {exc}", flush=True)


async def collect_timeframe_data(binance_symbol: str, mode: str) -> dict[str, pd.DataFrame | None]:
    """
    Fetch multiple timeframes in parallel worker threads.

    Goal: keep requests.get() from blocking the Telegram bot's event loop, and also
    reduce wait time since 15M/1H/4H or 4H/1D/1W load in parallel.
    """
    configs = SHORT_TERM_TIMEFRAMES if mode == "short" else LONG_TERM_TIMEFRAMES
    tasks = {
        label: asyncio.to_thread(load_timeframe_data, binance_symbol, interval, limit)
        for label, (interval, limit) in configs.items()
    }
    results = await asyncio.gather(*tasks.values())
    return dict(zip(tasks.keys(), results))


async def prepare_analysis_context(
    binance_symbol: str,
    mode: str,
    user_id: int | None = None,
    timeframe_data: dict[str, pd.DataFrame | None] | None = None,
) -> dict:
    """Build the same GLM context for both manual analysis and Auto Scan."""
    if timeframe_data is None:
        timeframe_data = await collect_timeframe_data(binance_symbol, mode)

    if not any(df is not None and not df.empty for df in timeframe_data.values()):
        raise RuntimeError(f"Could not fetch Binance data for {binance_symbol}.")

    missing_critical = _missing_critical_timeframes(timeframe_data, mode)
    if missing_critical:
        raise RuntimeError(
            f"Thiếu dữ liệu Binance cho khung quan trọng ({', '.join(missing_critical)}) của {binance_symbol}."
        )

    is_btc = binance_symbol.upper() == f"BTC{BINANCE_QUOTE_ASSET}"
    system_prompt, fear_greed_info, price_tuple, funding_ctx, oi_ctx, long_short_ctx, btc_ctx = await asyncio.gather(
        asyncio.to_thread(load_system_prompt),
        asyncio.to_thread(lambda: "Không sử dụng Fear & Greed trong phân tích."),
        asyncio.to_thread(get_current_price_str, binance_symbol),
        asyncio.to_thread(get_funding_rate_context, binance_symbol),
        asyncio.to_thread(get_open_interest_context, binance_symbol),
        asyncio.to_thread(get_long_short_ratio_context, binance_symbol),
        asyncio.to_thread(get_btc_correlation_snapshot) if not is_btc else asyncio.sleep(0, result=None),
    )
    current_price_str, current_price = price_tuple
    if current_price is None:
        # The dedicated ticker call failed transiently even though the klines fetches above
        # succeeded — fall back to the freshest closed candle's close instead of building the
        # whole packet around "Giá hiện tại: không có dữ liệu" when a usable price is one
        # indicator-tick away.
        fallback_price = _last_close_from_data(timeframe_data)
        if fallback_price is not None:
            current_price = fallback_price
            current_price_str = f"Giá hiện tại: {fmt(fallback_price)} {BINANCE_QUOTE_ASSET} (giá ticker lỗi tạm thời, dùng giá đóng nến gần nhất)"
    feature_block = build_feature_engineering_block(timeframe_data, mode, current_price)
    feature_snapshot = build_feature_snapshot(timeframe_data, mode, current_price)
    decision_snapshot = build_synchronized_decision_snapshot(timeframe_data, mode, current_price)
    # do not send LONG/SHORT support scorecard into model prompts.
    # This prevents Python from anchoring the model direction.
    direction_scorecard_payload = None
    direction_scorecard = None
    market_snapshot = build_market_snapshot(timeframe_data, fear_greed_info, current_price_str)
    open_signal_context = None
    futures_block = build_futures_context_block(binance_symbol, funding_ctx, oi_ctx, long_short_ctx)
    btc_block = build_btc_correlation_block(btc_ctx) if not is_btc else None
    market_context_block = "\n\n".join(b for b in (futures_block, btc_block) if b) or None
    user_prompt = build_user_prompt(
        symbol=binance_symbol,
        mode=mode,
        timeframe_data=timeframe_data,
        fear_greed_info=fear_greed_info,
        current_price_str=current_price_str,
        feature_block=feature_block,
        open_signal_context=open_signal_context,
        decision_snapshot=decision_snapshot,
        direction_scorecard=direction_scorecard,
        market_context_block=market_context_block,
    )
    return {
        "timeframe_data": timeframe_data,
        "system_prompt": system_prompt,
        "fear_greed_info": fear_greed_info,
        "current_price_str": current_price_str,
        "current_price": current_price,
        "open_signals": [],
        "open_signal_context": open_signal_context,
        "feature_block": feature_block,
        "feature_snapshot": feature_snapshot,
        "decision_snapshot": decision_snapshot,
        "direction_scorecard": direction_scorecard,
        "direction_scorecard_payload": direction_scorecard_payload,
        "market_snapshot": market_snapshot,
        "user_prompt": user_prompt,
        "funding_context": funding_ctx,
        "open_interest_context": oi_ctx,
        "long_short_context": long_short_ctx,
        "btc_context": btc_ctx,
        "market_context_block": market_context_block,
    }


async def analyze_symbol(symbol: str, mode: str, user_id: int | None = None, chat_id: int | None = None) -> dict:
    """
    Async entry point used by Telegram handlers.

    Never call requests.get(), a synchronous AI API, or SQLite directly on the event loop.
    Blocking I/O is offloaded to a worker thread via asyncio.to_thread().
    """
    ensure_ai_config()

    await asyncio.to_thread(init_prediction_db)

    binance_symbol = resolve_binance_symbol(symbol)
    loop = asyncio.get_running_loop()
    manual_started = loop.time()
    print(f"[MANUAL_START] symbol={binance_symbol} mode={mode} user_id={user_id}", flush=True)

    # Manual GLM shares the same context builder as GLM Auto Scan.
    ctx = await prepare_analysis_context(binance_symbol, mode, user_id=user_id)
    print(
        f"[MANUAL_CONTEXT_READY] symbol={binance_symbol} mode={mode} elapsed={loop.time() - manual_started:.1f}s",
        flush=True,
    )
    timeframe_data = ctx["timeframe_data"]
    system_prompt = ctx["system_prompt"]
    current_price = ctx["current_price"]
    feature_snapshot = ctx["feature_snapshot"]
    market_snapshot = ctx["market_snapshot"]
    user_prompt = ctx["user_prompt"]

    # The AI API call is synchronous, so it runs in a worker thread to avoid blocking the bot.
    print(f"[MANUAL_LLM_START] symbol={binance_symbol} mode={mode}", flush=True)
    raw_output = await asyncio.to_thread(request_claude_analysis, system_prompt, user_prompt)
    print(
        f"[MANUAL_LLM_DONE] symbol={binance_symbol} mode={mode} elapsed={loop.time() - manual_started:.1f}s",
        flush=True,
    )
    planner_clean = (raw_output or "").strip()
    output = ensure_current_price_line(sanitize_user_output(planner_clean), current_price)
    pred = parse_prediction_from_output(output)
    await asyncio.to_thread(
        _save_analysis_snapshot,
        user_id=user_id, chat_id=chat_id, symbol=binance_symbol, mode=mode, source="manual",
        model=get_ai_model_name(), planner_input=user_prompt, planner_output=planner_clean,
        setup_status=_extract_setup_status(output),
        current_price=current_price, public_output=output,
        funding_context=ctx.get("funding_context"), open_interest_context=ctx.get("open_interest_context"),
        long_short_context=ctx.get("long_short_context"),
        btc_context_text=build_btc_correlation_block(ctx.get("btc_context")),
    )

    # Model-authoritative flow:
    # - The model alone chooses and is responsible for all of Entry/SL/TP.
    # - Python keeps the model's numbers exactly as returned.
    # - The only gate is the NO_TRADE label itself; Python does not reject based on RR/ATR/structure/geometry,
    #   and there is no separate scoring/review stage anymore.
    direction = (pred.get("direction") or "").upper()

    usage_note = "\n\nLượt phân tích hôm nay vẫn bị tính (đã gọi AI xong)."

    if direction == "NO_TRADE":
        # NO TRADE is not saved into predictions/history; only trades the user confirms are tracked.
        return {"text": _strip_public_evidence_for_user(output) + usage_note, "candidate_id": None}

    guard_errors = _validate_actionable_trade_plan(pred, timeframe_data, mode, current_price, output)
    if guard_errors:
        # Guard failures are pure output-format slips (a level Python could not read, a missing
        # status label) — try one cheap repair reusing the existing analysis before discarding it.
        # The repair may not touch any price.
        repaired_clean, repaired_pred, remaining_errors = await _repair_planner_format(
            system_prompt, planner_clean, guard_errors, timeframe_data, mode, current_price
        )
        if not remaining_errors:
            planner_clean = repaired_clean
            output = ensure_current_price_line(sanitize_user_output(planner_clean), current_price)
            pred = repaired_pred
            # The repair prompt is told not to change direction, but nothing enforces that —
            # re-derive from the repaired plan so a stale pre-repair value can't get persisted
            # against post-repair Entry/SL/TP and corrupt win/loss tracking downstream.
            direction = (pred.get("direction") or "").upper()
            guard_errors = []
        else:
            guard_errors = remaining_errors

    if guard_errors:
        guarded_output = _guarded_no_trade_output(binance_symbol, mode, current_price, guard_errors, pred, timeframe_data)
        log_hidden_rejection(binance_symbol, mode, pred, guard_errors, output)
        # rejected plans are no longer saved into predictions/history.
        return {"text": guarded_output + usage_note, "candidate_id": None}

    can_track = (
        direction in ("LONG", "SHORT")
        and pred.get("entry_low") is not None
        and pred.get("entry_high") is not None
        and pred.get("sl") is not None
        and pred.get("tp1") is not None
    )

    tracking_note = ""
    if can_track:
        reasoning_summary = build_local_reasoning_summary(output)
        await asyncio.to_thread(
            save_prediction,
            symbol=binance_symbol,
            mode=mode,
            direction=direction,
            entry_low=pred.get("entry_low"),
            entry_high=pred.get("entry_high"),
            sl=pred.get("sl"),
            tp1=pred.get("tp1"),
            tp2=pred.get("tp2"),
            market_snapshot=market_snapshot,
            feature_snapshot=feature_snapshot,
            reasoning_summary=reasoning_summary,
            full_response=output,
            user_id=user_id,
            chat_id=chat_id,
            setup_status=_extract_setup_status(output),
        )
        tracking_note = "\n\nBot đã tự lưu phân tích này để theo dõi kết quả."
    else:
        missing = []
        if direction not in ("LONG", "SHORT"):
            missing.append("Không parse được QUYẾT ĐỊNH LONG/SHORT/NO TRADE.")
        for field in ("entry_low", "entry_high", "sl", "tp1"):
            if pred.get(field) is None:
                missing.append(f"Không parse được {field}.")
        log_hidden_rejection(binance_symbol, mode, pred, missing, output)

    strength_index = await asyncio.to_thread(_btc_eth_strength_index)
    output = _insert_btc_strength_line(output, strength_index)

    print(
        f"[MANUAL_DONE] symbol={binance_symbol} mode={mode} elapsed={loop.time() - manual_started:.1f}s",
        flush=True,
    )
    return {"text": _strip_public_evidence_for_user(output) + tracking_note, "candidate_id": None}


# ─── Auto Scan Mode: hourly Planner call, gated only on NO_TRADE ─────────────

_auto_scan_db_initialized = False


def init_auto_scan_db() -> None:
    """Separate DB for auto scan, kept apart from manual mode/drafts."""
    global _auto_scan_db_initialized
    if _auto_scan_db_initialized:
        return
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS auto_scan_settings (
                user_id     INTEGER PRIMARY KEY,
                chat_id     INTEGER,
                enabled     INTEGER NOT NULL DEFAULT 0,
                symbols     TEXT NOT NULL DEFAULT '',
                night_resume INTEGER NOT NULL DEFAULT 0,
                quota_resume INTEGER NOT NULL DEFAULT 0,
                glm_calls_today INTEGER NOT NULL DEFAULT 0,
                glm_calls_day TEXT NOT NULL DEFAULT '',
                updated_at  TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS auto_scan_signals (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER,
                chat_id       INTEGER,
                symbol        TEXT NOT NULL,
                mode          TEXT NOT NULL,
                direction     TEXT NOT NULL,
                confidence    INTEGER,
                sent_at       TEXT NOT NULL,
                prediction_id INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS auto_scan_logs (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id           INTEGER,
                chat_id           INTEGER,
                symbol            TEXT NOT NULL,
                mode              TEXT NOT NULL,
                scan_slot         TEXT,
                scanned_at        TEXT NOT NULL,
                stage             TEXT NOT NULL,
                status            TEXT NOT NULL,
                final_direction   TEXT,
                final_confidence  INTEGER,
                reason            TEXT,
                prediction_id     INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS auto_scan_state (
                key        TEXT PRIMARY KEY,
                value      TEXT,
                updated_at TEXT NOT NULL
            )
        """)
        for col, definition in [
            ("symbols", "TEXT NOT NULL DEFAULT ''"),
            ("night_resume", "INTEGER NOT NULL DEFAULT 0"),
            ("quota_resume", "INTEGER NOT NULL DEFAULT 0"),
            ("glm_calls_today", "INTEGER NOT NULL DEFAULT 0"),
            ("glm_calls_day", "TEXT NOT NULL DEFAULT ''"),
        ]:
            try:
                conn.execute(f"ALTER TABLE auto_scan_settings ADD COLUMN {col} {definition}")
            except sqlite3.OperationalError:
                pass
        conn.execute("CREATE INDEX IF NOT EXISTS idx_auto_scan_settings_enabled ON auto_scan_settings(enabled)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_auto_scan_signals_user_symbol_mode ON auto_scan_signals(user_id, symbol, mode, sent_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_auto_scan_logs_user_id ON auto_scan_logs(user_id, id DESC)")

        # Keep a lightweight log over time; the UI still only shows the 5 most recent rows.
        log_cutoff = iso(utc_now() - timedelta(days=AUTOSCAN_LOG_RETENTION_DAYS))
        conn.execute("DELETE FROM auto_scan_logs WHERE scanned_at < ?", (log_cutoff,))
        conn.commit()
    _auto_scan_db_initialized = True


def _auto_scan_quota_day_key(now: datetime | None = None) -> str:
    """The Auto Scan quota day runs from 07:00 VN to 06:59 VN the next day."""
    local_now = (now or utc_now()).astimezone(VN_TZ)
    wake_hour = max(0, min(23, int(AUTOSCAN_WAKE_HOUR_VN)))
    quota_date = local_now.date() if local_now.hour >= wake_hour else (local_now - timedelta(days=1)).date()
    return quota_date.isoformat()


def set_auto_scan_enabled(user_id: int, chat_id: int, enabled: bool, symbols: list[str] | None = None) -> dict:
    init_auto_scan_db()
    normalized_symbols = []
    if symbols is not None:
        seen = set()
        for raw in symbols:
            sym = normalize_auto_scan_symbol(raw)
            if sym and sym not in seen:
                normalized_symbols.append(sym)
                seen.add(sym)
    symbols_text = ",".join(normalized_symbols) if symbols is not None else None
    day_key = _auto_scan_quota_day_key()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT glm_calls_today, glm_calls_day FROM auto_scan_settings WHERE user_id=?",
            (user_id,),
        ).fetchone()
        calls = int(row[0] or 0) if row else 0
        stored_day = str(row[1] or "") if row else ""
        if stored_day != day_key:
            calls = 0
        quota_blocked = bool(enabled and calls >= AUTOSCAN_MAX_PLANNER_CALLS_PER_DAY)
        effective_enabled = bool(enabled and not quota_blocked)
        quota_resume = 1 if quota_blocked else 0
        if symbols_text is None:
            conn.execute(
                """
                INSERT INTO auto_scan_settings
                    (user_id, chat_id, enabled, night_resume, quota_resume, glm_calls_today, glm_calls_day, updated_at)
                VALUES (?, ?, ?, 0, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    chat_id=excluded.chat_id,
                    enabled=excluded.enabled,
                    night_resume=0,
                    quota_resume=excluded.quota_resume,
                    glm_calls_today=excluded.glm_calls_today,
                    glm_calls_day=excluded.glm_calls_day,
                    updated_at=excluded.updated_at
                """,
                (user_id, chat_id, 1 if effective_enabled else 0, quota_resume, calls, day_key, iso(utc_now())),
            )
        else:
            conn.execute(
                """
                INSERT INTO auto_scan_settings
                    (user_id, chat_id, enabled, symbols, night_resume, quota_resume, glm_calls_today, glm_calls_day, updated_at)
                VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    chat_id=excluded.chat_id,
                    enabled=excluded.enabled,
                    symbols=excluded.symbols,
                    night_resume=0,
                    quota_resume=excluded.quota_resume,
                    glm_calls_today=excluded.glm_calls_today,
                    glm_calls_day=excluded.glm_calls_day,
                    updated_at=excluded.updated_at
                """,
                (user_id, chat_id, 1 if effective_enabled else 0, symbols_text, quota_resume, calls, day_key, iso(utc_now())),
            )
        if not enabled:
            conn.execute(
                "UPDATE auto_scan_settings SET night_resume=0, quota_resume=0 WHERE user_id=?",
                (user_id,),
            )
        if symbols_text is not None:
            # Drop trend-skip state for symbols no longer being scanned, so switching back to a
            # symbol later starts fresh instead of reusing a days-old skip window.
            try:
                if normalized_symbols:
                    placeholders = ",".join("?" for _ in normalized_symbols)
                    conn.execute(
                        f"DELETE FROM auto_scan_trend_state WHERE user_id=? AND symbol NOT IN ({placeholders})",
                        (user_id, *normalized_symbols),
                    )
                else:
                    conn.execute("DELETE FROM auto_scan_trend_state WHERE user_id=?", (user_id,))
            except sqlite3.OperationalError:
                pass  # table not created yet (no analysis has run); nothing to clean up
        conn.commit()
    return {
        "enabled": effective_enabled,
        "quota_blocked": quota_blocked,
        "glm_calls_today": calls,
        "glm_calls_remaining": max(0, AUTOSCAN_MAX_PLANNER_CALLS_PER_DAY - calls),
    }

def get_auto_scan_status(user_id: int) -> dict:
    init_auto_scan_db()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT user_id, chat_id, enabled, symbols, night_resume, quota_resume, glm_calls_today, glm_calls_day, updated_at FROM auto_scan_settings WHERE user_id=?",
            (user_id,),
        ).fetchone()
    if not row:
        return {"enabled": False, "chat_id": None, "updated_at": None}
    day_key = _auto_scan_quota_day_key()
    calls = int(row[6] or 0) if str(row[7] or "") == day_key else 0
    return {
        "user_id": row[0], "chat_id": row[1], "enabled": bool(row[2]),
        "symbols": row[3] or "", "night_resume": bool(row[4]),
        "quota_resume": bool(row[5]), "glm_calls_today": calls,
        "glm_calls_remaining": max(0, AUTOSCAN_MAX_PLANNER_CALLS_PER_DAY - calls),
        "glm_calls_limit": AUTOSCAN_MAX_PLANNER_CALLS_PER_DAY, "updated_at": row[8],
    }


def maintain_auto_scan_daily_window(now: datetime | None = None) -> dict:
    """Manage the sleep window and daily quota for the Auto Scan day (07:00-06:59 VN).

    Key rules:
    - 00:00-07:00: only users who are currently enabled get paused, via ``night_resume=1``.
    - A user who ran out of quota keeps ``enabled=0, quota_resume=1`` for the rest of the day;
      they must never be re-enabled by a daytime scheduler tick.
    - Only once a new quota day starts at 07:00 does the call count reset to 0, and only then
      is a quota-paused user re-enabled.
    - A user who manually used /autoscanoff has both resume flags set to 0, so they don't get auto re-enabled.
    """
    init_auto_scan_db()
    current = now or utc_now()
    local_now = current.astimezone(VN_TZ)
    hour = local_now.hour
    sleep_hour = max(0, min(23, int(AUTOSCAN_SLEEP_HOUR_VN)))
    wake_hour = max(0, min(23, int(AUTOSCAN_WAKE_HOUR_VN)))
    in_sleep_window = (
        (sleep_hour <= hour < wake_hour)
        if sleep_hour < wake_hour
        else (hour >= sleep_hour or hour < wake_hour)
    )
    day_key = _auto_scan_quota_day_key(current)
    disabled = 0
    resumed = 0
    quota_reset = 0
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("BEGIN IMMEDIATE")

        # On a new quota day (the 07:00 VN mark): reset the final-AI call count.
        # Doesn't change the enabled/resume flags here; the section below decides who gets re-enabled.
        cur = conn.execute(
            """
            UPDATE auto_scan_settings
            SET glm_calls_today=0, glm_calls_day=?, updated_at=?
            WHERE glm_calls_day IS NULL OR glm_calls_day<>?
            """,
            (day_key, iso(current), day_key),
        )
        quota_reset = int(cur.rowcount or 0)

        if in_sleep_window:
            # Only flag a night-resume for users who are actually currently enabled.
            # A user who already ran out of quota has enabled=0/quota_resume=1, so their state isn't changed.
            cur = conn.execute(
                """
                UPDATE auto_scan_settings
                SET enabled=0, night_resume=1, updated_at=?
                WHERE enabled=1 AND quota_resume=0
                """,
                (iso(current),),
            )
            disabled = int(cur.rowcount or 0)
        else:
            # A user paused for the night is re-enabled once outside the sleep window.
            cur = conn.execute(
                """
                UPDATE auto_scan_settings
                SET enabled=1, night_resume=0, updated_at=?
                WHERE night_resume=1 AND quota_resume=0
                """,
                (iso(current),),
            )
            resumed += int(cur.rowcount or 0)

            # A user who ran out of quota is ONLY re-enabled after the quota day has reset.
            # The condition calls=0 + current day_key stops the daytime scheduler from wrongly re-enabling someone at 5/5.
            cur = conn.execute(
                """
                UPDATE auto_scan_settings
                SET enabled=1, quota_resume=0, night_resume=0, updated_at=?
                WHERE quota_resume=1
                  AND glm_calls_day=?
                  AND glm_calls_today=0
                """,
                (iso(current), day_key),
            )
            resumed += int(cur.rowcount or 0)

        conn.commit()
    return {
        "in_sleep_window": in_sleep_window,
        "disabled": disabled,
        "resumed": resumed,
        "quota_reset": quota_reset,
        "quota_day": day_key,
        "local_time": local_now.isoformat(),
        "sleep_hour": sleep_hour,
        "wake_hour": wake_hour,
    }


def get_auto_scan_glm_quota_state(user_id: int, now: datetime | None = None) -> dict:
    """Read the quota before any heavy work and lock Auto Scan if the quota is already used up.

    This function does not hold or deduct quota. It's an early guard so Binance or DeepSeek
    aren't called once a user has used up their GLM quota. ``reserve_auto_scan_glm_call`` is
    still where the quota is atomically incremented right before the GLM request.
    """
    init_auto_scan_db()
    current = now or utc_now()
    day_key = _auto_scan_quota_day_key(current)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT glm_calls_today, glm_calls_day FROM auto_scan_settings WHERE user_id=?",
            (user_id,),
        ).fetchone()
        calls = int(row[0] or 0) if row else 0
        stored_day = str(row[1] or "") if row else ""
        if stored_day != day_key:
            calls = 0
            if row:
                conn.execute(
                    """
                    UPDATE auto_scan_settings
                    SET glm_calls_today=0, glm_calls_day=?, updated_at=?
                    WHERE user_id=?
                    """,
                    (day_key, iso(current), user_id),
                )
        exhausted = calls >= AUTOSCAN_MAX_PLANNER_CALLS_PER_DAY
        if exhausted and row:
            conn.execute(
                """
                UPDATE auto_scan_settings
                SET enabled=0, quota_resume=1, glm_calls_today=?, glm_calls_day=?, updated_at=?
                WHERE user_id=?
                """,
                (calls, day_key, iso(current), user_id),
            )
        conn.commit()
    return {
        "allowed": not exhausted,
        "used": calls,
        "remaining": max(0, AUTOSCAN_MAX_PLANNER_CALLS_PER_DAY - calls),
        "exhausted": exhausted,
        "day": day_key,
    }


def reserve_auto_scan_glm_call(user_id: int) -> dict:
    """Reserve one GLM call slot for the user. The Nth call still runs, and Auto Scan then turns itself off."""
    init_auto_scan_db()
    day_key = _auto_scan_quota_day_key()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT glm_calls_today, glm_calls_day FROM auto_scan_settings WHERE user_id=?",
            (user_id,),
        ).fetchone()
        calls = int(row[0] or 0) if row else 0
        stored_day = str(row[1] or "") if row else ""
        if stored_day != day_key:
            calls = 0
        if calls >= AUTOSCAN_MAX_PLANNER_CALLS_PER_DAY:
            conn.execute(
                "UPDATE auto_scan_settings SET enabled=0, quota_resume=1, glm_calls_today=?, glm_calls_day=?, updated_at=? WHERE user_id=?",
                (calls, day_key, iso(utc_now()), user_id),
            )
            conn.commit()
            return {"allowed": False, "used": calls, "remaining": 0, "exhausted": True}
        new_calls = calls + 1
        exhausted = new_calls >= AUTOSCAN_MAX_PLANNER_CALLS_PER_DAY
        conn.execute(
            """
            UPDATE auto_scan_settings
            SET glm_calls_today=?, glm_calls_day=?, enabled=?, quota_resume=?, updated_at=?
            WHERE user_id=?
            """,
            (new_calls, day_key, 0 if exhausted else 1, 1 if exhausted else 0, iso(utc_now()), user_id),
        )
        conn.commit()
    return {
        "allowed": True, "used": new_calls,
        "remaining": max(0, AUTOSCAN_MAX_PLANNER_CALLS_PER_DAY - new_calls),
        "exhausted": exhausted,
    }

def _refund_auto_scan_glm_call(user_id: int) -> None:
    """Undo one reserved quota slot when the Planner call raises outright (timeout,
    bad config, sustained API outage) instead of returning a normal result — otherwise a run of
    failed calls silently burns the whole day's quota without producing a single signal.

    If this same call had just pushed the user into the quota-exhausted auto-disabled state
    (enabled=0, quota_resume=1), also re-enables Auto Scan — otherwise the refunded slot would
    sit unused until the next day's 07:00 VN reset even though quota is available again."""
    init_auto_scan_db()
    day_key = _auto_scan_quota_day_key()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT glm_calls_today, glm_calls_day, quota_resume FROM auto_scan_settings WHERE user_id=?",
            (user_id,),
        ).fetchone()
        if row:
            calls = int(row[0] or 0)
            stored_day = str(row[1] or "")
            was_quota_resume = bool(row[2])
            if stored_day == day_key and calls > 0:
                new_calls = calls - 1
                if was_quota_resume and new_calls < AUTOSCAN_MAX_PLANNER_CALLS_PER_DAY:
                    conn.execute(
                        """
                        UPDATE auto_scan_settings
                        SET glm_calls_today=?, enabled=1, quota_resume=0, updated_at=?
                        WHERE user_id=?
                        """,
                        (new_calls, iso(utc_now()), user_id),
                    )
                else:
                    conn.execute(
                        "UPDATE auto_scan_settings SET glm_calls_today=?, updated_at=? WHERE user_id=?",
                        (new_calls, iso(utc_now()), user_id),
                    )
        conn.commit()


def get_auto_scan_enabled_users() -> list[dict]:
    init_auto_scan_db()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT user_id, chat_id, symbols FROM auto_scan_settings WHERE enabled=1 AND chat_id IS NOT NULL ORDER BY user_id"
        ).fetchall()
    return [{"user_id": int(r[0]), "chat_id": int(r[1]), "symbols": r[2] or ""} for r in rows]


def _normalize_auto_scan_modes() -> list[str]:
    result = []
    for m in AUTOSCAN_MODES or ["short"]:
        mm = str(m).strip().lower()
        if mm in {"scalp", "short", "15m"}:
            result.append("short")
        elif mm in {"swing", "long", "4h"}:
            result.append("long")
    return result or ["short"]


def _auto_scan_symbols_from_env_or_db() -> list[str]:
    raw = os.getenv("AUTO_SCAN_SYMBOLS", "").strip()
    if raw:
        symbols = [normalize_auto_scan_symbol(x) for x in raw.split(",") if x.strip()]
    else:
        try:
            with sqlite3.connect(DB_PATH) as conn:
                rows = conn.execute("SELECT symbol FROM allowed_symbols ORDER BY symbol").fetchall()
            symbols = [normalize_auto_scan_symbol(r[0]) for r in rows]
        except Exception:
            symbols = []
    clean = []
    seen = set()
    for s in symbols:
        if s and s not in seen:
            clean.append(s)
            seen.add(s)
    return clean[:1]


def _parse_auto_scan_symbols_text(symbols_text: str | None) -> list[str]:
    raw = (symbols_text or "").strip()
    if not raw:
        return []
    parts = []
    for chunk in raw.replace(";", ",").split(","):
        for item in chunk.split():
            if item.strip():
                parts.append(item.strip())
    clean = []
    seen = set()
    for item in parts:
        sym = normalize_auto_scan_symbol(item)
        if sym and sym not in seen:
            clean.append(sym)
            seen.add(sym)
    return clean[:1]


def normalize_auto_scan_symbol(symbol: str) -> str:
    return resolve_binance_symbol(symbol)


def _record_auto_scan_signal(user_id: int, chat_id: int, symbol: str, mode: str, direction: str, confidence: int | None, prediction_id: int | None) -> None:
    init_auto_scan_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO auto_scan_signals (user_id, chat_id, symbol, mode, direction, confidence, sent_at, prediction_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, chat_id, symbol, mode, direction, confidence, iso(utc_now()), prediction_id),
        )
        conn.commit()


def _rollback_auto_scan_signal(prediction_id: int | None) -> None:
    """Undo the auto_scan_signals row when Telegram send ultimately fails, so the signal-history
    log doesn't record a signal the user never actually saw. The prediction itself stays in /history."""
    if prediction_id is None:
        return
    init_auto_scan_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM auto_scan_signals WHERE prediction_id=?", (prediction_id,))
        conn.commit()


def _auto_scan_state_get(key: str) -> str | None:
    init_auto_scan_db()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT value FROM auto_scan_state WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def _auto_scan_state_set(key: str, value: str) -> None:
    init_auto_scan_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO auto_scan_state (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """,
            (key, value, iso(utc_now())),
        )
        conn.commit()


def _auto_scan_interval_seconds() -> int:
    return max(60, int(AUTOSCAN_INTERVAL_SECONDS or 900))


def _auto_scan_slot_info(now: datetime | None = None) -> dict:
    now = (now or utc_now()).astimezone(timezone.utc)
    interval = _auto_scan_interval_seconds()
    delay = max(0, int(AUTOSCAN_CANDLE_CLOSE_DELAY_SECONDS or 0))
    epoch = int(now.timestamp())
    slot_epoch = (epoch // interval) * interval
    slot_dt = datetime.fromtimestamp(slot_epoch, tz=timezone.utc)
    next_slot_dt = datetime.fromtimestamp(slot_epoch + interval, tz=timezone.utc)
    due = epoch >= slot_epoch + delay
    return {
        "slot_epoch": slot_epoch,
        "slot": iso(slot_dt),
        "next_slot": iso(next_slot_dt),
        "due": due,
        "seconds_after_slot": epoch - slot_epoch,
        "delay_seconds": delay,
        "interval_seconds": interval,
    }


def should_run_auto_scan_now() -> tuple[bool, dict]:
    info = _auto_scan_slot_info()
    last_slot = _auto_scan_state_get("last_scan_slot")
    if not info.get("due"):
        info["skip_reason"] = f"waiting candle close delay {info.get('delay_seconds')}s"
        return False, info
    if last_slot == info.get("slot"):
        info["skip_reason"] = "slot already scanned"
        return False, info
    return True, info


def mark_auto_scan_slot_done(slot: str) -> None:
    _auto_scan_state_set("last_scan_slot", slot)
    _auto_scan_state_set("last_scan_at", iso(utc_now()))


def _auto_scan_format_dt(value: str | None) -> str:
    return format_vn_datetime(value) if value else "-"


def _record_auto_scan_log(
    user_id: int | None,
    chat_id: int | None,
    symbol: str,
    mode: str,
    *,
    scan_slot: str | None = None,
    stage: str,
    status: str,
    reason: str | None = None,
    final_direction: str | None = None,
    final_confidence: int | None = None,
    prediction_id: int | None = None,
) -> None:
    init_auto_scan_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO auto_scan_logs
                (user_id, chat_id, symbol, mode, scan_slot, scanned_at, stage, status,
                 final_direction, final_confidence, reason, prediction_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, chat_id, symbol, mode, scan_slot, iso(utc_now()), stage, status,
             final_direction, final_confidence, reason, prediction_id),
        )
        log_cutoff = iso(utc_now() - timedelta(days=AUTOSCAN_LOG_RETENTION_DAYS))
        conn.execute("DELETE FROM auto_scan_logs WHERE scanned_at < ?", (log_cutoff,))
        conn.commit()
    if AUTOSCAN_DEBUG:
        print(
            f"[AUTO_SCAN] log user={user_id} symbol={symbol} mode={mode} stage={stage} "
            f"status={status} final={final_direction}/{final_confidence} reason={reason}",
            flush=True,
        )


def get_auto_scan_logs(user_id: int, limit: int | None = None) -> list[dict]:
    init_auto_scan_db()
    limit = max(1, min(AUTO_SCAN_LOG_LIMIT, int(limit or AUTO_SCAN_LOG_LIMIT)))
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT scanned_at, symbol, mode, stage, status,
                   final_direction, final_confidence, reason, prediction_id
            FROM auto_scan_logs
            WHERE user_id=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    keys = [
        "scanned_at", "symbol", "mode", "stage", "status",
        "final_direction", "final_confidence", "reason", "prediction_id",
    ]
    return [dict(zip(keys, row)) for row in rows]


def get_auto_scan_runtime_status(user_id: int) -> dict:
    window = maintain_auto_scan_daily_window()
    status = get_auto_scan_status(user_id)
    slot = _auto_scan_slot_info()
    logs = get_auto_scan_logs(user_id, limit=1)
    return {
        **status,
        "last_scan_slot": _auto_scan_state_get("last_scan_slot"),
        "last_scan_at": _auto_scan_state_get("last_scan_at"),
        "current_slot": slot.get("slot"),
        "next_scan_at": slot.get("next_slot"),
        "last_log": logs[0] if logs else None,
        "in_sleep_window": bool(window.get("in_sleep_window")),
        "sleep_hour_vn": int(window.get("sleep_hour", AUTOSCAN_SLEEP_HOUR_VN)),
        "wake_hour_vn": int(window.get("wake_hour", AUTOSCAN_WAKE_HOUR_VN)),
    }



def _auto_scan_text_header(symbol: str, mode: str) -> str:
    mode_label = "SCALP" if mode == "short" else "SWING"
    return f"🤖 AUTO SCAN — {symbol} — {mode_label}\n"


def _strip_public_evidence_for_user(output: str) -> str:
    """Hide the Evidence blocks from every public message (both Manual and Auto Scan).

    The planner still returns the full content; the DB still receives/saves the full,
    unedited full_response. Only the final text sent to Telegram is trimmed, from right after
    Activation straight to Risk.
    """
    lines = (output or "").splitlines()
    kept: list[str] = []
    buffered: list[str] = []  # Lines held during evidence-skip window
    skipping = False
    for line in lines:
        normalized = line.strip().lower()
        if not skipping and normalized.startswith("bằng chứng entry"):
            skipping = True
            buffered = []
            continue
        if skipping:
            if normalized.startswith("⚠️ rủi ro") or normalized.startswith("rủi ro"):
                # Found the Risk section — discard buffered evidence lines (correct behavior)
                skipping = False
                buffered = []
                kept.append(line)
            else:
                # Hold in buffer; will be restored if Risk section is never found
                buffered.append(line)
            continue
        kept.append(line)
    # Safety: if output ended while still in evidence-skip window (no Risk section found),
    # restore buffered lines so the user sees the full content rather than nothing.
    if skipping:
        kept.extend(buffered)
    # Avoid leaving too many blank lines after removing a long block.
    compact: list[str] = []
    for line in kept:
        if line.strip() or not compact or compact[-1].strip():
            compact.append(line)
    return "\n".join(compact).strip()


async def auto_scan_symbol_for_user(symbol: str, mode: str, user_id: int, chat_id: int, scan_slot: str | None = None) -> dict:
    """Run 1 symbol/mode for 1 user. Return {send: bool, text: str}."""
    init_prediction_db()
    init_auto_scan_db()
    binance_symbol = normalize_auto_scan_symbol(symbol)
    if not binance_symbol:
        return {"send": False, "reason": "empty symbol"}

    async def log_and_return(stage: str, status: str, reason: str, **kwargs) -> dict:
        await asyncio.to_thread(
            _record_auto_scan_log,
            user_id, chat_id, binance_symbol, mode,
            scan_slot=scan_slot, stage=stage, status=status, reason=reason, **kwargs,
        )
        return {"send": False, "reason": reason, "stage": stage, "status": status, **kwargs}

    # The quota guard MUST run before Binance and Planner.
    # This way, once a user hits N/N, their entire Auto Scan truly stops until 07:00.
    quota_state = await asyncio.to_thread(get_auto_scan_glm_quota_state, user_id)
    if not quota_state.get("allowed"):
        return await log_and_return(
            "quota",
            "skipped",
            f"Đã dùng đủ {AUTOSCAN_MAX_PLANNER_CALLS_PER_DAY} lượt gọi AI cuối trong ngày Auto Scan; sẽ tự bật lại lúc 07:00 VN.",
        )

    # Cost optimization: 2 consecutive scans that both came back the same LONG or SHORT already
    # confirmed the trend, so the next 2 cycles skip Binance + Planner entirely instead of paying
    # for a read that's very likely to repeat. See _auto_scan_update_trend_state for the trigger.
    skip_remaining = await asyncio.to_thread(_auto_scan_consume_trend_skip, user_id, binance_symbol, mode)
    if skip_remaining is not None:
        return await log_and_return(
            "trend", "skipped",
            f"2 lần quét liên tiếp đã cùng hướng, xu hướng coi như đã xác định; bỏ qua quét để tiết kiệm chi phí, còn {skip_remaining} lần bỏ qua.",
        )

    timeframe_data = await collect_timeframe_data(binance_symbol, mode)
    if not any(df is not None and not df.empty for df in timeframe_data.values()):
        return await log_and_return("binance", "error", "no binance data")

    missing_critical = _missing_critical_timeframes(timeframe_data, mode)
    if missing_critical:
        return await log_and_return(
            "binance", "error", f"thiếu dữ liệu khung quan trọng: {', '.join(missing_critical)}"
        )

    # Auto Scan uses the exact same context builder as manual analysis.
    ctx = await prepare_analysis_context(
        binance_symbol,
        mode,
        user_id=user_id,
        timeframe_data=timeframe_data,
    )
    system_prompt = ctx["system_prompt"]
    current_price = ctx["current_price"]
    feature_snapshot = ctx["feature_snapshot"]
    market_snapshot = ctx["market_snapshot"]

    quota = await asyncio.to_thread(reserve_auto_scan_glm_call, user_id)
    if not quota.get("allowed"):
        return await log_and_return(
            "quota", "skipped",
            f"Đã dùng đủ {AUTOSCAN_MAX_PLANNER_CALLS_PER_DAY} lượt gọi AI cuối trong ngày Auto Scan; sẽ tự bật lại lúc 07:00 VN.",
        )

    user_prompt = ctx["user_prompt"]
    # Auto Scan reframes the Planner's task: not "design the best plan for the coming hours" (which
    # always produces a pullback plan waiting on a future trigger — historically 100% of plans came
    # back as SETUP_WAITING_TRIGGER, never once READY_TO_ENTER) but "can this be entered right now?".
    # A prior version of this note also reassured the model that a missed setup gets re-evaluated on
    # the next scan, telling it not to lower its own standard just to produce a plan now — removed as
    # steering the model's decision behavior, not just reframing the task. Known tradeoff: without
    # that reassurance the model may lean toward forcing a marginal setup into READY_TO_ENTER rather
    # than NO_TRADE, now that SETUP_WAITING_TRIGGER is off the table below — worth watching for.
    flash_note = "\n\nBỐI CẢNH AUTO SCAN — CÂU HỎI BẠN PHẢI TRẢ LỜI:\n" + (
        "- Đây KHÔNG phải yêu cầu 'thiết kế kế hoạch tốt nhất cho vài giờ tới'. Câu hỏi duy nhất là: NGAY BÂY GIỜ, tại mức giá hiện tại, có vào lệnh được không?\n"
        "- Người nhận plan sẽ vào lệnh ngay khi đọc được, không theo dõi biểu đồ và không tự canh trigger.\n"
        "- Chỉ dùng hai trạng thái: READY_TO_ENTER (đúng nghĩa đã định nghĩa ở trên — vào lệnh được ngay) hoặc NO_TRADE. Không dùng SETUP_WAITING_TRIGGER trong luồng này; nếu setup chưa sẵn sàng để vào ngay bây giờ theo phán đoán của riêng bạn, trả NO_TRADE."
    )
    planner_input = user_prompt + flash_note
    try:
        raw_output = await asyncio.to_thread(request_claude_analysis, system_prompt, planner_input)
        planner_clean = (raw_output or "").strip()
    except Exception:
        # The quota slot was already reserved above; refund it so a Planner outage
        # (timeout, bad config, sustained API error) doesn't silently burn the day's quota.
        await asyncio.to_thread(_refund_auto_scan_glm_call, user_id)
        raise
    output = ensure_current_price_line(sanitize_user_output(planner_clean), current_price)
    pred = parse_prediction_from_output(output)
    direction = (pred.get("direction") or "").upper()
    # Record this scan's direction for the next cycle's trend-skip check, regardless of what
    # happens to the plan afterward (guard rejection, send failure, etc.) — this tracks the
    # model's own directional read, not whether a plan actually got sent.
    await asyncio.to_thread(_auto_scan_update_trend_state, user_id, binance_symbol, mode, direction)
    await asyncio.to_thread(
        _save_analysis_snapshot,
        user_id=user_id, chat_id=chat_id, symbol=binance_symbol, mode=mode, source="autoscan",
        model=get_ai_model_name(),
        planner_input=planner_input, planner_output=planner_clean,
        setup_status=_extract_setup_status(output),
        current_price=current_price, public_output=output,
        funding_context=ctx.get("funding_context"), open_interest_context=ctx.get("open_interest_context"),
        long_short_context=ctx.get("long_short_context"),
        btc_context_text=build_btc_correlation_block(ctx.get("btc_context")),
    )
    final_conf = pred.get("signal_score")
    if final_conf is None:
        final_conf = pred.get("confidence")
    final_conf = int(final_conf) if final_conf is not None else None

    setup_status = _extract_setup_status(output)
    # Auto Scan re-analyzes from scratch every scan cycle, so there is no reason to send the user
    # a "wait for this trigger" plan — the runtime instruction appended to the Planner call already
    # tells it to answer NO_TRADE instead of SETUP_WAITING_TRIGGER when the entry hasn't already
    # objectively happened. This is the Python-side safety net for the rare case it answers
    # SETUP_WAITING_TRIGGER anyway: treated exactly like NO_TRADE, never saved or sent. Only a plan
    # whose trigger has already happened (READY_TO_ENTER) reaches the send path below.
    if direction == "NO_TRADE" or setup_status == "SETUP_WAITING_TRIGGER":
        if direction == "NO_TRADE" and AUTOSCAN_SEND_NO_TRADE:
            return {"send": True, "text": _auto_scan_text_header(binance_symbol, mode) + output, "prediction_id": None}
        reason = (
            "Planner chọn NO TRADE sau phân tích đầy đủ." if direction == "NO_TRADE"
            else "Planner ra plan chờ trigger; không gửi vì Auto Scan chỉ gửi lệnh vào được ngay. Đợi chu kỳ quét sau."
        )
        return await log_and_return("planner", "rejected", reason, final_direction=direction, final_confidence=final_conf)

    if direction not in {"LONG", "SHORT"}:
        return await log_and_return("planner", "rejected", "Planner không trả quyết định LONG/SHORT hợp lệ.", final_direction=direction, final_confidence=final_conf)

    guard_errors = _validate_actionable_trade_plan(pred, timeframe_data, mode, current_price, output)
    if guard_errors:
        # Guard failures are pure output-format slips (a level Python could not read, a missing
        # status label) — try one cheap repair reusing the existing analysis before discarding it.
        # The repair may not touch any price.
        repaired_clean, repaired_pred, remaining_errors = await _repair_planner_format(
            system_prompt, planner_clean, guard_errors, timeframe_data, mode, current_price
        )
        if not remaining_errors:
            planner_clean = repaired_clean
            output = ensure_current_price_line(sanitize_user_output(planner_clean), current_price)
            pred = repaired_pred
            # The repair prompt is told not to change direction, but nothing enforces that —
            # re-derive from the repaired plan so a stale pre-repair value can't get persisted
            # against post-repair Entry/SL/TP and corrupt win/loss tracking downstream.
            direction = (pred.get("direction") or "").upper()
            guard_errors = []
        else:
            guard_errors = remaining_errors

    if guard_errors:
        log_hidden_rejection(binance_symbol, mode, pred, guard_errors, output)
        return await log_and_return("guard", "rejected", "guard rejected", final_direction=direction, final_confidence=final_conf)

    can_track = all(pred.get(k) is not None for k in ("entry_low", "entry_high", "sl", "tp1"))
    if not can_track:
        return await log_and_return("planner", "rejected", "Planner thiếu Entry/SL/TP bắt buộc", final_direction=direction, final_confidence=final_conf)

    reasoning_summary = build_local_reasoning_summary(output)
    prediction_id = await asyncio.to_thread(
        save_prediction,
        symbol=binance_symbol,
        mode=mode,
        direction=direction,
        entry_low=pred.get("entry_low"),
        entry_high=pred.get("entry_high"),
        sl=pred.get("sl"),
        tp1=pred.get("tp1"),
        tp2=pred.get("tp2"),
        market_snapshot=market_snapshot,
        feature_snapshot=feature_snapshot,
        reasoning_summary=reasoning_summary,
        full_response=output,
        user_id=user_id,
        chat_id=chat_id,
        setup_status=setup_status,
    )
    try:
        if _price_in_entry_range(current_price, pred.get("entry_low"), pred.get("entry_high")):
            entry_price = _entry_price(direction, pred.get("entry_low"), pred.get("entry_high"), current_price)
            if entry_price is not None:
                await asyncio.to_thread(mark_entry_filled, prediction_id, float(entry_price), utc_now(), mode)
    except Exception:
        pass

    await asyncio.to_thread(_record_auto_scan_signal, user_id, chat_id, binance_symbol, mode, direction, final_conf, int(prediction_id))
    strength_index = await asyncio.to_thread(_btc_eth_strength_index)
    output = _insert_btc_strength_line(output, strength_index)
    execution_note = "\n\n✅ Trigger đã sẵn sàng; có thể thực thi theo kế hoạch trong vùng Entry."
    public_output = _strip_public_evidence_for_user(output)
    text = (
        _auto_scan_text_header(binance_symbol, mode)
        + public_output
        + execution_note
        + "\n\nBot đã tự lưu tín hiệu Auto Scan này để theo dõi."
    )
    return {
        "send": True,
        "text": text,
        "prediction_id": int(prediction_id),
        "direction": direction,
        "confidence": final_conf,
        "final_direction": direction,
        "final_confidence": final_conf,
    }


async def _run_auto_scan_cycle(bot=None, force: bool = False) -> dict:
    """Run exactly one Auto Scan candle slot without overlap/catch-up handling."""
    window = await asyncio.to_thread(maintain_auto_scan_daily_window)
    if window.get("in_sleep_window") and not force:
        return {
            "users": 0, "symbols": 0, "modes": _normalize_auto_scan_modes(),
            "sent": 0, "checked": 0, "errors": 0, "skipped": True,
            "reason": f"daily sleep window {window.get('sleep_hour'):02d}:00-{window.get('wake_hour'):02d}:00 VN",
            "next_scan_at": None,
        }
    should_run, slot_info = should_run_auto_scan_now()
    if not force and not should_run:
        return {"users": 0, "symbols": 0, "modes": _normalize_auto_scan_modes(), "sent": 0, "checked": 0, "errors": 0, "skipped": True, "reason": slot_info.get("skip_reason"), "next_scan_at": slot_info.get("next_slot")}

    users = await asyncio.to_thread(get_auto_scan_enabled_users)
    modes = _normalize_auto_scan_modes()
    payload = {"users": len(users), "symbols": 0, "modes": modes, "sent": 0, "checked": 0, "errors": 0, "skipped": False, "slot": slot_info.get("slot"), "next_scan_at": slot_info.get("next_slot")}
    if not users:
        try:
            await asyncio.to_thread(mark_auto_scan_slot_done, slot_info.get("slot") or iso(utc_now()))
        except Exception as exc:
            print(f"[AUTO_SCAN] mark_slot_done lỗi (slot có thể bị scan lại): {exc}", flush=True)
        return payload
    for user in users:
        symbols = _parse_auto_scan_symbols_text(user.get("symbols")) or _auto_scan_symbols_from_env_or_db()
        payload["symbols"] += len(symbols)
        if not symbols:
            continue
        for symbol in symbols:
            for mode in modes:
                payload["checked"] += 1
                try:
                    result = await auto_scan_symbol_for_user(symbol, mode, user["user_id"], user["chat_id"], scan_slot=slot_info.get("slot"))
                    if result.get("send") and result.get("text") and bot is not None:
                        send_exc = None
                        sent_ok = False
                        for send_attempt in range(3):
                            try:
                                await bot.send_message(chat_id=user["chat_id"], text=result["text"])
                                sent_ok = True
                                break
                            except Exception as exc:
                                send_exc = exc
                                if send_attempt < 2:
                                    await asyncio.sleep(2.0 * (send_attempt + 1))
                        if sent_ok:
                            # A valid Auto Scan signal has already been saved into predictions (/history) above.
                            # After the Telegram message sends successfully, also save a separate record into
                            # auto_scan_logs so the signal also shows up in /autoscanlog.
                            await asyncio.to_thread(
                                _record_auto_scan_log,
                                user.get("user_id"),
                                user.get("chat_id"),
                                normalize_auto_scan_symbol(symbol),
                                mode,
                                scan_slot=slot_info.get("slot"),
                                stage="sent",
                                status="sent",
                                reason="Đã gửi tín hiệu Auto Scan và lưu đồng thời vào history cùng Auto Scan log.",
                                final_direction=result.get("final_direction") or result.get("direction"),
                                final_confidence=result.get("final_confidence") if result.get("final_confidence") is not None else result.get("confidence"),
                                prediction_id=result.get("prediction_id"),
                            )
                            payload["sent"] += 1
                        else:
                            # Telegram send failed after retries: the prediction stays in /history (so it's
                            # not lost), but the auto_scan_signals row is rolled back so the signal-history
                            # log doesn't record a signal the user never actually saw.
                            await asyncio.to_thread(_rollback_auto_scan_signal, result.get("prediction_id"))
                            payload["errors"] += 1
                            await asyncio.to_thread(
                                _record_auto_scan_log,
                                user.get("user_id"), user.get("chat_id"), normalize_auto_scan_symbol(symbol), mode,
                                scan_slot=slot_info.get("slot"), stage="sent_failed", status="error",
                                reason=f"Gửi Telegram thất bại sau 3 lần thử: {str(send_exc)[:300]}",
                                prediction_id=result.get("prediction_id"),
                            )
                            print(
                                f"[AUTO_SCAN_SEND_FAILED] user={user.get('user_id')} symbol={symbol} mode={mode} "
                                f"prediction_id={result.get('prediction_id')} error={send_exc}",
                                flush=True,
                            )
                except Exception as exc:
                    payload["errors"] += 1
                    await asyncio.to_thread(
                        _record_auto_scan_log,
                        user.get("user_id"), user.get("chat_id"), symbol, mode,
                        scan_slot=slot_info.get("slot"), stage="error", status="error", reason=str(exc)[:500],
                    )
                    print(f"[AUTO_SCAN] error user={user.get('user_id')} symbol={symbol} mode={mode}: {exc}", flush=True)
    try:
        await asyncio.to_thread(mark_auto_scan_slot_done, slot_info.get("slot") or iso(utc_now()))
    except Exception as exc:
        print(f"[AUTO_SCAN] mark_slot_done lỗi (slot có thể bị scan lại): {exc}", flush=True)
    return payload


async def run_auto_scan_once(bot=None, force: bool = False) -> dict:
    """Run Auto Scan without overlap and catch up only the newest missed slot.

    A scheduler tick that arrives while another scan is active returns immediately.
    The active runner checks for a newer due candle slot after completion and may
    process exactly one newest catch-up slot. It never queues every missed slot.
    """
    if _AUTO_SCAN_RUN_LOCK.locked():
        info = _auto_scan_slot_info()
        print(
            f"[AUTO_SCAN_OVERLAP_SKIP] active_run=1 latest_slot={info.get('slot')} "
            "catch_up_by_active_run=1",
            flush=True,
        )
        return {
            "users": 0,
            "symbols": 0,
            "modes": _normalize_auto_scan_modes(),
            "sent": 0,
            "checked": 0,
            "errors": 0,
            "skipped": True,
            "reason": "previous Auto Scan cycle still active; active cycle will check newest slot",
            "next_scan_at": info.get("next_slot"),
            "overlap": True,
        }

    async with _AUTO_SCAN_RUN_LOCK:
        started = utc_now()
        first = await _run_auto_scan_cycle(bot=bot, force=force)
        aggregate = dict(first)
        aggregate["catch_up_runs"] = 0

        if force or first.get("skipped"):
            aggregate["elapsed_seconds"] = round((utc_now() - started).total_seconds(), 1)
            return aggregate

        completed_slot = first.get("slot")
        latest_info = _auto_scan_slot_info()
        should_catch_up, due_info = should_run_auto_scan_now()
        if should_catch_up and due_info.get("slot") != completed_slot:
            print(
                f"[AUTO_SCAN_CATCH_UP] completed_slot={completed_slot} "
                f"latest_slot={due_info.get('slot')} skipped_intermediate_slots=1",
                flush=True,
            )
            catch = await _run_auto_scan_cycle(bot=bot, force=False)
            aggregate["catch_up_runs"] = 1
            aggregate["catch_up_slot"] = catch.get("slot")
            for key in ("users", "symbols", "sent", "checked", "errors"):
                aggregate[key] = int(aggregate.get(key, 0) or 0) + int(catch.get(key, 0) or 0)
            aggregate["next_scan_at"] = catch.get("next_scan_at") or latest_info.get("next_slot")

        aggregate["elapsed_seconds"] = round((utc_now() - started).total_seconds(), 1)
        print(
            f"[AUTO_SCAN_RUN_COMPLETE] slot={aggregate.get('slot')} "
            f"catch_up_runs={aggregate.get('catch_up_runs', 0)} "
            f"elapsed={aggregate.get('elapsed_seconds')}s",
            flush=True,
        )
        return aggregate

