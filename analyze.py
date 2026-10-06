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
    AUTOSCAN_LOG_RETENTION_DAYS,
    ENTRY_WAIT_HOURS,
    TRADE_MAX_HOLD_HOURS,
    cleanup_evaluation_data,
    clear_evaluation_data,
    prompt_hash,
    save_evaluation_case,
)
from mode_migration import migrate_mode_values

load_dotenv()

# Futures mode uses Binance USDⓈ-M endpoints; Spot mode uses Binance Spot endpoints.
BINANCE_FUTURES_API_BASE = "https://fapi.binance.com"
BINANCE_FUTURES_KLINES_URL = f"{BINANCE_FUTURES_API_BASE}/fapi/v1/klines"
BINANCE_FUTURES_DEPTH_URL = f"{BINANCE_FUTURES_API_BASE}/fapi/v1/depth"
BINANCE_SPOT_API_BASE = "https://api.binance.com"
BINANCE_SPOT_KLINES_URL = f"{BINANCE_SPOT_API_BASE}/api/v3/klines"
BINANCE_SPOT_DEPTH_URL = f"{BINANCE_SPOT_API_BASE}/api/v3/depth"
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


def resolve_binance_symbol(raw: str, market: str = "futures") -> str:
    """Normalize a symbol for the selected Binance market (futures or spot)."""
    s = (raw or "").strip().lstrip("/").upper()
    if not s:
        return ""
    for quote in ("USDT", "USDC"):
        if s.endswith(quote):
            s = s[:-len(quote)]
            break
    if market == "spot":
        reverse_aliases = {value: key for key, value in BINANCE_FUTURES_SYMBOL_ALIASES.items()}
        s = reverse_aliases.get(s, s)
    else:
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
# Default "high": token reasoning dùng chung hạn mức PLANNER_MAX_OUTPUT_TOKENS với câu trả lời.
PLANNER_REASONING_EFFORT = os.getenv("PLANNER_REASONING_EFFORT", "high").strip()
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
AUTOSCAN_MODES = [m.strip().lower() for m in os.getenv("AUTOSCAN_MODES", "futures").split(",") if m.strip()]
AUTOSCAN_SEND_NO_TRADE = os.getenv("AUTOSCAN_SEND_NO_TRADE", "0").strip().lower() in {"1", "true", "yes", "on"}
AUTOSCAN_CANDLE_CLOSE_DELAY_SECONDS = int(os.getenv("AUTOSCAN_CANDLE_CLOSE_DELAY_SECONDS", "5"))
# Job scheduler only wakes up to check whether a candle-close slot is due.
# It does NOT call Binance/LLM unless should_run_auto_scan_now() returns true.
AUTOSCAN_SCHEDULER_TICK_SECONDS = max(30, int(os.getenv("AUTOSCAN_SCHEDULER_TICK_SECONDS", "60") or "60"))
# Log retention nhận từ evaluation_store — trước đây định nghĩa 2 lần với 2 fallback env
# khác nhau, nên retention phụ thuộc vào đường code nào chạy sau.
AUTOSCAN_DEBUG = os.getenv("AUTOSCAN_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}

# Prevent overlapping Auto Scan cycles. If a candle-close slot arrives while a cycle
# is still running, the active cycle performs at most one catch-up pass for the
# newest closed slot after it finishes. Older missed slots are intentionally skipped.
_AUTO_SCAN_RUN_LOCK = asyncio.Lock()
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

# The fetch windows below are each frame's raw-candle count for indicator warm-up
# (see SPOT_TIMEFRAMES; Futures mode uses FUTURES_TIMEFRAMES ở trên).
def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return default


# ─── Futures (mode "futures") config ───────────────────────────────────────────
# Mode "futures" hiển thị cho user là FUTURES, dùng 3 khung 4H/1H/15m.
# Tên nội bộ "futures" giữ nguyên (DB schema, lifecycle, quota không đổi).
FUTURES_TIMEFRAMES = {
    "4H":  ("4h",  _env_int("FUTURES_FETCH_4H", 300)),
    "1H":  ("1h",  _env_int("FUTURES_FETCH_1H", 300)),
    "15m": ("15m", _env_int("FUTURES_FETCH_15M", 300)),
}


def _parse_futures_display() -> dict[str, int]:
    defaults = {"4H": 30, "1H": 48, "15m": 64}
    raw = (os.getenv("FUTURES_DISPLAY", "") or "").strip()
    if not raw:
        return defaults
    try:
        for chunk in raw.replace(";", ",").split(","):
            if ":" not in chunk:
                continue
            key, value = chunk.split(":", 1)
            key = key.strip()
            if key in defaults:
                defaults[key] = max(1, int(float(value.strip())))
    except Exception:
        pass
    return defaults


FUTURES_DISPLAY = _parse_futures_display()
# Chỉ N nến đã đóng gần nhất mỗi khung được in đủ cột vr/tb%/rng/cl%; nến cũ hơn in rút gọn.
FUTURES_FULL_COLS_N = _env_int("FUTURES_FULL_COLS_N", 24)

# Ngưỡng kiểm tra số học cho plan Futures.
MIN_RR = _env_float("MIN_RR", 1.0)
SL_ATR_MIN = _env_float("SL_ATR_MIN", 0.6)
SL_ATR_MAX = _env_float("SL_ATR_MAX", 3.0)
ENTRY_READY_ATR15 = _env_float("ENTRY_READY_ATR15", 0.25)
CITE_REL_TOL = _env_float("CITE_REL_TOL", 0.005)
# Giới hạn % giá cho khoảng cách Entry→SL (quy tắc MAX_SL_PCT trong prompt/validator).
MAX_SL_PCT = _env_float("MAX_SL_PCT", 2.0)
_ORDERBOOK_DEPTH_ALLOWED = (5, 10, 20, 50, 100, 500, 1000)
_orderbook_depth_requested = _env_int("ORDERBOOK_DEPTH_LIMIT", 100)
ORDERBOOK_DEPTH_LIMIT = _orderbook_depth_requested if _orderbook_depth_requested in _ORDERBOOK_DEPTH_ALLOWED else 100
ORDERBOOK_BANDS_PCT = (0.10, 0.25, 0.50)
ORDERBOOK_ZONE_BIN_PCT = 0.05

SPOT_TIMEFRAMES = {
    # Weekly context, daily structure, 4H trigger. Fetch enough history for EMA/ADX/volume
    # warm-up; only a compact recent window is sent to the planner (see _v50_raw_limit).
    "4H": ("4h", _env_int("SPOT_FETCH_4H", 300)),
    "1D": ("1d", _env_int("SPOT_FETCH_1D", 300)),
    "1W": ("1w", _env_int("SPOT_FETCH_1W", 300)),
}


def _parse_spot_display() -> dict[str, int]:
    defaults = {"4H": 30, "1D": 48, "1W": 64}
    raw = (os.getenv("SPOT_DISPLAY", "") or "").strip()
    if not raw:
        return defaults
    try:
        for chunk in raw.replace(";", ",").split(","):
            if ":" not in chunk:
                continue
            key, value = chunk.split(":", 1)
            key = key.strip()
            if key in defaults:
                defaults[key] = max(1, int(float(value.strip())))
    except Exception:
        pass
    return defaults


SPOT_DISPLAY = _parse_spot_display()
SPOT_FULL_COLS_N = max(1, _env_int("SPOT_FULL_COLS_N", 24))

# Lifecycle is stored per mode: futures or spot
# (ENTRY_WAIT_HOURS / TRADE_MAX_HOLD_HOURS are imported from evaluation_store.py above -
# the single source of truth, to avoid the hour mismatch between the two modules that happened before.)

CHECK_INTERVAL_HOURS = {
    "futures": 0.5,     # Futures: check every 30min, matching job_check_predictions' own 30min interval
    "spot": 12,       # Spot: check every 12h
}

RESULT_CHECK_INTERVAL = {
    "futures": "5m",    # Futures: chấm chạm SL/TP bằng nến 5m
    "spot": "1h",     # Spot: score the outcome using 1-hour candles
}


def get_result_check_interval(mode: str) -> str:
    return RESULT_CHECK_INTERVAL.get(mode, "15m")

VISIBLE_PREDICTION_RETENTION_LIMIT = 10
HIDDEN_LEARNING_RETENTION_LIMIT = 5
# Predictions still being tracked (đang chờ entry / đã khớp, chờ chấm SL-TP). Không bao giờ
# được prune: xóa chúng thì job auto-check không còn thấy vị thế → không bao giờ ra WIN/LOSS,
# và auto_scan_signals.prediction_id thành tham chiếu treo.
OPEN_PREDICTION_RESULTS = ("PENDING_ENTRY", "ENTRY_FILLED")
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

        # Indexes actually used: history/prune (user_id, id) + auto-check (result, next_check_at).
        # Đã gỡ idx_predictions_user_symbol_mode_id — không còn query nào filter symbol/mode
        # (bỏ /stats theo symbol, /history hiện mọi coin).
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_user_id_id ON predictions(user_id, id DESC)")
        conn.execute("DROP INDEX IF EXISTS idx_predictions_user_symbol_mode_id")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_result_next_check ON predictions(result, next_check_at)")
        migrate_mode_values(conn)

        # Retention: mỗi user giữ 10 dòng terminal mới nhất (dòng ẩn giữ 5) — chỉ đếm dòng
        # terminal, nên lệnh đang mở không chiếm chỗ và không bao giờ bị xóa.
        hidden_a, hidden_b = HIDDEN_LEARNING_RESULTS
        open_a, open_b = OPEN_PREDICTION_RESULTS
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
                      AND result NOT IN (?, ?)
                ) ranked
                WHERE keep_rank > ?
            )
            """,
            (hidden_a, hidden_b, open_a, open_b, VISIBLE_PREDICTION_RETENTION_LIMIT),
        )
        conn.commit()
    _prediction_db_initialized = True


def prune_prediction_history(user_id: int | None) -> None:
    """Keep the DB lean: each user only keeps the 10 most recent finished trades.

    - /history only uses the visible group, so that group is kept at exactly the 10 newest rows.
    - NO_TRADE/REJECTED_PLAN are hidden learning records, not shown in /history; they're still
      limited separately so the DB doesn't grow unbounded over time.
    - Lệnh đang mở (PENDING_ENTRY/ENTRY_FILLED) KHÔNG bị đếm và KHÔNG bị xóa — chúng phải
      tồn tại tới khi job auto-check chấm kết quả.
    """
    if user_id is None:
        return

    hidden_a, hidden_b = HIDDEN_LEARNING_RESULTS
    open_a, open_b = OPEN_PREDICTION_RESULTS
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            DELETE FROM predictions
            WHERE user_id=?
              AND result NOT IN (?, ?)
              AND result NOT IN (?, ?)
              AND id NOT IN (
                  SELECT id
                  FROM predictions
                  WHERE user_id=?
                    AND result NOT IN (?, ?)
                    AND result NOT IN (?, ?)
                  ORDER BY id DESC
                  LIMIT ?
              )
            """,
            (user_id, hidden_a, hidden_b, open_a, open_b,
             user_id, hidden_a, hidden_b, open_a, open_b, VISIBLE_PREDICTION_RETENTION_LIMIT),
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
    """setup_status lưu nhãn hai trạng thái (TRADE/NO_TRADE) tại thời điểm tạo plan để truy ngược."""
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
    - force=True: fetches all PENDING_ENTRY/ENTRY_FILLED rows regardless of next_check_at.
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

def get_current_price_raw(symbol: str, market: str = "futures") -> float | None:
    base = BINANCE_SPOT_API_BASE if market == "spot" else BINANCE_FUTURES_API_BASE
    path = "/api/v3/ticker/price" if market == "spot" else "/fapi/v1/ticker/price"
    r = _binance_get_with_retry(
        f"{base}{path}", {"symbol": symbol}, max_retries=1, timeout=10
    )
    if r is None:
        return None
    try:
        return float(r.json()["price"])
    except Exception:
        return None


def fetch_orderbook_snapshot(symbol: str, market: str = "futures") -> dict | None:
    """Fetch the public visible order book from the same market used by this analysis."""
    is_spot = market == "spot"
    url = BINANCE_SPOT_DEPTH_URL if is_spot else BINANCE_FUTURES_DEPTH_URL
    response = _binance_get_with_retry(
        url, {"symbol": symbol, "limit": ORDERBOOK_DEPTH_LIMIT}, max_retries=1, timeout=10
    )
    if response is None:
        return None
    try:
        payload = response.json()
        bids = [(float(level[0]), float(level[1])) for level in payload.get("bids", [])]
        asks = [(float(level[0]), float(level[1])) for level in payload.get("asks", [])]
        bids = [(price, qty) for price, qty in bids if price > 0 and qty > 0]
        asks = [(price, qty) for price, qty in asks if price > 0 and qty > 0]
        if not bids or not asks:
            return None
        return {
            "market": "SPOT" if is_spot else "FUTURES",
            "sampled_at": utc_now(),
            "bids": bids,
            "asks": asks,
            "requested_levels": ORDERBOOK_DEPTH_LIMIT,
        }
    except Exception as exc:
        print(f"Binance order book parse error {symbol} market={market}: {exc}", flush=True)
        return None


def summarize_orderbook(snapshot: dict | None, ticker_price: float | None = None) -> str | None:
    """Compress a book snapshot into quote-notional depth bands; this is not a stop map."""
    if not snapshot:
        return None
    bids, asks = snapshot.get("bids") or [], snapshot.get("asks") or []
    if not bids or not asks:
        return None
    best_bid = max(price for price, _ in bids)
    best_ask = min(price for price, _ in asks)
    midpoint = (best_bid + best_ask) / 2.0
    if midpoint <= 0:
        return None
    spread_pct = (best_ask - best_bid) / midpoint * 100.0
    lines = [
        f"== SỔ LỆNH CÔNG KHAI {snapshot.get('market', '')} (snapshot) ==",
        f"sampled_at_vn={format_vn_datetime(snapshot.get('sampled_at'))} | levels_per_side≤{snapshot.get('requested_levels')} | mid={fmt(midpoint)} {BINANCE_QUOTE_ASSET} | spread={spread_pct:.4f}%",
    ]
    if ticker_price and ticker_price > 0:
        lines.append(f"ticker_vs_mid={(ticker_price - midpoint) / midpoint * 100:+.4f}%")

    def top_visible_zones(levels: list[tuple[float, float]], side: str) -> list[tuple[float, float, float, float, float]]:
        buckets: dict[int, dict] = {}
        for price, qty in levels:
            distance_pct = ((midpoint - price) if side == "bid" else (price - midpoint)) / midpoint * 100.0
            if distance_pct < 0 or distance_pct > ORDERBOOK_BANDS_PCT[-1]:
                continue
            bucket_id = int(math.floor((distance_pct + 1e-12) / ORDERBOOK_ZONE_BIN_PCT))
            bucket = buckets.setdefault(bucket_id, {"notional": 0.0, "prices": []})
            bucket["notional"] += price * qty
            bucket["prices"].append(price)
        result = []
        for bucket_id, bucket in buckets.items():
            result.append((
                bucket["notional"], bucket_id * ORDERBOOK_ZONE_BIN_PCT,
                (bucket_id + 1) * ORDERBOOK_ZONE_BIN_PCT,
                min(bucket["prices"]), max(bucket["prices"]),
            ))
        return sorted(result, reverse=True)[:3]

    for side, levels in (("bid", bids), ("ask", asks)):
        zones = top_visible_zones(levels, side)
        if zones:
            parts = [
                f"dist={lo:.2f}–{hi:.2f}% price={pmin:g}–{pmax:g} notional={notional:.0f} {BINANCE_QUOTE_ASSET}"
                for notional, lo, hi, pmin, pmax in zones
            ]
            lines.append(f"top_visible_{side}_bands (0–{ORDERBOOK_BANDS_PCT[-1]:.2f}%, bin {ORDERBOOK_ZONE_BIN_PCT:.2f}%): " + " | ".join(parts))
    for band_pct in ORDERBOOK_BANDS_PCT:
        band = band_pct / 100.0
        bid_levels = [(price, qty) for price, qty in bids if midpoint * (1.0 - band) <= price <= midpoint]
        ask_levels = [(price, qty) for price, qty in asks if midpoint <= price <= midpoint * (1.0 + band)]
        bid_notional = sum(price * qty for price, qty in bid_levels)
        ask_notional = sum(price * qty for price, qty in ask_levels)
        total = bid_notional + ask_notional
        imbalance_pct = ((bid_notional - ask_notional) / total * 100.0) if total > 0 else None
        imbalance_text = f"{imbalance_pct:+.1f}%" if imbalance_pct is not None else "N/A"
        lines.append(
            f"depth_±{band_pct:.2f}%_{BINANCE_QUOTE_ASSET}: bid={bid_notional:.0f} ({len(bid_levels)} levels) "
            f"ask={ask_notional:.0f} ({len(ask_levels)} levels) bid_imbalance={imbalance_text}"
        )
    lines.append(
        "Top bands là nơi tập trung notional lệnh giới hạn hiển thị trong từng dải giá; các lệnh có thể bị sửa/hủy. "
        "Đây không phải vị thế, stop-loss hay bản đồ thanh lý. Độ sâu chỉ bao gồm các mức trong snapshot được trả về."
    )
    return "\n".join(lines)


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
    market: str = "futures",
) -> pd.DataFrame | None:
    r = _binance_get_with_retry(
        BINANCE_SPOT_KLINES_URL if market == "spot" else BINANCE_FUTURES_KLINES_URL,
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
    candle_label = "5M" if pred.get("mode") == "futures" else "1H"
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
    Users who want to see results actively use /history.
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
            result_interval = get_result_check_interval(pred.get("mode", "futures"))
            fetch_start = start_dt - _interval_to_timedelta(result_interval)
            market = "spot" if pred.get("mode") == "spot" else "futures"
            current_price = await asyncio.to_thread(get_current_price_raw, pred["symbol"], market) if (pred.get("result") or pred.get("entry_status")) == "PENDING_ENTRY" else None
            candles = await asyncio.to_thread(get_binance_klines_since, pred["symbol"], result_interval, fetch_start, 1000, market)
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
                    price = await asyncio.to_thread(get_current_price_raw, pred["symbol"], market)
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
    }


# ─── History helpers ─────────────────────────────────────────────────────────

def format_history(limit: int = 10, user_id: int | None = None) -> str:
    """10 lệnh gần nhất của ĐÚNG user_id, mới nhất là #1.

    Không filter theo symbol: user phân tích BTC/ETH/XRP thì hiện hết trong 1 danh sách.
    user_id=None không có scope → không trả về dữ liệu (trước đây nó từng hiện lệnh của
    MỌI user kèm User ID, nay /historyall đã gỡ nên không còn lý do gì để lộ dữ liệu chung).
    """
    if user_id is None:
        return "Chưa có lịch sử dự đoán."
    init_prediction_db()
    limit = max(1, min(10, int(limit or 10)))
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT symbol, mode, direction, entry_low, entry_high, sl, tp1, tp2,
                   result, result_price, created_at
            FROM predictions
            WHERE user_id=? AND result NOT IN ('REJECTED_PLAN', 'NO_TRADE')
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    if not rows:
        return "Chưa có lịch sử dự đoán."

    # /history shows a stable index number over a rolling window of the 10 most recent trades:
    # newest first (#1) down to the oldest (#10). When an 11th trade is saved, the oldest one is
    # pruned and the list stays #1..#10.
    # The DB id stays the same inside the database, but it isn't used as the display number for the user.
    lines = [f"🧾 {limit} lệnh đã trade theo bot gần nhất của bạn"]
    for display_idx, row in enumerate(rows, 1):
        sym, mode, direction, entry_low, entry_high, sl, tp1, tp2, result, result_price, created_at = row
        mode_label = "FUTURES" if mode == "futures" else "SPOT"
        created_label = format_vn_datetime(created_at) if created_at else "không rõ"
        lines.append(
            f"#{display_idx} {sym} {mode_label} {direction} → {result}\n"
            f"Thời gian phân tích: {created_label}\n"
            f"Entry {fmt(entry_low)}–{fmt(entry_high)} | SL {fmt(sl)} | TP1 {fmt(tp1)} | TP2 {fmt(tp2)}"
            + (f" | Giá check {fmt(result_price)}" if result_price else "")
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

def get_binance_klines(symbol: str, interval: str, limit: int, market: str = "futures") -> pd.DataFrame | None:
    r = _binance_get_with_retry(
        BINANCE_SPOT_KLINES_URL if market == "spot" else BINANCE_FUTURES_KLINES_URL,
        {"symbol": symbol, "interval": interval, "limit": limit}, timeout=20
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
    r["adx_14"] = calculate_adx(r, 14)
    # Baseline excludes the current candle so a real spike isn't diluted by itself.
    r["vol_ma20"]  = r["volume"].shift(1).rolling(20).mean()
    r["vol_ratio"] = r["volume"] / r["vol_ma20"]
    # A fully flat candle run (zero true range) or a zero-volume baseline divides to inf, which
    # dropna() alone doesn't catch — turn those into NaN too so they're dropped like any other
    # incomplete row instead of leaking a literal "inf" into the LLM prompt.
    r = r.replace([np.inf, -np.inf], np.nan)
    return r.dropna().reset_index(drop=True)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def calculate_daily_vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].astype(float)
    try:
        days = pd.to_datetime(df["timestamp"], utc=True).dt.date
    except Exception:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    vwap = pd.Series(np.nan, index=df.index, dtype="float64")
    for _, idx in df.groupby(days).groups.items():
        group_idx = df.index[idx] if not isinstance(idx, list) else idx
        cum_pv = (tp.loc[group_idx] * vol.loc[group_idx]).cumsum()
        cum_v = vol.loc[group_idx].cumsum().replace(0, np.nan)
        vwap.loc[group_idx] = cum_pv / cum_v
    return vwap


def add_indicators_futures(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    r = df.copy()
    r["ema_20"] = calculate_ema(r["close"], 20)
    r["ema_50"] = calculate_ema(r["close"], 50)
    r["ema_200"] = calculate_ema(r["close"], 200)
    r["rsi_14"] = calculate_rsi(r["close"], 14)
    r["adx_14"] = calculate_adx(r, 14)
    r["atr_14"] = calculate_atr(r, 14)
    r["vol_ma20"] = r["volume"].shift(1).rolling(20).mean()
    r["vol_ratio"] = r["volume"] / r["vol_ma20"]
    r["rng"] = (r["high"] - r["low"]) / r["atr_14"]
    hl_range = (r["high"] - r["low"]).replace(0, np.nan)
    r["cl_pct"] = (r["close"] - r["low"]) / hl_range * 100.0
    r["cl_pct"] = r["cl_pct"].fillna(50.0)
    r["vwap"] = calculate_daily_vwap(r)
    r = r.replace([np.inf, -np.inf], np.nan)
    required = ["ema_20", "ema_50", "rsi_14", "atr_14", "vol_ratio", "rng", "cl_pct"]
    return r.dropna(subset=required).reset_index(drop=True)


# ─── Feature engineering: Structure / Fibonacci / Liquidity ────────────

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


def build_market_snapshot(
    timeframe_data: dict[str, pd.DataFrame | None],
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


def get_current_price_str(symbol: str, market: str = "futures") -> tuple[str, float | None]:
    price = get_current_price_raw(symbol, market=market)
    if price is None:
        return "Giá hiện tại: không có dữ liệu", None
    return f"Giá hiện tại: {fmt(price)} {BINANCE_QUOTE_ASSET}", price


# ─── Select current AI provider/API key/model (multi-provider config) ───


def _truncate_text(text: str | None, limit: int = 600) -> str | None:
    if not text:
        return None
    text = str(text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."



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
    response_format: dict | None = None,
) -> dict:
    ensure_ai_config()
    effective_reasoning_effort = (
        reasoning_effort if reasoning_effort is not None else PLANNER_REASONING_EFFORT
    )
    return _openrouter_create_once(
        system, messages, model=PLANNER_MODEL, max_tokens=max_tokens, timeout=timeout,
        reasoning_effort=effective_reasoning_effort, response_format=response_format,
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
    response_format: dict | None = None,
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
                    response_format=response_format,
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
    m = re.search(r"QUYẾT ĐỊNH[:\s]+[*_]*(LONG|SHORT|BUY|MUA|NO[_\s-]?TRADE|KHÔNG\s+VÀO\s+LỆNH|KHONG\s+VAO\s+LENH)", output, re.IGNORECASE)
    if m:
        raw_direction = m.group(1).upper().replace("-", "_").replace(" ", "_")
        direction = "NO_TRADE" if raw_direction in ("NO_TRADE", "NO__TRADE", "KHÔNG_VÀO_LỆNH", "KHONG_VAO_LENH") else ("LONG" if raw_direction in ("BUY", "MUA") else raw_direction)
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


def sanitize_user_output(output: str) -> str:
    """Clean up confusing wording and internal technical labels before sending to the user / saving full_response."""
    replacements = {
        "swing gần": "đỉnh/đáy gần",
        "Spot gần": "Đỉnh/đáy gần",
        "swing lớn": "biên lớn",
        "Spot lớn": "Biên lớn",
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

    return text


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


def load_system_prompt(mode: str = "spot") -> str:
    if mode == "futures":
        text = _load_prompt_file("analyze_system_prompt_futures.txt")
        return (
            text.replace("{MIN_RR}", f"{MIN_RR}")
            .replace("{SL_ATR_MIN}", f"{SL_ATR_MIN}")
            .replace("{SL_ATR_MAX}", f"{SL_ATR_MAX}")
            .replace("{ENTRY_READY_ATR15}", f"{ENTRY_READY_ATR15:g}")
            .replace("{CITE_TOL_PCT}", f"{CITE_REL_TOL * 100:g}")
            .replace("{MAX_SL_PCT}", f"{MAX_SL_PCT:g}")
        )
    return _load_prompt_file("analyze_system_prompt_spot.txt")


def load_timeframe_data(binance_symbol: str, interval: str, limit: int, market: str = "spot") -> pd.DataFrame | None:
    """Sync helper: fetch Binance candles then calculate indicators."""
    return add_indicators(get_binance_klines(binance_symbol, interval, limit, market=market))


def request_json_analysis(system_prompt: str, user_prompt: str) -> str:
    max_tokens = max(800, min(PLANNER_MAX_OUTPUT_TOKENS, PLANNER_OUTPUT_TOKEN_CAP))
    try:
        return create_with_continuation(
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            timeout=PLANNER_TIMEOUT_SECONDS,
            allow_continuation=False,
            call_type="main_json",
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        if "400" in str(exc) and "response_format" in str(exc).lower():
            return create_with_continuation(
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=max_tokens,
                timeout=PLANNER_TIMEOUT_SECONDS,
                allow_continuation=False,
                call_type="main_json",
            )
        raise


# ─── Objective market packet ──────────────────────────────────────────────

def _mode_frame_roles(mode: str) -> tuple[str, str, str]:
    """Return frame labels in analysis order: trigger, structure, higher-timeframe context."""
    if mode == "futures":
        return "15m", "1H", "4H"
    return "4H", "1D", "1W"


def load_timeframe_data_futures(
    binance_symbol: str, interval: str, limit: int, market: str = "futures"
) -> pd.DataFrame | None:
    return add_indicators_futures(get_binance_klines(binance_symbol, interval, limit, market=market))


def _missing_critical_timeframes(timeframe_data: dict, mode: str) -> list[str]:
    """No timeframe is presumed less important than another, so all three are required to have been
    fetched at all — if any is missing, the model must not be trusted to notice a "không có dữ liệu"
    text line and quietly work around it; Python forces NO_TRADE instead of letting a partial
    packet reach the planner.

    This is about a real fetch failure only (network/API error -> load_timeframe_data returns None).
    It is deliberately NOT triggered by an empty-but-not-None DataFrame: that shape means the fetch
    itself succeeded but this coin doesn't have enough closed history yet for any indicator to
    produce a real value at this interval (e.g. weekly candles for a newly listed coin, or 4H for a coin
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
    # deliberately much smaller than the FETCH window (see FUTURES_TIMEFRAMES/SPOT_TIMEFRAMES)
    # that add_indicators uses for EMA/RSI/MACD/ADX warm-up: showing hundreds of raw rows doesn't
    # help the model (long, repetitive numeric tables are unreliable to read in full — the earlier
    # single-snapshot + trailing indicator series already carry the "how has this been trending"
    # signal), it just adds noise and cost. Fetching stays wide for indicator accuracy either way.
    # FUTURES: 30/36/32 rows. SPOT: 30 4H candles (~5 days), 48 daily (~7 weeks),
    # and 64 weekly (~15 months). If a coin doesn't have this many closed candles yet, the
    # caller (_v50_raw_candles' .tail()) just returns however many actually exist — this is an
    # upper bound, never a forced/padded count.
    limits = {
        "futures": {"4H": 30, "1H": 36, "15m": 32},
        "spot": SPOT_DISPLAY,
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
    full_cols_n = min(len(rows), SPOT_FULL_COLS_N) if mode == "spot" else len(rows)
    full_start = max(0, len(rows) - full_cols_n)
    out = [f"{label} — {len(rows)} nến đã đóng gần nhất (nến cũ chỉ OHLC; {full_cols_n} nến mới nhất có thêm vol_ratio,takerBuy%,CVD):"]
    cvd = 0.0
    for row_idx, (_, row) in enumerate(rows.iterrows()):
        if row_idx < full_start:
            out.append(
                f"{_v50_time_value(row)} | {fmt(_safe_float(row.get('open')))} | "
                f"{fmt(_safe_float(row.get('high')))} | {fmt(_safe_float(row.get('low')))} | "
                f"{fmt(_safe_float(row.get('close')))}"
            )
            continue
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

    All three timeframes receive the same indicator treatment. For SPOT they are presented
    context-first (1W → 1D → 4H); the system prompt tells the model how to use those roles, while
    Python still avoids deciding trend or entry. Only standard formula-defined indicators appear;
    no tunable Python pattern detector is added.
    """
    labels = list(_mode_frame_roles(mode))
    if mode == "spot":
        labels.reverse()  # Put SPOT context first: 1W → 1D → 4H, matching the prompt's workflow.
    lines = [
        "OBJECTIVE_MARKET_PACKET",
        "Múi giờ của mọi timestamp trong packet: giờ Việt Nam (UTC+7), hậu tố VN.",
        f"Giá hiện tại: {fmt(current_price)}",
        "Python chỉ chuẩn bị dữ kiện khách quan; không kết luận hướng và không dựng Entry/SL/TP.",
        "Packet không có Fibonacci, market-regime label hay trend label.",
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
            f"adx14={fmt(_safe_float(row.get('adx_14')),1)}, "
            f"takerBuy={fmt(_taker_buy_ratio(row),1)}%."
        )
        indicator_series = _v50_indicator_series(label, df)
        if indicator_series:
            lines.append(indicator_series)
        # Daily Futures background and weekly Spot context are intentionally excluded from
        # Ichimoku: its shifted cloud is less useful on these sparse context views.
        skip_ichimoku = (mode == "futures" and label == "1D") or (mode == "spot" and label == "1W")
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
    trigger, trend, big = _mode_frame_roles(mode)
    lines = [
        f"Mode={'FUTURES' if mode == 'futures' else 'SPOT'}; price={fmt(current_price)}",
    ]
    # timing 12, trend 16, macro 6. SPOT uses the same allocation, mapped to the corresponding roles.
    recent_counts = {trigger: 12, trend: 16, big: 6}
    for label in (trigger, trend, big):
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
            f"adx14={fmt(_safe_float(row.get('adx_14')),1)},"
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
    lines = ["SYNCHRONIZED_DECISION_SNAPSHOT", "Mọi timestamp bên dưới dùng giờ Việt Nam (UTC+7), hậu tố VN."]
    decision_labels = _mode_frame_roles(mode)
    if mode == "spot":
        decision_labels = tuple(reversed(decision_labels))
    lines += [
        line for label in decision_labels
        if (line := _v50_live_line(label, timeframe_data.get(label)))
    ]
    return "\n".join(lines)


# ─── Futures packet (mode "futures": 4H / 1H / 15m) ────────────────────────────

FUTURES_FRAME_ORDER = ("4H", "1H", "15m")
FUTURES_FRAME_EMA = {"4H": "ema_50", "1H": "ema_20", "15m": "ema_20"}


def _futures_time_label(row) -> str:
    ts = _v50_timestamp_value(row)
    if ts is None or pd.isna(ts):
        return "-- --:--"
    try:
        return ts.tz_convert("Asia/Ho_Chi_Minh").strftime("%m-%d %H:%M")
    except Exception:
        return "-- --:--"


def _futures_dist_text(level: float | None, price: float | None, atr_ref: float | None) -> str:
    if level is None or price is None or not price:
        return ""
    pct = (level - price) / abs(price) * 100.0
    if atr_ref:
        return f"[{pct:+.2f}% | {(level - price) / atr_ref:+.1f}atr]"
    return f"[{pct:+.2f}%]"


def _futures_candle_block(
    label: str,
    df: pd.DataFrame | None,
    display_n: int,
    price: float | None,
    atr_ref: float | None,
    facts: dict,
    show_vwap: bool = False,
) -> str:
    closed = _v50_closed_df(df)
    if closed is None or closed.empty:
        return ""
    rows = closed.tail(display_n)
    frame_key = label.lower()
    ema_col = FUTURES_FRAME_EMA[label]
    total = len(rows)
    full_n = min(max(1, FUTURES_FULL_COLS_N), total)
    # Hai đoạn: đoạn cũ (cột rút gọn) rồi đoạn gần nhất (đủ cột); nếu tổng ≤ N thì chỉ một đoạn đủ cột.
    has_old_segment = total > full_n
    old_count = total - full_n if has_old_segment else 0
    new_count = full_n if has_old_segment else total

    def _base_parts(tag: str, row) -> list[str]:
        parts = [
            tag,
            _futures_time_label(row),
            f"{fmt(_safe_float(row.get('open')))} {fmt(_safe_float(row.get('high')))} "
            f"{fmt(_safe_float(row.get('low')))} {fmt(_safe_float(row.get('close')))}",
            fmt(_safe_float(row.get(ema_col))),
        ]
        if show_vwap:
            parts.append(fmt(_safe_float(row.get("vwap"))))
        return parts

    def _register(row, tag: str, full_cols: bool) -> None:
        facts[f"o_{frame_key}_{tag}"] = _safe_float(row.get("open"))
        facts[f"h_{frame_key}_{tag}"] = _safe_float(row.get("high"))
        facts[f"l_{frame_key}_{tag}"] = _safe_float(row.get("low"))
        facts[f"c_{frame_key}_{tag}"] = _safe_float(row.get("close"))
        if not full_cols:
            return
        vr = _safe_float(row.get("vol_ratio"))
        tb = _taker_buy_ratio(row)
        rng = _safe_float(row.get("rng"))
        cl = _safe_float(row.get("cl_pct"))
        if vr is not None:
            facts[f"vr_{frame_key}_{tag}"] = vr
        if tb is not None:
            facts[f"tb_{frame_key}_{tag}"] = tb
        if rng is not None:
            facts[f"rng_{frame_key}_{tag}"] = rng
        if cl is not None:
            facts[f"cl_{frame_key}_{tag}"] = cl

    out: list[str] = []
    if has_old_segment:
        vwap_col = " | vwap" if show_vwap else ""
        out.append(f"== {label}: đoạn cũ, {old_count} nến == cột: n | thời gian | O H L C | ema{vwap_col}")
        for i, (_, row) in enumerate(rows.iloc[:old_count].iterrows()):
            k = total - 1 - i
            tag = "t0" if k == 0 else f"t-{k}"
            out.append(" | ".join(_base_parts(tag, row)))
            _register(row, tag, full_cols=False)
    vwap_col = " | vwap" if show_vwap else ""
    out.append(
        f"== {label}: đoạn gần nhất, {new_count} nến (đủ cột) == cột: n | thời gian | O H L C | ema{vwap_col}"
        " | vr | tb% | rng | cl%"
    )
    start = old_count
    for i, (_, row) in enumerate(rows.iloc[start:].iterrows(), start=start):
        k = total - 1 - i
        tag = "t0" if k == 0 else f"t-{k}"
        parts = _base_parts(tag, row)
        vr = _safe_float(row.get("vol_ratio"))
        tb = _taker_buy_ratio(row)
        rng = _safe_float(row.get("rng"))
        cl = _safe_float(row.get("cl_pct"))
        parts += [
            f"{fmt(vr, 2)}x" if vr is not None else "N/A",
            f"{fmt(tb, 1)}%" if tb is not None else "N/A",
            fmt(rng, 2) if rng is not None else "N/A",
            f"{fmt(cl, 0)}" if cl is not None else "N/A",
        ]
        out.append(" | ".join(parts))
        _register(row, tag, full_cols=True)
    return "\n".join(out)


def _futures_singles_block(label: str, df: pd.DataFrame | None, price: float | None, atr_ref: float | None, facts: dict) -> str:
    row = _analysis_row(df)
    if row is None:
        return ""
    frame_key = label.lower()
    bits = []
    if label == "4H":
        for key, col in (("ema50_4h", "ema_50"), ("ema200_4h", "ema_200"), ("atr14_4h", "atr_14"), ("adx14_4h", "adx_14")):
            v = _safe_float(row.get(col))
            if v is None:
                continue
            facts[key] = v
            if col == "atr_14":
                bits.append(f"atr14_4h={fmt(v)}")
            elif col == "adx_14":
                bits.append(f"adx14_4h={fmt(v, 1)}")
            else:
                bits.append(f"{key}={fmt(v)} {_futures_dist_text(v, price, atr_ref)}")
    elif label == "1H":
        for key, col, dec in (("ema20_1h", "ema_20", None), ("ema50_1h", "ema_50", None), ("atr14_1h", "atr_14", None), ("rsi14_1h", "rsi_14", 1)):
            v = _safe_float(row.get(col))
            if v is None:
                continue
            facts[key] = v
            if col == "atr_14":
                bits.append(f"atr14_1h={fmt(v)}")
            elif col == "rsi_14":
                bits.append(f"rsi14_1h={fmt(v, 1)}")
            else:
                bits.append(f"{key}={fmt(v)} {_futures_dist_text(v, price, atr_ref)}")
    else:
        for key, col, dec in (("ema20_15m", "ema_20", None), ("ema50_15m", "ema_50", None), ("atr14_15m", "atr_14", None), ("rsi14_15m", "rsi_14", 1), ("vwap_15m", "vwap", None)):
            v = _safe_float(row.get(col))
            if v is None:
                continue
            facts[key] = v
            if col == "atr_14":
                bits.append(f"atr14_15m={fmt(v)}")
            elif col == "rsi_14":
                bits.append(f"rsi14_15m={fmt(v, 1)}")
            else:
                bits.append(f"{key}={fmt(v)} {_futures_dist_text(v, price, atr_ref)}")
    _ = frame_key
    return " | ".join(bits)


def _futures_hh_ll(df: pd.DataFrame | None, n: int) -> tuple[float | None, float | None]:
    closed = _v50_closed_df(df)
    if closed is None or closed.empty:
        return None, None
    window = closed.tail(max(1, n))
    try:
        return float(window["high"].max()), float(window["low"].min())
    except Exception:
        return None, None


def _fetch_daily_weekly_levels(symbol: str) -> dict:
    ref: dict = {}
    try:
        daily = get_binance_klines(symbol, "1d", 10)
        if daily is not None and len(daily) >= 2:
            prev = daily.iloc[-2]
            live = daily.iloc[-1]
            ref["prev_day_high"] = float(prev["high"])
            ref["prev_day_low"] = float(prev["low"])
            ref["prev_day_close"] = float(prev["close"])
            ref["today_open"] = float(live["open"])
            ref["today_high"] = float(live["high"])
            ref["today_low"] = float(live["low"])
    except Exception:
        pass
    try:
        weekly = get_binance_klines(symbol, "1w", 3)
        if weekly is not None and len(weekly) >= 2:
            prev_w = weekly.iloc[-2]
            ref["prev_week_high"] = float(prev_w["high"])
            ref["prev_week_low"] = float(prev_w["low"])
    except Exception:
        pass
    return ref


def get_funding_rate_history(symbol: str, limit: int = 4) -> dict | None:
    r = _binance_get_with_retry(
        f"{BINANCE_FUTURES_API_BASE}/fapi/v1/fundingRate",
        {"symbol": symbol, "limit": max(1, limit)},
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


def _futures_oi_price_changes(symbol: str) -> dict:
    out: dict = {}
    try:
        r = _binance_get_with_retry(
            f"{BINANCE_FUTURES_API_BASE}/futures/data/openInterestHist",
            {"symbol": symbol, "period": "1h", "limit": 25},
            max_retries=1, timeout=10,
        )
        if r is not None:
            data = r.json()
            if isinstance(data, list) and len(data) >= 2:
                vals = [float(x["sumOpenInterest"]) for x in data]
                last = vals[-1]
                for name, back in (("oi_chg_1h", 1), ("oi_chg_4h", 4), ("oi_chg_24h", 24)):
                    if len(vals) > back and vals[-1 - back]:
                        out[name] = (last - vals[-1 - back]) / abs(vals[-1 - back]) * 100.0
    except Exception:
        pass
    try:
        kl = get_binance_klines(symbol, "1h", 26)
        if kl is not None and len(kl) >= 2:
            closes = [float(v) for v in kl["close"].tolist()]
            last = closes[-1]
            for name, back in (("price_chg_1h", 1), ("price_chg_4h", 4), ("price_chg_24h", 24)):
                if len(closes) > back and closes[-1 - back]:
                    out[name] = (last - closes[-1 - back]) / abs(closes[-1 - back]) * 100.0
    except Exception:
        pass
    return out


def _futures_taker_windows(df_1h: pd.DataFrame | None) -> dict:
    out: dict = {}
    try:
        closed = _v50_closed_df(df_1h)
        if closed is None or closed.empty:
            return out
        for name, n in (("taker_buy_pct_1h", 1), ("taker_buy_pct_4h", 4)):
            window = closed.tail(n)
            vol = pd.to_numeric(window["volume"], errors="coerce").sum()
            tbv = pd.to_numeric(window["taker_buy_volume"], errors="coerce").sum()
            if vol and vol > 0 and pd.notna(tbv):
                out[name] = float(tbv) / float(vol) * 100.0
    except Exception:
        pass
    return out


def _fetch_futures_derivs(symbol: str) -> dict:
    out: dict = {}
    funding = get_funding_rate_history(symbol, 4)
    if funding:
        out["funding"] = {"latest_pct": funding["latest_pct"], "history_pct": funding["history_pct"]}
        out["funding_hist"] = list(funding["history_pct"])
    out.update(_futures_oi_price_changes(symbol))
    try:
        ls = get_long_short_ratio_context(symbol)
        if ls:
            out["long_short_top"] = float(ls["top_ratio"])
            out["long_short_crowd"] = float(ls["global_ratio"])
    except Exception:
        pass
    return out


def build_futures_packet(
    timeframe_data: dict[str, pd.DataFrame | None],
    ref_levels: dict | None,
    derivs: dict | None,
    current_price: float | None,
    symbol: str = "",
) -> tuple[str, dict]:
    facts: dict = {}
    created = utc_now().astimezone(VN_TZ).strftime("%Y-%m-%d %H:%M")
    lines = [
        "OBJECTIVE_MARKET_PACKET",
        f"Symbol {symbol} | FUTURES | tạo lúc {created} | giờ VN (UTC+7)",
        f"Giá hiện tại: price={fmt(current_price)}",
        "Python chỉ chuẩn bị dữ kiện khách quan; không kết luận hướng và không dựng Entry/SL/TP.",
    ]
    df_1h = timeframe_data.get("1H")
    atr_ref = None
    try:
        row_1h = _analysis_row(df_1h)
        atr_ref = _safe_float(row_1h.get("atr_14")) if row_1h is not None else None
    except Exception:
        atr_ref = None
    if atr_ref:
        facts["atr14_1h"] = atr_ref
        lines.append("Khoảng cách trong [ ]: (mức − giá)/giá theo %, và theo ATR 1H (atr14_1h). Dấu + = mức nằm trên giá.")
    else:
        lines.append("Khoảng cách trong [ ]: (mức − giá)/giá theo %. Dấu + = mức nằm trên giá.")
    for label in FUTURES_FRAME_ORDER:
        df = timeframe_data.get(label)
        block = _futures_candle_block(label, df, FUTURES_DISPLAY.get(label, 30), current_price, atr_ref, facts, show_vwap=(label == "15m"))
        if not block:
            continue
        lines += ["", block]
        singles = _futures_singles_block(label, df, current_price, atr_ref, facts)
        if singles:
            lines.append(singles)
        live = _v50_live_line(label, df)
        if live:
            lines.append(live)
    ref_lines = []
    for key in ("prev_day_high", "prev_day_low", "prev_day_close", "today_open", "today_high", "today_low", "prev_week_high", "prev_week_low"):
        v = (ref_levels or {}).get(key)
        if v is None:
            continue
        facts[key] = float(v)
        ref_lines.append(f"{key}={fmt(float(v))} {_futures_dist_text(float(v), current_price, atr_ref)}")
    for key, df, n in (
        ("hh_15m_12", timeframe_data.get("15m"), 12), ("ll_15m_12", timeframe_data.get("15m"), 12),
        ("hh_15m_24", timeframe_data.get("15m"), 24), ("ll_15m_24", timeframe_data.get("15m"), 24),
        ("hh_15m_48", timeframe_data.get("15m"), 48), ("ll_15m_48", timeframe_data.get("15m"), 48),
        ("hh_1h_24", timeframe_data.get("1H"), 24), ("ll_1h_24", timeframe_data.get("1H"), 24),
        ("hh_1h_48", timeframe_data.get("1H"), 48), ("ll_1h_48", timeframe_data.get("1H"), 48),
    ):
        hi, lo = _futures_hh_ll(df, n)
        v = hi if key.startswith("hh_") else lo
        if v is None:
            continue
        facts[key] = float(v)
        ref_lines.append(f"{key}={fmt(float(v))} {_futures_dist_text(float(v), current_price, atr_ref)}")
    if ref_lines:
        lines += ["", "== MỨC THAM CHIẾU == key=giá [%, atr]", " | ".join(ref_lines)]
    deriv_lines = []
    funding_hist = (derivs or {}).get("funding_hist") or []
    if funding_hist:
        facts["funding_last"] = float(funding_hist[-1])
        for i, v in enumerate(funding_hist, 1):
            facts[f"funding_{i}"] = float(v)
        bits = [f"funding_{i}={float(v):+.4f}" for i, v in enumerate(funding_hist, 1)]
        deriv_lines.append(" | ".join(bits) + f" (cũ→mới; funding_last = funding_{len(funding_hist)})")
    for key in ("oi_chg_1h", "oi_chg_4h", "oi_chg_24h", "price_chg_1h", "price_chg_4h", "price_chg_24h",
                "long_short_top", "long_short_crowd", "taker_buy_pct_1h", "taker_buy_pct_4h"):
        v = (derivs or {}).get(key)
        if v is None:
            continue
        facts[key] = float(v)
    oi_bits = [f"{k}={facts[k]:+.2f}" for k in ("oi_chg_1h", "oi_chg_4h", "oi_chg_24h") if k in facts]
    if oi_bits:
        deriv_lines.append(" | ".join(oi_bits))
    price_bits = [f"{k}={facts[k]:+.2f}" for k in ("price_chg_1h", "price_chg_4h", "price_chg_24h") if k in facts]
    if price_bits:
        deriv_lines.append(" | ".join(price_bits))
    ls_bits = [f"{k}={facts[k]:.2f}" for k in ("long_short_top", "long_short_crowd") if k in facts]
    if ls_bits:
        deriv_lines.append(" | ".join(ls_bits))
    tb_bits = [f"{k}={facts[k]:.1f}%" for k in ("taker_buy_pct_1h", "taker_buy_pct_4h") if k in facts]
    if tb_bits:
        deriv_lines.append(" | ".join(tb_bits))
    if deriv_lines:
        lines += ["", "== PHÁI SINH == (oi/price/funding tính theo %)", *deriv_lines]
    return "\n".join(lines), facts



def build_user_prompt(
    symbol: str,
    mode: str,
    timeframe_data: dict[str, pd.DataFrame | None],
    current_price_str: str,
    feature_block: str | None = None,
    decision_snapshot: str | None = None,
    market_context_block: str | None = None,
) -> str:
    """Data-first planner prompt; analytical rules live only in system prompt."""
    mode_label = "FUTURES" if mode == "futures" else "SPOT"
    raw_labels = _mode_frame_roles(mode)
    if mode == "spot":
        raw_labels = tuple(reversed(raw_labels))
    raw_sections = [
        section for label in raw_labels
        if (section := _v50_raw_candles(label, timeframe_data.get(label), mode))
    ]
    spot = mode == "spot"
    if spot:
        return "\n".join([
            f"PHÂN TÍCH {symbol} — SPOT",
            f"Thời điểm tạo packet: {utc_now().astimezone(VN_TZ).strftime('%Y-%m-%d %H:%M:%S VN')}",
            current_price_str,
            "Dữ liệu thị trường của symbol này lấy từ Binance Spot: ticker, nến và snapshot sổ lệnh nếu có. Mọi chỉ báo và phép tính đều dựa trên dữ liệu Spot của symbol này.",
            market_context_block or "",
            "RAW OHLCV:",
            "\n\n".join(raw_sections),
            feature_block or "OBJECTIVE_MARKET_PACKET: N/A",
            decision_snapshot or "LIVE SNAPSHOT: N/A",
            "Thực hiện thứ tự phân tích trong system prompt. Trả về đúng một JSON hợp lệ theo schema Spot ở system prompt, không thêm chữ hoặc markdown.",
        ])
    decision_example = "BUY / NO TRADE" if spot else "LONG / SHORT / NO TRADE"
    side_example = "BUY" if spot else "LONG/SHORT"
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
        f"🏆 QUYẾT ĐỊNH: [CHỌN MỘT: {decision_example}]",
        f"Giá hiện tại: ... {BINANCE_QUOTE_ASSET}",
        "Nếu NO TRADE:",
        "Lý do: (1–2 câu ngắn gọn nêu đúng lý do bạn không vào lệnh)",
        f"Nếu {side_example}:",
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


def build_futures_user_prompt(
    symbol: str,
    current_price_str: str,
    feature_block: str | None = None,
) -> str:
    return "\n".join([
        f"PHÂN TÍCH {symbol} — FUTURES",
        f"Thời điểm tạo packet: {utc_now().astimezone(VN_TZ).strftime('%Y-%m-%d %H:%M:%S VN')}",
        current_price_str,
        "Packet bên dưới đã chứa toàn bộ dữ liệu cần thiết (nến 4H/1H/15m, chỉ báo, mức tham chiếu, phái sinh).",
        "",
        feature_block or "OBJECTIVE_MARKET_PACKET: N/A",
        "",
        "Trả về đúng MỘT đối tượng JSON theo system prompt, không thêm chữ ngoài JSON.",
    ])


def render_plan_text(plan: dict, symbol: str, mode_label: str, current_price: float | None) -> str:
    plan = plan or {}
    quyet_dinh = str(plan.get("quyet_dinh") or "NO_TRADE").upper()
    lines = [
        f"🎯 {symbol} — {mode_label}",
        f"🏆 QUYẾT ĐỊNH: {quyet_dinh if quyet_dinh != 'NO_TRADE' else 'NO TRADE'}",
        f"Giá hiện tại: {fmt(current_price)} {BINANCE_QUOTE_ASSET}" if current_price is not None else "Giá hiện tại: N/A",
    ]
    if quyet_dinh == "NO_TRADE":
        lines.append(f"Lý do: {plan.get('ly_do') or 'Không có setup đủ tốt.'}")
        return "\n".join(lines)
    lines += [
        f"Entry: {fmt(plan.get('entry_thap'))}–{fmt(plan.get('entry_cao'))}",
        f"SL: {fmt(plan.get('sl'))}",
        f"TP1: {fmt(plan.get('tp1'))}",
        f"TP2: {fmt(plan.get('tp2')) if plan.get('tp2') is not None else 'N/A'}",
        f"Kích hoạt: {plan.get('kich_hoat') or '-'}",
    ]
    bang = plan.get("bang_chung") or {}
    for key, label in (("entry", "Bằng chứng Entry"), ("sl", "Bằng chứng SL"), ("tp1", "Bằng chứng TP1"), ("tp2", "Bằng chứng TP2")):
        val = bang.get(key) if isinstance(bang, dict) else None
        if key == "tp2" and (val is None or plan.get("tp2") is None):
            lines.append(f"{label}: N/A")
        else:
            lines.append(f"{label}: {val or '-'}")
    risks = plan.get("rui_ro") or []
    lines.append("⚠️ Rủi ro:")
    if isinstance(risks, list) and risks:
        for r in risks:
            lines.append(f"- {r}")
    else:
        lines.append(f"- {risks}" if risks else "- Xem bằng chứng và SL/TP ở trên.")
    return "\n".join(lines)


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
        planner_output = kwargs.get("planner_output") or ""
        public_output = kwargs.get("public_output") or planner_output
        # Ưu tiên đọc JSON (luồng futures/spot trả raw JSON); fallback sang parser văn bản cũ.
        parsed_json = _extract_json_object(public_output)
        if parsed_json is not None and parsed_json.get("quyet_dinh") is not None:
            raw = str(parsed_json["quyet_dinh"]).upper().replace(" ", "_").replace("-", "_")
            direction = {"BUY": "LONG", "MUA": "LONG", "SELL": "SHORT"}.get(raw, raw)
            entry_low = _num_or_none(parsed_json.get("entry_thap"))
            entry_high = _num_or_none(parsed_json.get("entry_cao"))
            sl = _num_or_none(parsed_json.get("sl"))
            tp1 = _num_or_none(parsed_json.get("tp1"))
            tp2 = _num_or_none(parsed_json.get("tp2"))
        else:
            parsed = parse_prediction_from_output(public_output)
            direction = (parsed.get("direction") or "").upper()
            entry_low = parsed.get("entry_low")
            entry_high = parsed.get("entry_high")
            sl = parsed.get("sl")
            tp1 = parsed.get("tp1")
            tp2 = parsed.get("tp2")
        # Hai trạng thái: LONG/SHORT = TRADE, mọi thứ khác (kể cả quyết định không đọc được) = NO_TRADE.
        if direction in ("LONG", "SHORT"):
            status, phase, final_result = "TRADE", "PLANNER_APPROVED", direction
        else:
            if direction not in ("NO_TRADE",):
                print(
                    f"[PLANNER_INVALID_DECISION] symbol={kwargs.get('symbol')} mode={kwargs.get('mode')} "
                    f"direction={direction!r} -> ghi nhận như NO_TRADE",
                    flush=True,
                )
            status, phase, final_result = "NO_TRADE", "PLANNER_NO_TRADE", "NO_TRADE"

        funding_ctx = kwargs.get("funding_context") or {}
        save_evaluation_case(
            user_id=kwargs.get("user_id"), chat_id=kwargs.get("chat_id"), source=kwargs.get("source") or "unknown",
            symbol=kwargs.get("symbol"), mode=kwargs.get("mode"), pipeline_phase=phase, final_result=final_result,
            current_price=kwargs.get("current_price"),
            planner_direction=direction, planner_status=status,
            entry_low=entry_low,
            entry_high=entry_high, sl=sl, tp1=tp1, tp2=tp2,
            market_packet=kwargs.get("planner_input"), planner_output=planner_output,
            public_output=public_output, planner_prompt_hash=prompt_hash(load_system_prompt(kwargs.get("mode") or "spot")),
            funding_rate_pct=funding_ctx.get("latest_pct"),
        )
        cleanup_evaluation_data()
    except Exception as exc:
        print(f"[SNAPSHOT_SAVE_ERROR] {exc}", flush=True)


async def collect_timeframe_data(binance_symbol: str, mode: str) -> dict[str, pd.DataFrame | None]:
    """
    Fetch multiple timeframes in parallel worker threads.

    Goal: keep requests.get() from blocking the Telegram bot's event loop, and also
    reduce wait time since FUTURES (4H/1H/15m) or SPOT (1W/1D/4H) load in parallel.
    """
    if mode == "futures":
        configs = FUTURES_TIMEFRAMES
        loader = load_timeframe_data_futures
    else:
        configs = SPOT_TIMEFRAMES
        loader = load_timeframe_data
    tasks = {
        label: asyncio.to_thread(loader, binance_symbol, interval, limit, "futures" if mode == "futures" else "spot")
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

    system_prompt, price_tuple, ref_levels, derivs_parts, orderbook_snapshot = await asyncio.gather(
        asyncio.to_thread(load_system_prompt, mode),
        asyncio.to_thread(get_current_price_str, binance_symbol, "futures" if mode == "futures" else "spot"),
        asyncio.to_thread(_fetch_daily_weekly_levels, binance_symbol) if mode == "futures" else asyncio.sleep(0, result=None),
        asyncio.to_thread(_fetch_futures_derivs, binance_symbol) if mode == "futures" else asyncio.sleep(0, result=None),
        asyncio.to_thread(fetch_orderbook_snapshot, binance_symbol, "futures" if mode == "futures" else "spot"),
    )
    orderbook_context = summarize_orderbook(orderbook_snapshot, price_tuple[1] if price_tuple else None)
    funding_ctx = (derivs_parts or {}).get("funding") if mode == "futures" else None
    if mode == "futures":
        derivs = dict(derivs_parts or {})
        derivs.update(_futures_taker_windows(timeframe_data.get("1H")))
    else:
        derivs = None
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
    if mode == "futures":
        packet_text, facts = build_futures_packet(timeframe_data, ref_levels or {}, derivs or {}, current_price, symbol=binance_symbol)
        if orderbook_context:
            packet_text += "\n\n" + orderbook_context
        facts = dict(facts)
        facts["price"] = current_price
        facts["current_price"] = current_price
        user_prompt = build_futures_user_prompt(
            symbol=binance_symbol, current_price_str=current_price_str,
            feature_block=packet_text,
        )
    else:
        feature_block = build_feature_engineering_block(timeframe_data, mode, current_price)
        decision_snapshot = build_synchronized_decision_snapshot(timeframe_data, mode, current_price)
        market_context_block = orderbook_context
        user_prompt = build_user_prompt(
            symbol=binance_symbol,
            mode=mode,
            timeframe_data=timeframe_data,
            current_price_str=current_price_str,
            feature_block=feature_block,
            decision_snapshot=decision_snapshot,
            market_context_block=market_context_block,
        )
        facts = {}
    # Snapshot debug (ghi vào cột predictions, không gửi model) — phục hồi hành vi gốc đợt 1.
    feature_snapshot = build_feature_snapshot(timeframe_data, mode, current_price)
    market_snapshot = build_market_snapshot(timeframe_data, current_price_str)
    return {
        "timeframe_data": timeframe_data,
        "system_prompt": system_prompt,
        "current_price": current_price,
        "user_prompt": user_prompt,
        "funding_context": funding_ctx,
        "facts": facts if mode == "futures" else {},
        "market_snapshot": market_snapshot,
        "feature_snapshot": feature_snapshot,
    }


async def analyze_symbol(symbol: str, mode: str, user_id: int | None = None, chat_id: int | None = None) -> dict:
    """
    Async entry point used by Telegram handlers.

    Never call requests.get(), a synchronous AI API, or SQLite directly on the event loop.
    Blocking I/O is offloaded to a worker thread via asyncio.to_thread().
    """
    ensure_ai_config()

    await asyncio.to_thread(init_prediction_db)

    binance_symbol = resolve_binance_symbol(symbol, "futures" if mode == "futures" else "spot")
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
    user_prompt = ctx["user_prompt"]
    feature_snapshot = ctx["feature_snapshot"]
    market_snapshot = ctx["market_snapshot"]
    facts = ctx.get("facts") or {}

    if mode == "futures":
        return await _analyze_symbol_futures(
            binance_symbol=binance_symbol, mode=mode, user_id=user_id, chat_id=chat_id,
            ctx=ctx, timeframe_data=timeframe_data, system_prompt=system_prompt,
            current_price=current_price, user_prompt=user_prompt, facts=facts,
            market_snapshot=market_snapshot, feature_snapshot=feature_snapshot,
            manual_started=manual_started, loop=loop,
        )

    # The AI API call is synchronous, so it runs in a worker thread to avoid blocking the bot.
    print(f"[MANUAL_LLM_START] symbol={binance_symbol} mode={mode}", flush=True)
    raw_output = await asyncio.to_thread(request_json_analysis, system_prompt, user_prompt)
    print(
        f"[MANUAL_LLM_DONE] symbol={binance_symbol} mode={mode} elapsed={loop.time() - manual_started:.1f}s",
        flush=True,
    )
    planner_clean = raw_output if isinstance(raw_output, str) else str(raw_output or "")
    await asyncio.to_thread(
        _save_analysis_snapshot,
        user_id=user_id, chat_id=chat_id, symbol=binance_symbol, mode=mode, source="manual",
        model=get_ai_model_name(), planner_input=user_prompt, planner_output=planner_clean,
        current_price=current_price, public_output=planner_clean,
        funding_context=ctx.get("funding_context"),
    )

    # SPOT stays model-authoritative: no numeric validator or repair pass.
    # Người dùng Telegram nhận bản render; JSON thô trả qua khóa "json" cho agent dịch vụ.
    spot_pred = _extract_json_object(planner_clean) or {}
    spot_direction = str(spot_pred.get("quyet_dinh") or "").upper().replace(" ", "_").replace("-", "_")
    tracker_direction = "LONG" if spot_direction == "BUY" else spot_direction
    spot_entry_low = _num_or_none(spot_pred.get("entry_thap"))
    spot_entry_high = _num_or_none(spot_pred.get("entry_cao"))
    spot_sl = _num_or_none(spot_pred.get("sl"))
    spot_tp1 = _num_or_none(spot_pred.get("tp1"))
    spot_tp2 = _num_or_none(spot_pred.get("tp2"))
    spot_saved = False
    if (
        spot_direction == "BUY"
        and all(value is not None for value in (spot_entry_low, spot_entry_high, spot_sl, spot_tp1))
    ):
        await asyncio.to_thread(
            save_prediction,
            symbol=binance_symbol,
            mode=mode,
            direction=tracker_direction,
            entry_low=spot_entry_low,
            entry_high=spot_entry_high,
            sl=spot_sl,
            tp1=spot_tp1,
            tp2=spot_tp2,
            market_snapshot=market_snapshot,
            feature_snapshot=feature_snapshot,
            reasoning_summary=build_local_reasoning_summary(planner_clean),
            full_response=planner_clean,
            user_id=user_id,
            chat_id=chat_id,
            setup_status="TRADE",
        )
        spot_saved = True
    if spot_direction in ("BUY", "NO_TRADE"):
        spot_display_plan = {**spot_pred, "quyet_dinh": spot_direction}
    else:
        spot_display_plan = {"quyet_dinh": "NO_TRADE", "ly_do": "Không đọc được quyết định từ output."}
    spot_display = _strip_public_evidence_for_user(
        render_plan_text(spot_display_plan, binance_symbol, "SPOT", current_price)
    )
    if spot_saved:
        spot_display += (
            "\n\n✅ Bot đã tự lưu lệnh này để đánh giá. Gõ /history để xem danh sách lệnh đã lưu."
            "\n_Dữ liệu chỉ dùng để hiển thị và cải thiện prompt, không truyền cho Planner._"
        )
    print(
        f"[MANUAL_DONE] symbol={binance_symbol} mode={mode} elapsed={loop.time() - manual_started:.1f}s",
        flush=True,
    )
    return {"text": spot_display, "json": planner_clean}


async def _analyze_symbol_futures(
    *, binance_symbol: str, mode: str, user_id: int | None, chat_id: int | None,
    ctx: dict, timeframe_data: dict, system_prompt: str, current_price: float | None,
    user_prompt: str, facts: dict, market_snapshot: str | None,
    feature_snapshot: str | None, manual_started: float, loop,
) -> dict:
    from plan_validator import validate_plan

    print(f"[MANUAL_LLM_START] symbol={binance_symbol} mode={mode} variant=futures_json", flush=True)
    try:
        raw_output = await asyncio.to_thread(request_json_analysis, system_prompt, user_prompt)
    except Exception as exc:
        print(f"[MANUAL_FUTURES_ERROR] symbol={binance_symbol} error={exc}", flush=True)
        raise
    print(f"[MANUAL_LLM_DONE] symbol={binance_symbol} mode={mode} elapsed={loop.time() - manual_started:.1f}s", flush=True)
    planner_clean = (raw_output or "").strip()
    plan = _extract_json_object(planner_clean)
    if plan is None:
        raise RuntimeError("Planner không trả JSON hợp lệ.")
    errors = validate_plan(plan, facts)
    if errors:
        repair_text = (
            "Kế hoạch JSON của bạn bị lỗi kiểm tra số học sau (chỉ sửa số cho đúng, không đổi quan điểm thị trường "
            "nếu không cần; nếu sửa xong kế hoạch không còn đạt thì đổi sang NO_TRADE):\n"
            + "\n".join(f"- {e}" for e in errors)
            + "\n\nTrả lại đúng MỘT đối tượng JSON theo schema JSON mô tả trong system prompt, không thêm chữ ngoài JSON."
        )
        try:
            repaired_raw = await asyncio.to_thread(
                request_json_analysis, system_prompt,
                user_prompt + "\n\nKẾ HOẠCH TRƯỚC:\n" + planner_clean + "\n\nYÊU CẦU SỬA:\n" + repair_text,
            )
        except Exception as exc:
            print(f"[FUTURES_REPAIR_ERROR] {exc}", flush=True)
            repaired_raw = ""
        repaired = _extract_json_object(repaired_raw or "")
        if repaired is not None:
            plan = repaired
            errors = validate_plan(plan, facts)
    direction = str(plan.get("quyet_dinh") or "NO_TRADE").upper().replace(" ", "_").replace("-", "_")
    if direction not in ("LONG", "SHORT", "NO_TRADE"):
        print(f"[PLANNER_INVALID_DECISION] symbol={binance_symbol} mode={mode} quyet_dinh={plan.get('quyet_dinh')!r} -> ghi nhận như NO_TRADE", flush=True)
        direction = "NO_TRADE"
    # JSON thuần cho agent dịch vụ (trả qua khóa "json"); người dùng Telegram nhận bản render.
    output = json.dumps({**plan, "quyet_dinh": direction}, ensure_ascii=False)
    display = _strip_public_evidence_for_user(
        render_plan_text({**plan, "quyet_dinh": direction}, binance_symbol, "FUTURES", current_price)
    )
    usage_note = "\n\n_ℹ️ Lượt phân tích hôm nay vẫn bị tính._"
    tracking_note = (
        "\n\n✅ Bot đã tự lưu lệnh này để đánh giá. Gõ /history để xem danh sách lệnh đã lưu."
        "\n_Dữ liệu chỉ dùng để hiển thị và cải thiện prompt, không truyền cho Planner._"
    )
    direction_label = direction.replace("_", " ")
    await asyncio.to_thread(
        _save_analysis_snapshot,
        user_id=user_id, chat_id=chat_id, symbol=binance_symbol, mode=mode, source="manual",
        model=get_ai_model_name(), planner_input=user_prompt, planner_output=planner_clean,
        current_price=current_price, public_output=output,
        funding_context=ctx.get("funding_context"),
    )
    if direction == "NO_TRADE":
        return {"text": display + usage_note, "json": output}
    if errors:
        log_hidden_rejection(binance_symbol, mode, {
            "direction": direction_label,
            "entry_low": plan.get("entry_thap"), "entry_high": plan.get("entry_cao"),
            "sl": plan.get("sl"), "tp1": plan.get("tp1"),
        }, errors, output)
        guarded_plan = {
            "quyet_dinh": "NO_TRADE",
            "ly_do": "Kiểm tra số học của bot không đạt: " + "; ".join(str(e) for e in errors[:3]),
        }
        guarded = json.dumps(guarded_plan, ensure_ascii=False)
        guarded_text = render_plan_text(guarded_plan, binance_symbol, "FUTURES", current_price)
        return {"text": guarded_text + usage_note, "json": guarded}
    pred = {
        "direction": direction_label,
        "entry_low": plan.get("entry_thap"), "entry_high": plan.get("entry_cao"),
        "sl": plan.get("sl"), "tp1": plan.get("tp1"), "tp2": plan.get("tp2"),
    }
    await asyncio.to_thread(
        save_prediction,
        symbol=binance_symbol, mode=mode, direction=direction_label,
        entry_low=pred.get("entry_low"), entry_high=pred.get("entry_high"),
        sl=pred.get("sl"), tp1=pred.get("tp1"), tp2=pred.get("tp2"),
        market_snapshot=market_snapshot, feature_snapshot=feature_snapshot,
        reasoning_summary=str(plan.get("kich_hoat") or "")[:420], full_response=output,
        user_id=user_id, chat_id=chat_id, setup_status="TRADE",
    )
    print(f"[MANUAL_DONE] symbol={binance_symbol} mode={mode} elapsed={loop.time() - manual_started:.1f}s", flush=True)
    return {"text": display + tracking_note, "json": output}


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
        # Phiên quét theo (user, market): futures và spot độc lập, kèm cấu hình đặt lệnh.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS auto_scan_market_settings (
                user_id     INTEGER NOT NULL,
                market      TEXT NOT NULL,
                chat_id     INTEGER,
                enabled     INTEGER NOT NULL DEFAULT 0,
                symbol      TEXT NOT NULL DEFAULT '',
                night_resume INTEGER NOT NULL DEFAULT 0,
                qty         TEXT NOT NULL DEFAULT '',
                leverage    INTEGER NOT NULL DEFAULT 0,
                updated_at  TEXT NOT NULL,
                PRIMARY KEY(user_id, market)
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
        # Cột cho plan_id + giá + ids lệnh Binance của từng lệnh trong phiên.
        for col, definition in [
            ("plan_id", "TEXT"),
            ("entry_low", "REAL"),
            ("entry_high", "REAL"),
            ("sl", "REAL"),
            ("tp1", "REAL"),
            ("entry_order_id", "TEXT"),
            ("tp_algo_id", "TEXT"),
            ("sl_algo_id", "TEXT"),
            ("qty", "TEXT"),
            ("leverage", "INTEGER"),
            ("order_status", "TEXT"),
        ]:
            try:
                conn.execute(f"ALTER TABLE auto_scan_signals ADD COLUMN {col} {definition}")
            except sqlite3.OperationalError:
                pass
        # Không còn query nào filter auto_scan_settings.enabled (mọi truy vấn theo user_id
        # hoặc glm_calls_day) → bỏ index chết.
        conn.execute("DROP INDEX IF EXISTS idx_auto_scan_settings_enabled")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_auto_scan_signals_user_symbol_mode ON auto_scan_signals(user_id, symbol, mode, sent_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_auto_scan_logs_user_id ON auto_scan_logs(user_id, id DESC)")
        migrate_mode_values(conn)

        # Migration: dòng cài đặt cũ → phiên futures (bảng mới trống mới migrate; không drop bảng cũ).
        market_count = int(conn.execute("SELECT COUNT(*) FROM auto_scan_market_settings").fetchone()[0] or 0)
        if market_count == 0:
            legacy = conn.execute(
                "SELECT user_id, chat_id, enabled, symbols, night_resume FROM auto_scan_settings"
            ).fetchall()
            for user_id, chat_id, enabled, symbols_text, night_resume in legacy:
                first_symbol = (symbols_text or "").split(",")[0].strip()
                conn.execute(
                    """
                    INSERT OR IGNORE INTO auto_scan_market_settings
                        (user_id, market, chat_id, enabled, symbol, night_resume, updated_at)
                    VALUES (?, 'futures', ?, ?, ?, ?, ?)
                    """,
                    (user_id, chat_id, int(enabled or 0), first_symbol,
                     int(night_resume or 0), iso(utc_now())),
                )

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


def set_auto_scan_market_enabled(
    user_id: int, chat_id: int, market: str, enabled: bool, symbol: str,
    qty: str = "", leverage: int = 0,
) -> dict:
    """Bật/tắt phiên quét theo (user, market) + lưu cấu hình đặt lệnh của phiên."""
    init_auto_scan_db()
    symbol = normalize_auto_scan_symbol(symbol)
    quota_blocked = False
    if enabled:
        quota_state = get_auto_scan_glm_quota_state(user_id)
        quota_blocked = not quota_state.get("allowed")
    effective = bool(enabled and not quota_blocked)
    with sqlite3.connect(DB_PATH) as conn:
        # Dòng quota theo user phải tồn tại để window/quota logic hoạt động.
        conn.execute(
            """
            INSERT OR IGNORE INTO auto_scan_settings (user_id, chat_id, enabled, updated_at)
            VALUES (?, ?, 0, ?)
            """,
            (user_id, chat_id, iso(utc_now())),
        )
        if not enabled:
            conn.execute(
                "UPDATE auto_scan_settings SET quota_resume=0, night_resume=0 WHERE user_id=?",
                (user_id,),
            )
        conn.execute(
            """
            INSERT INTO auto_scan_market_settings
                (user_id, market, chat_id, enabled, symbol, night_resume, qty, leverage, updated_at)
            VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?)
            ON CONFLICT(user_id, market) DO UPDATE SET
                chat_id=excluded.chat_id,
                enabled=excluded.enabled,
                symbol=excluded.symbol,
                night_resume=0,
                qty=excluded.qty,
                leverage=excluded.leverage,
                updated_at=excluded.updated_at
            """,
            (user_id, market, chat_id, 1 if effective else 0, symbol,
             str(qty or ""), int(leverage or 0), iso(utc_now())),
        )
        # Đổi phiên sang symbol mới thì bỏ state trend-skip của symbol cũ.
        # PHẢI lọc theo mode: thiếu thì đổi symbol futures sẽ xóa luôn state trend của
        # spot (bảng chỉ có user_id/symbol/mode), làm hỏng bộ đếm 2 lần liên tiếp.
        try:
            if enabled and symbol:
                conn.execute(
                    "DELETE FROM auto_scan_trend_state WHERE user_id=? AND mode=? AND symbol<>?",
                    (user_id, market, symbol),
                )
            elif not enabled:
                # Tắt phiên → quên luôn bộ nhớ hướng; nếu giữ, lần quét đầu tiên của phiên
                # mới có thể kích hoạt skip 2 chu kỳ ngay lập tức từ dữ liệu của phiên cũ.
                conn.execute(
                    "DELETE FROM auto_scan_trend_state WHERE user_id=? AND mode=?",
                    (user_id, market),
                )
        except sqlite3.OperationalError:
            pass
        conn.commit()
    return {
        "enabled": effective,
        "quota_blocked": quota_blocked,
        "glm_calls_today": get_auto_scan_glm_quota_state(user_id).get("used", 0),
        "glm_calls_remaining": get_auto_scan_glm_quota_state(user_id).get("remaining", 0),
    }


def get_auto_scan_market_settings(user_id: int, market: str) -> dict | None:
    init_auto_scan_db()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT chat_id, enabled, symbol, night_resume, qty, leverage "
            "FROM auto_scan_market_settings WHERE user_id=? AND market=?",
            (user_id, market),
        ).fetchone()
    if row is None:
        return None
    return {
        "chat_id": row[0], "enabled": bool(row[1]), "symbol": row[2] or "",
        "night_resume": bool(row[3]), "qty": row[4] or "", "leverage": int(row[5] or 0),
    }


def next_session_plan_id(user_id: int, market: str, symbol: str) -> str:
    """futu-eth-1, futu-eth-2, ... — đếm theo phiên (off xóa dữ liệu nên tự reset)."""
    init_auto_scan_db()
    prefix = "futu" if market == "futures" else "spot"
    short = symbol[:-len(BINANCE_QUOTE_ASSET)] if symbol.endswith(BINANCE_QUOTE_ASSET) else symbol
    with sqlite3.connect(DB_PATH) as conn:
        count = int(conn.execute(
            "SELECT COUNT(*) FROM auto_scan_signals WHERE user_id=? AND mode=? AND symbol=?",
            (user_id, market, symbol),
        ).fetchone()[0] or 0)
    return f"{prefix}-{short.lower()}-{count + 1}"


def update_signal_orders(
    plan_id: str,
    *,
    prediction_id: int | None = None,
    user_id: int | None = None,
    plan_id_used: str | None = None,
    entry_order_id=None,
    tp_algo_id=None,
    sl_algo_id=None,
    qty=None,
    leverage=None,
    order_status: str = "placed",
) -> int:
    """Ghi orderId/algoId vào đúng dòng signal. Trả về số dòng được update.

    KHÔNG BAO GIỜ update chỉ với `WHERE plan_id=?`:
    - plan_id reset mỗi phiên ('futu-eth-1') nên hai user quét cùng symbol trùng nhau →
      update của user này ghi đè orderId sang dòng user kia;
    - executor có thể bump plan_id khi trùng clientOrderId (futu-eth-1 → futu-eth-2), làm
      2 dòng cùng user cùng plan_id → update 1 lệnh làm hỏng ledger dòng kia.
    Ưu tiên prediction_id (unique, không đổi); fallback plan_id + user_id.
    """
    init_auto_scan_db()
    sql = """
            UPDATE auto_scan_signals
            SET plan_id=COALESCE(?, plan_id), entry_order_id=COALESCE(?, entry_order_id),
                tp_algo_id=COALESCE(?, tp_algo_id), sl_algo_id=COALESCE(?, sl_algo_id),
                qty=COALESCE(?, qty), leverage=COALESCE(?, leverage), order_status=?
            WHERE 1=1
        """
    params: list = [
        plan_id_used, str(entry_order_id) if entry_order_id is not None else None,
        str(tp_algo_id) if tp_algo_id is not None else None,
        str(sl_algo_id) if sl_algo_id is not None else None,
        str(qty) if qty is not None else None,
        int(leverage) if leverage is not None else None,
        order_status,
    ]
    if prediction_id is not None:
        sql += " AND prediction_id=?"
        params.append(prediction_id)
    else:
        sql += " AND plan_id=?"
        params.append(plan_id)
        if user_id is not None:
            sql += " AND user_id=?"
            params.append(user_id)
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(sql, params)
        conn.commit()
        return int(cur.rowcount or 0)


def delete_session_signals(user_id: int, market: str, keep_placed: bool = False) -> int:
    """Xóa lịch sử lệnh của phiên (/autoscanoff*).

    keep_placed=True: GIỮ lại các dòng đã đặt lệnh trên sàn mà chưa hủy được (thiếu key /
    lỗi mạng). Xóa các dòng đó thì bot mất mỗi đường về orderId → không thể hủy sau.
    """
    init_auto_scan_db()
    sql = "DELETE FROM auto_scan_signals WHERE user_id=? AND mode=?"
    params: list = [user_id, market]
    if keep_placed:
        sql += " AND COALESCE(order_status,'')<>'placed'"
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(sql, params)
        conn.commit()
        return int(cur.rowcount or 0)


def delete_all_session_signals() -> int:
    """Xóa TOÀN BỘ lịch sử lệnh phiên (mọi user/market) — chạy 1 lần khi vào cửa sổ ngủ đêm.
    Chỉ xóa auto_scan_signals (log hiển thị); predictions (lịch đánh giá) giữ nguyên,
    lệnh đã đặt trên Binance cũng giữ nguyên."""
    init_auto_scan_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("DELETE FROM auto_scan_signals")
        conn.commit()
        return int(cur.rowcount or 0)


def list_session_signals(user_id: int, market: str) -> list[dict]:
    """Toàn bộ lệnh của phiên (không giới hạn 5) — dùng cho /autoscanlog*."""
    init_auto_scan_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT plan_id, symbol, direction, confidence, sent_at, qty, leverage,
                   entry_low, entry_high, sl, tp1,
                   entry_order_id, tp_algo_id, sl_algo_id, order_status, prediction_id
            FROM auto_scan_signals
            WHERE user_id=? AND mode=?
            ORDER BY id ASC
            """,
            (user_id, market),
        ).fetchall()
    return [dict(r) for r in rows]


def maintain_auto_scan_daily_window(now: datetime | None = None, allow_wipe: bool = True) -> dict:
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
    # sleep == wake (vd cả hai = 0) → coi là KHÔNG có cửa sổ ngủ. Với công thức cũ,
    # "hour >= sleep or hour < wake" luôn đúng → autoscan tắt vĩnh viễn không cảnh báo.
    in_sleep_window = (
        False if sleep_hour == wake_hour else
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
            # Chỉ tạm dừng phiên đang bật và KHÔNG bị pause vì hết quota (flag ở bảng cũ).
            cur = conn.execute(
                """
                UPDATE auto_scan_market_settings
                SET enabled=0, night_resume=1, updated_at=?
                WHERE enabled=1
                  AND user_id IN (SELECT user_id FROM auto_scan_settings WHERE quota_resume=0)
                """,
                (iso(current),),
            )
            disabled = int(cur.rowcount or 0)
        else:
            # Phiên tạm dừng ban đêm được bật lại khi ra khỏi cửa sổ ngủ.
            cur = conn.execute(
                """
                UPDATE auto_scan_market_settings
                SET enabled=1, night_resume=0, updated_at=?
                WHERE night_resume=1
                  AND user_id IN (SELECT user_id FROM auto_scan_settings WHERE quota_resume=0)
                """,
                (iso(current),),
            )
            resumed += int(cur.rowcount or 0)

            # User hết quota chỉ được bật lại sau khi quota ngày mới đã reset.
            cur = conn.execute(
                """
                UPDATE auto_scan_market_settings
                SET enabled=1, night_resume=0, updated_at=?
                WHERE user_id IN (
                    SELECT user_id FROM auto_scan_settings
                    WHERE quota_resume=1 AND glm_calls_day=? AND glm_calls_today=0
                )
                """,
                (iso(current), day_key),
            )
            resumed += int(cur.rowcount or 0)
            if resumed:
                # Dọn cờ quota_resume trên bảng cũ để các cửa sổ sau xử lý user này bình thường.
                conn.execute(
                    """
                    UPDATE auto_scan_settings
                    SET quota_resume=0, enabled=1, night_resume=0
                    WHERE quota_resume=1 AND glm_calls_day=? AND glm_calls_today=0
                    """,
                    (day_key,),
                )

        conn.commit()

    # Vào cửa sổ ngủ đêm (00:00–07:00): hủy MỌI lệnh TREO chưa khớp (theo ledger),
    # lệnh đã khớp giữ nguyên; rồi xóa lịch sử lệnh phiên của ngày cũ —
    # 1 lần mỗi đêm (idempotent theo ngày VN), 07:00 bật lại với log trống.
    #
    # CHỈ wipe khi hủy thành công: nếu thiếu API key (hoặc hủy lỗi) mà vẫn xóa hết dòng
    # auto_scan_signals thì ledger biến mất trong khi lệnh còn treo trên sàn → không còn
    # cách nào hủy/tra cứu sau đó. Lúc đó giữ nguyên + không set signals_wiped_day để
    # tick sau thử lại.
    wiped = 0
    # allow_wipe=False: lệnh CHỈ ĐỌC (vd /autoscanstatus) không được phép hủy lệnh +
    # xóa ledger của MỌI user — đó là việc của scheduler/job.
    if in_sleep_window and allow_wipe:
        wipe_day = local_now.strftime("%Y-%m-%d")
        if _auto_scan_state_get("signals_wiped_day") != wipe_day:
            cancel_ok = True
            try:
                cancel_result = cancel_pending_plan_orders_for(None, None)
                if cancel_result.get("no_keys"):
                    cancel_ok = False
                    print(
                        f"[CANCEL_PENDING] đêm nay thiếu key, GIỮ ledger: {cancel_result['no_keys']}",
                        flush=True,
                    )
            except Exception as exc:
                cancel_ok = False
                print(f"[CANCEL_PENDING] đêm nay hủy treo lỗi, GIỮ ledger: {exc}", flush=True)
            if cancel_ok:
                wiped = delete_all_session_signals()
                _auto_scan_state_set("signals_wiped_day", wipe_day)
            else:
                print("[CANCEL_PENDING] đêm nay KHÔNG wipe lịch sử để giữ tham chiếu lệnh", flush=True)

    return {
        "in_sleep_window": in_sleep_window,
        "disabled": disabled,
        "resumed": resumed,
        "quota_reset": quota_reset,
        "signals_wiped": wiped,
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
    """Các phiên đang bật — mỗi dòng là 1 (user, market) để scheduler quét riêng futures/spot."""
    init_auto_scan_db()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT user_id, chat_id, symbol, market FROM auto_scan_market_settings "
            "WHERE enabled=1 AND chat_id IS NOT NULL ORDER BY user_id, market"
        ).fetchall()
    return [
        {"user_id": int(r[0]), "chat_id": int(r[1]), "symbols": r[2] or "", "market": r[3]}
        for r in rows
    ]


def _normalize_auto_scan_modes() -> list[str]:
    result = []
    for m in AUTOSCAN_MODES or ["futures"]:
        mm = str(m).strip().lower()
        if mm in {"scalp", "short", "intraday", "futures", "15m"}:
            result.append("futures")
        elif mm in {"swing", "long", "spot", "4h"}:
            result.append("spot")
    return result or ["futures"]


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
    return resolve_binance_symbol(symbol, "spot")


def _record_auto_scan_signal(
    user_id: int, chat_id: int, symbol: str, mode: str, direction: str,
    confidence: int | None, prediction_id: int | None, plan_id: str | None = None,
    order_status: str = "pending",
    entry_low: float | None = None, entry_high: float | None = None,
    sl: float | None = None, tp1: float | None = None,
) -> None:
    init_auto_scan_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO auto_scan_signals
                (user_id, chat_id, symbol, mode, direction, confidence, sent_at, prediction_id,
                 plan_id, order_status, entry_low, entry_high, sl, tp1)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, chat_id, symbol, mode, direction, confidence, iso(utc_now()),
             prediction_id, plan_id, order_status, entry_low, entry_high, sl, tp1),
        )
        conn.commit()


def _rollback_auto_scan_signal(prediction_id: int | None) -> None:
    """Undo the auto_scan_signals row when Telegram send ultimately fails, so the signal-history
    log doesn't record a signal the user never actually saw. The prediction itself stays in /history.

    KHÔNG xóa dòng nào đã đặt lệnh (order_status='placed'): dòng đó là bản ghi duy nhất về
    entry/TP/SL orderId trên sàn. Xóa đi thì /offfutu, hủy đêm và autoscanlog đều không còn
    cách nào tìm lại lệnh — position treo vô chủ. Dòng đó được đổi sang 'send_failed' để
    vẫn hiện trong log nhưng biết tin nhắn chưa tới tay user.
    """
    if prediction_id is None:
        return
    init_auto_scan_db()
    with sqlite3.connect(DB_PATH) as conn:
        # XÓA trước, đổi trạng thái SAU. Đảo thứ tự thì dòng vừa đổi sang 'send_failed'
        # sẽ khớp ngay câu DELETE WHERE order_status<>'placed' và bị xóa mất.
        conn.execute(
            "DELETE FROM auto_scan_signals "
            "WHERE prediction_id=? AND COALESCE(order_status,'') NOT IN ('placed','send_failed')",
            (prediction_id,),
        )
        conn.execute(
            """
            UPDATE auto_scan_signals
            SET order_status='send_failed'
            WHERE prediction_id=? AND order_status='placed'
            """,
            (prediction_id,),
        )
        conn.commit()


_DEMO_SYMBOL_CACHE: dict[str, set] = {}


def _resolve_demo_symbol(exec_base: str, symbol: str) -> str:
    """Sàn demo có symbol riêng khớp giá live (ETHU cho ETHUSDT) — map nếu tồn tại trên demo,
    không có thì giữ nguyên symbol gốc. Kết quả cache theo base."""
    if not symbol.endswith("USDT"):
        return symbol
    cached = _DEMO_SYMBOL_CACHE.get(exec_base)
    if cached is None:
        symbols = set()
        resp = _binance_get_with_retry(f"{exec_base}/fapi/v1/exchangeInfo", {}, max_retries=1, timeout=15)
        if resp is not None:
            try:
                symbols = {s.get("symbol") for s in (resp.json().get("symbols") or [])}
            except Exception:
                symbols = set()
        cached = symbols or {symbol}
        _DEMO_SYMBOL_CACHE[exec_base] = cached
    candidate = symbol[:-4] + "U"
    return candidate if candidate in cached else symbol


async def _auto_execute_plan(
    *, user_id: int, mode: str, symbol: str, direction: str, plan: dict,
    plan_id: str, current_price: float | None,
) -> tuple[str, dict]:
    """Đặt lệnh thật nếu user có cấu hình qty cho phiên.

    Trả (block text, info):
    - {"status": "no_auto"} — không bật tự động (không có qty) → chỉ gửi tín hiệu, block "".
    - {"status": "aborted"} — bật tự động nhưng combo (entry+TP+SL) KHÔNG đặt được →
      caller KHÔNG lưu DB, KHÔNG gửi plan; block là thông báo hủy ngắn.
    - {"status": "placed", "order": {...}, "leverage": N} — đã đặt xong → block "ĐÃ ĐẶT LỆNH".
    """
    from key_store import KeyError_, get_api_keys

    import binance_executor as executor

    def _abort(reason: str, uncertain: bool = False) -> tuple[str, dict]:
        # Không nêu Entry/SL/TP của plan trong thông báo này — plan bị hủy thì không gửi plan.
        if uncertain:
            # Lỗi mạng: request có thể đã tới exchange nên lệnh CÓ THỂ đã treo —
            # nói "không có lệnh nào" là sai và làm user bỏ sót position.
            text = (
                f"⛔ Kế hoạch bị hủy — lỗi đặt lệnh: {reason}.\n"
                "⚠️ Trạng thái lệnh TRÊN SÀN KHÔNG XÁC ĐỊNH (lỗi mạng) — vào GUI kiểm tra "
                "lệnh treo của plan này giúp bot. Kế hoạch KHÔNG được lưu."
            )
            return text, {"status": "aborted", "reason": reason, "uncertain": True}
        text = (
            f"⛔ Kế hoạch bị hủy — không đặt được combo lệnh: {reason}.\n"
            "Không có lệnh nào trên sàn; kế hoạch KHÔNG được lưu và KHÔNG được gửi."
        )
        return text, {"status": "aborted", "reason": reason}

    cfg = get_auto_scan_market_settings(user_id, mode)
    qty_raw = str((cfg or {}).get("qty") or "").strip()
    if not qty_raw:
        return "", {"status": "no_auto"}
    qty_text = qty_raw
    if "," in qty_text and "." in qty_text:
        # Dấu xuất hiện CUỐI là dấu thập phân (1.000,50 = 1000.5 / 1,000.50 = 1000.5).
        qty_text = (qty_text.replace(".", "").replace(",", ".")
                    if qty_text.rfind(",") > qty_text.rfind(".")
                    else qty_text.replace(",", ""))
    elif "," in qty_text:
        qty_text = qty_text.replace(",", ".")
    try:
        qty = float(qty_text)
        if not math.isfinite(qty) or qty <= 0:
            raise ValueError
    except ValueError:
        return _abort(f"khối lượng '{qty_raw}' không hợp lệ")
    try:
        keys = get_api_keys(user_id, mode)
    except KeyError_ as exc:
        return _abort(str(exc))

    tp1 = _num_or_none(plan.get("tp1"))
    sl = _num_or_none(plan.get("sl"))
    entry_thap = _num_or_none(plan.get("entry_thap"))
    entry_cao = _num_or_none(plan.get("entry_cao"))
    if tp1 is None or sl is None or current_price is None:
        return _abort("plan thiếu TP1/SL hoặc thiếu giá hiện tại")
    if entry_thap is None or entry_cao is None:
        # Thiếu vùng entry: không đặt lệnh LIMIT, và caller sẽ không lưu được dòng signal
        # (can_track=False) → lệnh đặt ra không có trong sổ để /offfutu hủy được.
        return _abort("plan thiếu vùng Entry (entry_thap/entry_cao)")
    leverage = int((cfg or {}).get("leverage") or 0)
    if mode == "futures" and not 1 <= leverage <= 125:
        # leverage=0 làm executor bỏ qua /fapi/v1/leverage → lệnh chạy với đòn bẩy ĐANG GIỮ
        # trên symbol (có thể là 125x còn sót từ lệnh tay) — không bao giờ đặt thầm lặng như vậy.
        return _abort(f"đòn bẩy cấu hình là {leverage} (chưa hợp lệ) — bật lại phiên và nhập 1..125")

    # Giá THẬT tại thời điểm đặt lệnh (giá trong packet đã cũ — LLM suy nghĩ vài phút).
    real_now = await asyncio.to_thread(get_current_price_raw, symbol, mode)
    if real_now is None:
        real_now = float(current_price)

    if direction == "LONG":
        # Entry mua đặt ở đỉnh vùng plan (giá chạm từ trên xuống là khớp trước).
        entry_ref = entry_cao
        plan_stale = real_now >= tp1 or real_now <= sl
    else:
        entry_ref = entry_thap
        plan_stale = real_now <= tp1 or real_now >= sl
    if plan_stale:
        # Giá thật đã chạy ra ngoài cặp SL–TP1: plan hết hiệu lực — không treo lệnh mồ côi.
        return _abort("giá thật đã chạy ra ngoài SL/TP1 của plan (hết hiệu lực)")

    # Sàn đặt lệnh khác sàn phân tích (demo): dùng symbol riêng của demo (vd ETHU) —
    # symbol demo này khớp giá live nên Entry/TP/SL giữ nguyên từng số, không cần re-anchor.
    exec_symbol = symbol
    if mode == "futures":
        exec_base = executor.FUTURES_API_BASE
        if exec_base.rstrip("/") != BINANCE_FUTURES_API_BASE:
            exec_symbol = await asyncio.to_thread(_resolve_demo_symbol, exec_base, symbol)

    entry_price = entry_ref if entry_ref is not None else real_now

    try:
        result = await asyncio.to_thread(
            executor.place_plan, mode, exec_symbol, direction, float(entry_price),
            float(tp1), float(sl), qty, leverage or None, keys, plan_id,
        )
    except executor.ExecutorError as exc:
        # -1000 = lỗi mạng do executor gói lại: lệnh có thể đã được gửi đi trước khi mất kết nối.
        return _abort(f"mã {exc.code}: {exc.msg}", uncertain=(exc.code == -1000))
    except Exception as exc:  # lỗi không lường trước — không biết lệnh đã lên sàn hay chưa
        return _abort(str(exc), uncertain=True)

    used_plan = result.get("plan_id") or plan_id
    # Block cố tình ngắn gọn — orderId/algoId đầy đủ nằm DB và hiện trong /autoscanlog*.
    lines = ["", "🤖 ĐÃ ĐẶT LỆNH TỰ ĐỘNG:"]
    lev_note = f" | đòn bẩy x{leverage}" if mode == "futures" and leverage else ""
    lines.append(f"Plan: {used_plan} | qty {result.get('qty')}{lev_note}")
    if mode == "spot" and not result.get("filled"):
        status = result.get("status", "?")
        executed = float(result.get("executed") or 0)
        if result.get("remainder_cancelled") is False:
            lines.append(
                f"⚠️ Lệnh mua {status} sau 30s, KHÔNG hủy được phần còn lại "
                f"({result.get('cancel_error') or 'lỗi không rõ'}) — vào GUI kiểm tra."
            )
        elif executed > 0:
            lines.append(
                f"⏳ Lệnh mua {status} sau 30s — đã hủy phần chưa khớp, đã gắn TP/SL cho "
                f"{fmt(executed)} coin đã mua."
            )
        elif result.get("remainder_cancelled"):
            lines.append(
                f"⏳ Lệnh mua không khớp trong 30s ({status}) — ĐÃ HỦY lệnh mua, không có vị thế nào."
            )
        else:
            lines.append(f"⏳ Lệnh mua {status} — chưa có vị thế nào trên sàn.")
    return ("\n" + "\n".join(lines),
            {"status": "placed", "order": result, "leverage": leverage})


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


def get_auto_scan_runtime_status(user_id: int) -> dict:
    # Chỉ đọc: không cho lệnh status trigger hủy lệnh + xóa ledger của mọi user.
    window = maintain_auto_scan_daily_window(allow_wipe=False)
    slot = _auto_scan_slot_info()
    quota = get_auto_scan_glm_quota_state(user_id)
    init_auto_scan_db()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT market, enabled, symbol, night_resume, qty, leverage "
            "FROM auto_scan_market_settings WHERE user_id=? ORDER BY market",
            (user_id,),
        ).fetchall()
        qrow = conn.execute(
            "SELECT quota_resume FROM auto_scan_settings WHERE user_id=?", (user_id,)
        ).fetchone()
    markets = [
        {"market": r[0], "enabled": bool(r[1]), "symbol": r[2] or "",
         "night_resume": bool(r[3]), "qty": r[4] or "", "leverage": int(r[5] or 0)}
        for r in rows
    ]
    primary = next((m for m in markets if m["market"] == "futures"), markets[0] if markets else None)
    return {
        "markets": markets,
        "enabled": bool(primary and primary["enabled"]),
        "night_resume": bool(primary and primary["night_resume"]),
        "symbol": (primary or {}).get("symbol", ""),
        "symbols": (primary or {}).get("symbol", ""),
        "quota_resume": bool(qrow and qrow[0]),
        "glm_calls_today": quota.get("used", 0),
        "glm_calls_remaining": quota.get("remaining", AUTOSCAN_MAX_PLANNER_CALLS_PER_DAY),
        "last_scan_slot": _auto_scan_state_get("last_scan_slot"),
        "last_scan_at": _auto_scan_state_get("last_scan_at"),
        "current_slot": slot.get("slot"),
        "next_scan_at": slot.get("next_slot"),
        "in_sleep_window": bool(window.get("in_sleep_window")),
        "sleep_hour_vn": int(window.get("sleep_hour", AUTOSCAN_SLEEP_HOUR_VN)),
        "wake_hour_vn": int(window.get("wake_hour", AUTOSCAN_WAKE_HOUR_VN)),
    }



def _auto_scan_text_header(symbol: str, mode: str) -> str:
    mode_label = "FUTURES" if mode == "futures" else "SPOT"
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
    binance_symbol = resolve_binance_symbol(symbol, "futures" if mode == "futures" else "spot")
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
    facts = ctx.get("facts") or {}

    quota = await asyncio.to_thread(reserve_auto_scan_glm_call, user_id)
    if not quota.get("allowed"):
        return await log_and_return(
            "quota", "skipped",
            f"Đã dùng đủ {AUTOSCAN_MAX_PLANNER_CALLS_PER_DAY} lượt gọi AI cuối trong ngày Auto Scan; sẽ tự bật lại lúc 07:00 VN.",
        )

    if mode == "futures":
        return await _auto_scan_futures(
            symbol=binance_symbol, mode=mode, user_id=user_id, chat_id=chat_id,
            scan_slot=scan_slot, ctx=ctx, timeframe_data=timeframe_data,
            system_prompt=system_prompt, current_price=current_price,
            market_snapshot=market_snapshot, feature_snapshot=feature_snapshot,
            facts=facts, log_and_return=log_and_return,
        )

    user_prompt = ctx["user_prompt"]
    planner_input = user_prompt
    try:
        raw_output = await asyncio.to_thread(request_json_analysis, system_prompt, planner_input)
        planner_clean = (raw_output or "").strip()
    except Exception:
        # The quota slot was already reserved above; refund it so a Planner outage
        # (timeout, bad config, sustained API error) doesn't silently burn the day's quota.
        await asyncio.to_thread(_refund_auto_scan_glm_call, user_id)
        raise
    output = planner_clean
    plan = _extract_json_object(output)
    if plan is None:
        # Model gọi thành công nhưng trả rác → không dùng được slot nào, hoàn lại
        # (futures vẫn hoàn trong cùng trường hợp; thiếu ở spot làm ngày quota hết sớm).
        await asyncio.to_thread(_refund_auto_scan_glm_call, user_id)
        return await log_and_return("planner", "rejected", "Planner không trả JSON hợp lệ cho Spot.", final_direction="UNKNOWN")

    decision = str(plan.get("quyet_dinh") or "").upper().replace(" ", "_").replace("-", "_")
    if decision not in {"BUY", "NO_TRADE"}:
        await asyncio.to_thread(_refund_auto_scan_glm_call, user_id)
        return await log_and_return("planner", "rejected", "Spot chỉ chấp nhận quyết định BUY hoặc NO_TRADE.", final_direction=decision or "UNKNOWN")
    direction = "LONG" if decision == "BUY" else "NO_TRADE"
    await asyncio.to_thread(_auto_scan_update_trend_state, user_id, binance_symbol, mode, direction)
    await asyncio.to_thread(
        _save_analysis_snapshot,
        user_id=user_id, chat_id=chat_id, symbol=binance_symbol, mode=mode, source="autoscan",
        model=get_ai_model_name(),
        planner_input=planner_input, planner_output=planner_clean,
        current_price=current_price, public_output=output,
        funding_context=ctx.get("funding_context"),
    )
    final_conf = plan.get("do_tin_cay")
    final_conf = int(final_conf) if final_conf is not None else None
    display = _strip_public_evidence_for_user(
        render_plan_text({**plan, "quyet_dinh": decision}, binance_symbol, "SPOT", current_price)
    )
    if direction == "NO_TRADE":
        # Không gửi NO TRADE (trừ khi admin bật AUTOSCAN_SEND_NO_TRADE để gỡ rối).
        if AUTOSCAN_SEND_NO_TRADE:
            return {
                "send": True,
                "text": display,
                "json": output,
                "prediction_id": None,
                "direction": direction,
                "confidence": final_conf,
                "final_direction": direction,
                "final_confidence": final_conf,
            }
        return await log_and_return(
            "planner", "rejected", "Planner chọn NO TRADE sau phân tích đầy đủ.",
            final_direction=direction, final_confidence=final_conf,
        )

    entry_low = _num_or_none(plan.get("entry_thap"))
    entry_high = _num_or_none(plan.get("entry_cao"))
    sl = _num_or_none(plan.get("sl"))
    tp1 = _num_or_none(plan.get("tp1"))
    tp2 = _num_or_none(plan.get("tp2"))
    can_track = all(value is not None for value in (entry_low, entry_high, sl, tp1))
    prediction_id = None
    plan_id = await asyncio.to_thread(next_session_plan_id, user_id, mode, binance_symbol)
    # Đặt lệnh TRƯỚC, lưu DB SAU: combo không đặt được → không lưu prediction/signal, không gửi plan.
    order_block, exec_info = await _auto_execute_plan(
        user_id=user_id, mode=mode, symbol=binance_symbol, direction="LONG",
        plan=plan, plan_id=plan_id, current_price=current_price,
    )
    if exec_info.get("status") == "aborted":
        return {
            "send": True,
            "text": _auto_scan_text_header(binance_symbol, mode) + "\n" + order_block,
            "json": output, "prediction_id": None,
            "direction": decision, "confidence": final_conf,
            "final_direction": decision, "final_confidence": final_conf,
            "reason": exec_info.get("reason"),
        }
    if can_track:
        prediction_id = await asyncio.to_thread(
            save_prediction,
            symbol=binance_symbol,
            mode=mode,
            direction="LONG",
            entry_low=entry_low,
            entry_high=entry_high,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            market_snapshot=market_snapshot,
            feature_snapshot=feature_snapshot,
            reasoning_summary=build_local_reasoning_summary(output),
            full_response=output,
            user_id=user_id,
            chat_id=chat_id,
            setup_status="TRADE",
        )
        try:
            if _price_in_entry_range(current_price, entry_low, entry_high):
                entry_price = _entry_price("LONG", entry_low, entry_high, current_price)
                if entry_price is not None:
                    await asyncio.to_thread(mark_entry_filled, prediction_id, float(entry_price), utc_now(), mode)
        except Exception:
            pass
        order_status = "placed" if exec_info.get("status") == "placed" else "no_auto"
        await asyncio.to_thread(
            _record_auto_scan_signal, user_id, chat_id, binance_symbol, mode, "BUY",
            final_conf, int(prediction_id), plan_id, order_status=order_status,
            entry_low=entry_low, entry_high=entry_high, sl=sl, tp1=tp1,
        )
        if exec_info.get("status") == "placed":
            order = exec_info.get("order") or {}
            await asyncio.to_thread(
                update_signal_orders, plan_id,
                prediction_id=int(prediction_id),
                user_id=user_id,
                plan_id_used=order.get("plan_id"),
                entry_order_id=order.get("entry_order_id"),
                tp_algo_id=order.get("tp_algo_id") or order.get("oco_list_id"),
                sl_algo_id=order.get("sl_algo_id"),
                qty=str(order.get("qty") or ""),
                leverage=exec_info.get("leverage") or 0,
                order_status="placed",
            )

    return {
        "send": True,
        "text": display + order_block,
        "json": output,
        "prediction_id": int(prediction_id) if prediction_id is not None else None,
        "direction": decision,
        "confidence": final_conf,
        "final_direction": decision,
        "final_confidence": final_conf,
    }


async def _auto_scan_futures(
    *, symbol: str, mode: str, user_id: int, chat_id: int, scan_slot: str | None,
    ctx: dict, timeframe_data: dict, system_prompt: str, current_price: float | None,
    market_snapshot: str | None, feature_snapshot: str | None, facts: dict,
    log_and_return,
) -> dict:
    from plan_validator import validate_plan

    binance_symbol = symbol
    user_prompt = ctx["user_prompt"]
    # Ý "vào được ngay bây giờ" đã nằm trong system prompt dùng chung cho Manual và Auto Scan
    # (hai trạng thái TRADE/NO_TRADE) → không cần flash_note riêng nữa.
    planner_input = user_prompt
    try:
        raw_output = await asyncio.to_thread(request_json_analysis, system_prompt, planner_input)
        planner_clean = (raw_output or "").strip()
    except Exception:
        await asyncio.to_thread(_refund_auto_scan_glm_call, user_id)
        raise
    plan = _extract_json_object(planner_clean)
    if plan is None:
        await asyncio.to_thread(_refund_auto_scan_glm_call, user_id)
        return await log_and_return("planner", "rejected", "Planner không trả JSON hợp lệ.", final_direction="UNKNOWN")
    errors = validate_plan(plan, facts)
    if errors:
        # Quy tắc quota: mỗi lần gọi model thật (kể cả lần sửa lỗi) đều phải reserve 1 slot.
        repair_quota = await asyncio.to_thread(reserve_auto_scan_glm_call, user_id)
        if not repair_quota.get("allowed"):
            return await log_and_return(
                "quota", "skipped",
                f"Hết quota ({AUTOSCAN_MAX_PLANNER_CALLS_PER_DAY} lượt/ngày) nên không gọi được lần sửa lỗi; kế hoạch bị loại.",
                final_direction="UNKNOWN")
        repair_text = (
            "Kế hoạch JSON của bạn bị lỗi kiểm tra số học sau (chỉ sửa số cho đúng; "
            "nếu sửa xong kế hoạch không còn đạt thì đổi sang NO_TRADE):\n"
            + "\n".join(f"- {e}" for e in errors)
            + "\n\nTrả lại đúng MỘT đối tượng JSON theo schema JSON mô tả trong system prompt."
        )
        try:
            repaired_raw = await asyncio.to_thread(
                request_json_analysis, system_prompt,
                planner_input + "\n\nKẾ HOẠCH TRƯỚC:\n" + planner_clean + "\n\nYÊU CẦU SỬA:\n" + repair_text)
            repaired = _extract_json_object((repaired_raw or "").strip())
        except Exception:
            # Lần gọi repair lỗi: refund đúng slot của lần repair, slot lần đầu vẫn giữ.
            await asyncio.to_thread(_refund_auto_scan_glm_call, user_id)
            raise
        if repaired is None:
            # Lần repair đã reserve 1 slot mà không ra kết quả → hoàn slot đó.
            await asyncio.to_thread(_refund_auto_scan_glm_call, user_id)
            return await log_and_return("planner", "rejected", "Planner sửa lỗi nhưng không trả JSON hợp lệ.", final_direction="UNKNOWN")
        plan = repaired
        errors = validate_plan(plan, facts)
    direction = str(plan.get("quyet_dinh") or "NO_TRADE").upper().replace(" ", "_").replace("-", "_")
    if direction not in ("LONG", "SHORT", "NO_TRADE"):
        print(f"[PLANNER_INVALID_DECISION] symbol={binance_symbol} mode={mode} quyet_dinh={plan.get('quyet_dinh')!r} -> ghi nhận như NO_TRADE", flush=True)
        direction = "NO_TRADE"
        plan = {**plan, "quyet_dinh": "NO_TRADE"}
    final_conf = plan.get("do_tin_cay")
    try:
        final_conf = int(final_conf) if final_conf is not None else None
    except Exception:
        final_conf = None
    await asyncio.to_thread(_auto_scan_update_trend_state, user_id, binance_symbol, mode, direction)
    output = json.dumps({**plan, "quyet_dinh": direction}, ensure_ascii=False)
    display = _strip_public_evidence_for_user(
        render_plan_text({**plan, "quyet_dinh": direction}, binance_symbol, "FUTURES", current_price)
    )
    await asyncio.to_thread(
        _save_analysis_snapshot,
        user_id=user_id, chat_id=chat_id, symbol=binance_symbol, mode=mode, source="autoscan",
        model=get_ai_model_name(), planner_input=planner_input, planner_output=planner_clean,
        current_price=current_price, public_output=output,
        funding_context=ctx.get("funding_context"),
    )
    direction_label = direction.replace("_", " ")
    # Hai trạng thái: chỉ LONG/SHORT mới gửi; NO_TRADE thì bỏ qua.
    if direction == "NO_TRADE":
        if AUTOSCAN_SEND_NO_TRADE:
            return {"send": True, "text": _auto_scan_text_header(binance_symbol, mode) + display,
                    "json": output, "prediction_id": None}
        return await log_and_return(
            "planner", "rejected", "Planner chọn NO TRADE sau phân tích đầy đủ.",
            final_direction=direction, final_confidence=final_conf,
        )
    if direction not in {"LONG", "SHORT"}:
        return await log_and_return("planner", "rejected", "Planner không trả quyết định LONG/SHORT hợp lệ.", final_direction=direction, final_confidence=final_conf)
    if errors:
        log_hidden_rejection(binance_symbol, mode, {
            "direction": direction_label,
            "entry_low": plan.get("entry_thap"), "entry_high": plan.get("entry_cao"),
            "sl": plan.get("sl"), "tp1": plan.get("tp1"),
        }, errors, output)
        return await log_and_return("guard", "rejected", "guard rejected", final_direction=direction, final_confidence=final_conf)
    if any(plan.get(k) is None for k in ("entry_thap", "entry_cao", "sl", "tp1")):
        return await log_and_return("planner", "rejected", "Planner thiếu Entry/SL/TP bắt buộc", final_direction=direction, final_confidence=final_conf)
    plan_id = await asyncio.to_thread(next_session_plan_id, user_id, mode, binance_symbol)
    # Đặt lệnh TRƯỚC, lưu DB SAU: combo không đặt được → không lưu prediction/signal, không gửi plan.
    order_block, exec_info = await _auto_execute_plan(
        user_id=user_id, mode=mode, symbol=binance_symbol, direction=direction,
        plan=plan, plan_id=plan_id, current_price=current_price,
    )
    if exec_info.get("status") == "aborted":
        return {
            "send": True,
            "text": _auto_scan_text_header(binance_symbol, mode) + "\n" + order_block,
            "json": output, "prediction_id": None,
            "direction": direction_label, "confidence": final_conf,
            "final_direction": direction, "final_confidence": final_conf,
            "reason": exec_info.get("reason"),
        }
    prediction_id = await asyncio.to_thread(
        save_prediction,
        symbol=binance_symbol, mode=mode, direction=direction_label,
        entry_low=plan.get("entry_thap"), entry_high=plan.get("entry_cao"),
        sl=plan.get("sl"), tp1=plan.get("tp1"), tp2=plan.get("tp2"),
        market_snapshot=market_snapshot, feature_snapshot=feature_snapshot,
        reasoning_summary=str(plan.get("kich_hoat") or "")[:420], full_response=output,
        user_id=user_id, chat_id=chat_id, setup_status="TRADE",
    )
    try:
        if _price_in_entry_range(current_price, plan.get("entry_thap"), plan.get("entry_cao")):
            entry_price = _entry_price(direction_label, plan.get("entry_thap"), plan.get("entry_cao"), current_price)
            if entry_price is not None:
                await asyncio.to_thread(mark_entry_filled, prediction_id, float(entry_price), utc_now(), mode)
    except Exception:
        pass
    order_status = "placed" if exec_info.get("status") == "placed" else "no_auto"
    await asyncio.to_thread(
        _record_auto_scan_signal, user_id, chat_id, binance_symbol, mode, direction_label,
        final_conf, int(prediction_id), plan_id, order_status=order_status,
        entry_low=plan.get("entry_thap"), entry_high=plan.get("entry_cao"),
        sl=plan.get("sl"), tp1=plan.get("tp1"),
    )
    if exec_info.get("status") == "placed":
        order = exec_info.get("order") or {}
        await asyncio.to_thread(
            update_signal_orders, plan_id,
            prediction_id=int(prediction_id),
            user_id=user_id,
            plan_id_used=order.get("plan_id"),
            entry_order_id=order.get("entry_order_id"),
            tp_algo_id=order.get("tp_algo_id") or order.get("oco_list_id"),
            sl_algo_id=order.get("sl_algo_id"),
            qty=str(order.get("qty") or ""),
            leverage=exec_info.get("leverage") or 0,
            order_status="placed",
        )
    execution_note = "\n\n✅ Có thể vào lệnh theo kế hoạch trong vùng Entry." + order_block
    text = (
        _auto_scan_text_header(binance_symbol, mode)
        + display
        + execution_note
        + "\n\nBot đã tự lưu tín hiệu Auto Scan này để theo dõi."
    )
    return {
        "send": True, "text": text, "json": output, "prediction_id": int(prediction_id),
        "direction": direction_label, "confidence": final_conf,
        "final_direction": direction, "final_confidence": final_conf,
    }


def cancel_pending_plan_orders_for(user_id: int | None = None, market: str | None = None) -> dict:
    """Hủy MỌI lệnh TREO (chưa khớp) theo ledger auto_scan_signals của user/market
    (None = tất cả). Lệnh đã khớp giữ nguyên. Thiếu key → ghi nhận để báo user tự hủy tay."""
    from key_store import KeyError_, get_api_keys

    import binance_executor as executor

    init_auto_scan_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        sql = "SELECT user_id, mode, symbol, entry_order_id, tp_algo_id, sl_algo_id " \
              "FROM auto_scan_signals WHERE order_status='placed'"
        params: list = []
        if user_id is not None:
            sql += " AND user_id=?"
            params.append(user_id)
        if market is not None:
            sql += " AND mode=?"
            params.append(market)
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]

    cancelled: list[dict] = []
    no_keys: list[tuple[int, str]] = []
    grouped: dict[tuple[int, str], list[dict]] = {}
    for r in rows:
        grouped.setdefault((int(r["user_id"]), str(r["mode"])), []).append(r)
    for (uid, mkt), group in grouped.items():
        try:
            keys = get_api_keys(uid, mkt)
        except KeyError_:
            no_keys.append((uid, mkt))
            continue
        try:
            items = executor.cancel_pending_plan_orders(keys, mkt, group)
        except Exception as exc:
            print(f"[CANCEL_PENDING] user={uid} market={mkt} lỗi: {exc}", flush=True)
            continue
        for item in items:
            item["user_id"] = uid
            cancelled.append(item)
            print(f"[CANCEL_PENDING] user={uid} market={mkt} đã hủy {item}", flush=True)
    return {"cancelled": cancelled, "no_keys": no_keys}


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

    # Claim slot NGAY TẠI ĐÂY (không phải ở cuối cycle): đánh dấu đã quét trước khi đặt lệnh.
    # Trước đây mark ở cuối → nếu process chết giữa chừng (Railway redeploy, OOM) thì
    # restart chạy lại cùng slot, `next_session_plan_id` lại sinh id mới → ĐẶT LỆNH TRÙNG
    # cho cùng một tín hiệu. Claim ở đầu đánh đổi: cycle lỗi/mất điện sẽ mất 1 vòng quét
    # (không retry) — an toàn hơn nhiều so với đặt lệnh trùng.
    slot_claimed = slot_info.get("slot") or iso(utc_now())
    try:
        await asyncio.to_thread(mark_auto_scan_slot_done, slot_claimed)
    except Exception as exc:
        print(f"[AUTO_SCAN] claim_slot lỗi (slot có thể bị quét lại): {exc}", flush=True)

    users = await asyncio.to_thread(get_auto_scan_enabled_users)
    modes = sorted({u.get("market") or "futures" for u in users}) or _normalize_auto_scan_modes()
    payload = {"users": len(users), "symbols": 0, "modes": modes, "sent": 0, "checked": 0, "errors": 0, "skipped": False, "slot": slot_claimed, "next_scan_at": slot_info.get("next_slot")}
    if not users:
        return payload
    for user in users:
        mode = user.get("market") or "futures"
        symbols = _parse_auto_scan_symbols_text(user.get("symbols")) or _auto_scan_symbols_from_env_or_db()
        payload["symbols"] += len(symbols)
        if not symbols:
            continue
        for symbol in symbols:
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
    # Slot đã được claim ở đầu cycle — không mark lại ở đây. Xem lý do ở claim phía trên.
    if payload["errors"]:
        print(
            f"[AUTO_SCAN] slot={payload.get('slot')} hoàn tất với {payload['errors']} lỗi "
            f"(đã gửi {payload['sent']}/{payload['checked']})",
            flush=True,
        )
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

