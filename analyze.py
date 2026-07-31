import asyncio
import json
import math
import os
import re
import sqlite3
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
    prompt_hash,
    save_evaluation_case,
)

load_dotenv()

BINANCE_API_URL   = "https://api.binance.com/api/v3/klines"


def _env_int(name: str, default: int) -> int:
    """Parse integer env safely; invalid or blank values fall back to default."""
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


# ─── AI provider config ───────────────────────────────────────────────────────
# Current default provider is DeepSeek (see the AI_PROVIDER default below).
# OpenRouter/Z.AI/Claude code paths are kept so old imports don't break, and so the provider can be switched via a Railway variable when needed.
AI_PROVIDER = os.getenv("AI_PROVIDER", "deepseek").strip().lower()

# Native DeepSeek for the final analysis layer. Kept separate from the DEEPSEEK_* prefilter Flash settings.
# Can share a single API key; DEEPSEEK_FINAL_API_KEY falls back to DEEPSEEK_API_KEY.
DEEPSEEK_FINAL_API_KEY = os.getenv("DEEPSEEK_FINAL_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_FINAL_BASE_URL = os.getenv("DEEPSEEK_FINAL_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_FINAL_MODEL = os.getenv("DEEPSEEK_FINAL_MODEL", "deepseek-v4-pro")
# Defaults to "high" since the main goal is cost control at scale. Can be changed to "max" on Railway.
DEEPSEEK_FINAL_REASONING_EFFORT = os.getenv("DEEPSEEK_FINAL_REASONING_EFFORT", "max").strip()
DEEPSEEK_FINAL_RETRY_REASONING_EFFORT = os.getenv(
    "DEEPSEEK_FINAL_RETRY_REASONING_EFFORT", DEEPSEEK_FINAL_REASONING_EFFORT or "max"
).strip()
DEEPSEEK_FINAL_SUMMARY_REASONING_EFFORT = os.getenv(
    "DEEPSEEK_FINAL_SUMMARY_REASONING_EFFORT", "off"
).strip()

# Max reasoning shares the same completion token budget as the final answer.
# The cap must be large enough that after reasoning, the model still has room to output a parseable format.
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "12000"))
LLM_MAIN_OUTPUT_TOKEN_CAP = int(os.getenv("LLM_MAIN_OUTPUT_TOKEN_CAP", "12000"))
# Main analysis has no continuation: output is short, and continuation would just turn one request into multiple rounds that can hang.
LLM_MAX_CONTINUATIONS = int(os.getenv("LLM_MAX_CONTINUATIONS", "0"))
# Timeout/retry settings for the AI provider.
# GLM uses max reasoning for both the first attempt and the retry; the retry is still capped at one attempt.
LLM_MAIN_TIMEOUT_SECONDS = int(os.getenv("LLM_MAIN_TIMEOUT_SECONDS", "240"))
LLM_RETRY_TIMEOUT_SECONDS = int(os.getenv("LLM_RETRY_TIMEOUT_SECONDS", "150"))
LLM_SUMMARY_TIMEOUT_SECONDS = int(os.getenv("LLM_SUMMARY_TIMEOUT_SECONDS", "60"))
LLM_API_RETRIES = int(os.getenv("LLM_API_RETRIES", "1"))
LLM_MAIN_RETRY_LIMIT = int(os.getenv("LLM_MAIN_RETRY_LIMIT", "1"))
LLM_RETRY_SLEEP_SECONDS = float(os.getenv("LLM_RETRY_SLEEP_SECONDS", "2"))

# ─── Auto Scan mode config ──────────────────────────────────────────────────
# Auto Scan is a separate mode: DeepSeek Flash runs a quick filter every 15 minutes, and the final AI
# only runs a deep analysis when the prefilter sees a signal that's good enough.
AUTO_SCAN_INTERVAL_SECONDS = int(os.getenv("AUTO_SCAN_INTERVAL_SECONDS", "900"))
AUTO_SCAN_MODES = [m.strip().lower() for m in os.getenv("AUTO_SCAN_MODES", "short").split(",") if m.strip()]
AUTO_SCAN_MIN_PREFILTER_CONFIDENCE = int(os.getenv("AUTO_SCAN_MIN_PREFILTER_CONFIDENCE", "72"))
# If LONG/SHORT scores are too close together, the prefilter treats it as NEUTRAL and skips the final AI call.
# This is the minimum gap between the two mini-rubric totals, not a confidence percentage.
AUTO_SCAN_PREFILTER_MIN_DIRECTION_GAP = max(
    0,
    min(100, int(os.getenv("AUTO_SCAN_PREFILTER_MIN_DIRECTION_GAP", "20"))),
)
# Auto Scan uses a single final rubric, self-scored by the final AI: Signal Score /100.
# The new variable name takes priority; the old name is kept as a fallback so deploys don't break if Railway still has it.
AUTO_SCAN_MIN_FINAL_SIGNAL_SCORE = int(os.getenv(
    "AUTO_SCAN_MIN_FINAL_SIGNAL_SCORE",
    os.getenv("AUTO_SCAN_MIN_FINAL_CONFIDENCE", "72"),
))
AUTO_SCAN_MIN_FINAL_CONFIDENCE = AUTO_SCAN_MIN_FINAL_SIGNAL_SCORE
AUTO_SCAN_SIGNAL_COOLDOWN_MINUTES = int(os.getenv("AUTO_SCAN_SIGNAL_COOLDOWN_MINUTES", "180"))
AUTO_SCAN_MAX_SYMBOLS_PER_RUN = 1  # Auto Scan only allows 1 symbol per user to avoid wasting resources.
AUTO_SCAN_SEND_NO_TRADE = os.getenv("AUTO_SCAN_SEND_NO_TRADE", "0").strip().lower() in {"1", "true", "yes", "on"}
AUTO_SCAN_CANDLE_CLOSE_DELAY_SECONDS = int(os.getenv("AUTO_SCAN_CANDLE_CLOSE_DELAY_SECONDS", "5"))
# Job scheduler only wakes up to check whether a candle-close slot is due.
# It does NOT call Binance/LLM unless should_run_auto_scan_now() returns true.
AUTO_SCAN_SCHEDULER_TICK_SECONDS = max(30, int(os.getenv("AUTO_SCAN_SCHEDULER_TICK_SECONDS", "60") or "60"))
# The user-facing log only ever keeps the 5 most recent entries. This is fixed in code so the old
# Railway variable AUTO_SCAN_LOG_LIMIT=20 doesn't accidentally make the DB/Telegram log long again.
AUTO_SCAN_LOG_LIMIT = 5  # number of rows shown to the user
AUTO_SCAN_LOG_RETENTION_DAYS = max(1, int(os.getenv("AUTO_SCAN_LOG_RETENTION_DAYS", "14")))
AUTO_SCAN_DEBUG = os.getenv("AUTO_SCAN_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}

# Prevent overlapping Auto Scan cycles. If a candle-close slot arrives while a cycle
# is still running, the active cycle performs at most one catch-up pass for the
# newest closed slot after it finishes. Older missed slots are intentionally skipped.
_AUTO_SCAN_RUN_LOCK = asyncio.Lock()
_AUTO_SCAN_CATCH_UP_MAX_PASSES = 1
# Auto Scan sleep window in Vietnam time: 00:00-07:00.
AUTO_SCAN_SLEEP_HOUR_VN = int(os.getenv("AUTO_SCAN_SLEEP_HOUR_VN", "0"))
AUTO_SCAN_WAKE_HOUR_VN = int(os.getenv("AUTO_SCAN_WAKE_HOUR_VN", "7"))
# Each user can call the final AI at most N times per Auto Scan day (07:00 VN to 06:59 the next day).
AUTO_SCAN_MAX_GLM_CALLS_PER_DAY = max(
    1,
    int(os.getenv("AUTO_SCAN_MAX_FINAL_AI_CALLS_PER_DAY", os.getenv("AUTO_SCAN_MAX_GLM_CALLS_PER_DAY", "5"))),
)
# New name for display/new code; old name kept for backward compatibility with the DB/old Railway variables.
AUTO_SCAN_MAX_FINAL_AI_CALLS_PER_DAY = AUTO_SCAN_MAX_GLM_CALLS_PER_DAY

# DeepSeek filter: uses the OpenAI-compatible Chat Completions API. Defaults to OpenRouter
# so you can use deepseek/deepseek-v4-flash or an equivalent model via Railway.
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_TIMEOUT_SECONDS = int(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "60"))
DEEPSEEK_MAX_OUTPUT_TOKENS = int(os.getenv("DEEPSEEK_MAX_OUTPUT_TOKENS", "3000"))
DEEPSEEK_TEMPERATURE = _env_float("DEEPSEEK_TEMPERATURE", 0.05)
DEEPSEEK_REVIEW_MODEL = os.getenv("DEEPSEEK_REVIEW_MODEL", DEEPSEEK_MODEL)
DEEPSEEK_REVIEW_MAX_OUTPUT_TOKENS = int(os.getenv("DEEPSEEK_REVIEW_MAX_OUTPUT_TOKENS", "6000"))
DEEPSEEK_REVIEW_TEMPERATURE = _env_float("DEEPSEEK_REVIEW_TEMPERATURE", 0.0)
DEEPSEEK_REVIEW_REASONING_EFFORT = os.getenv("DEEPSEEK_REVIEW_REASONING_EFFORT", "max").strip().lower() or "max"

# OpenRouter reviewer — used for the plan-review step instead of DeepSeek Flash.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
OPENROUTER_REVIEWER_MODEL = os.getenv("OPENROUTER_REVIEWER_MODEL", "openai/gpt-4o")
OPENROUTER_REVIEWER_MAX_OUTPUT_TOKENS = int(os.getenv("OPENROUTER_REVIEWER_MAX_OUTPUT_TOKENS", "6000"))
OPENROUTER_REVIEWER_TEMPERATURE = _env_float("OPENROUTER_REVIEWER_TEMPERATURE", 0.0)
OPENROUTER_REVIEWER_TIMEOUT_SECONDS = int(os.getenv("OPENROUTER_REVIEWER_TIMEOUT_SECONDS", "120"))

# The prefilter must also reason through the LONG/SHORT rubric itself before returning the final JSON.
# Format-repair only reformats the output, so reasoning is always disabled to save tokens and avoid empty content.
DEEPSEEK_PREFILTER_REASONING_EFFORT = os.getenv(
    "DEEPSEEK_PREFILTER_REASONING_EFFORT", "max"
).strip().lower() or "max"
FINAL_REVIEW_MIN_SIGNAL_SCORE = int(os.getenv("FINAL_REVIEW_MIN_SIGNAL_SCORE", os.getenv("AUTO_SCAN_MIN_FINAL_SIGNAL_SCORE", "72")))
AUTO_SCAN_DIRECTION_CONFIRMATIONS = max(1, int(os.getenv("AUTO_SCAN_DIRECTION_CONFIRMATIONS", "2")))
ANALYSIS_DATA_VARIANT = os.getenv("ANALYSIS_DATA_VARIANT", "C").strip().upper() or "C"
# The reasoning-summary call uses its own token budget and has NO continuation, to stop the model from burning tokens on hidden reasoning.
LLM_SUMMARY_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_SUMMARY_MAX_OUTPUT_TOKENS", "600"))
# Reasoning is disabled by default for the summary. The main analysis still uses the provider-specific reasoning effort if set.
DB_PATH           = os.getenv("DB_PATH", "bot.db")

# Timeframe roles:
# SCALP: 4H decides direction; 1H designs Entry/SL/TP; 15M is timing only; 1D is macro context.
SHORT_TERM_TIMEFRAMES = {
    "15M": ("15m", 480),   # ~5 days, timing/confirmation only; does not set direction or Entry/SL/TP width
    "1H":  ("1h",  360),   # ~15 days, the timeframe that designs the setup, Entry, SL, TP
    "4H":  ("4h",  360),   # ~60 days, main direction/structure and larger targets
    "1D":  ("1d",  365),   # ~1 year, macro context; avoid scalping clearly against the macro trend
}

# SWING: 1D decides direction; 4H designs Entry/SL/TP; 1H is timing only; 1W is macro/reference for major structure zones.
LONG_TERM_TIMEFRAMES = {
    "1H": ("1h",  480),   # timing/confirmation only; does not set direction or Entry/SL/TP width
    "4H": ("4h",  360),   # the timeframe that designs the setup, Entry, SL, TP1
    "1D": ("1d",  365),   # the main direction/structure decision for SWING
    "1W": ("1w",  208),   # macro context and reference zones for major structure; TP2 is not mandatory
}

# Lifecycle by mode: short = SCALP, long = SWING
# (ENTRY_WAIT_HOURS / TRADE_MAX_HOLD_HOURS are imported from evaluation_store.py above -
# the single source of truth, to avoid the hour mismatch between the two modules that happened before.)

CHECK_INTERVAL_HOURS = {
    "short": 1,       # Scalp: check every 1h
    "long": 12,       # Swing: check every 12h
}

RESULT_CHECK_INTERVAL = {
    "short": "15m",   # Scalp: score the outcome using 15-minute candles
    "long": "1h",     # Swing: score the outcome using 1-hour candles
}


def get_result_check_interval(mode: str) -> str:
    return RESULT_CHECK_INTERVAL.get(mode, "15m")

PREDICTION_HISTORY_COUNT = max(1, min(10, _env_int("PREDICTION_HISTORY_COUNT", 3)))
# /history and the hidden learning logs each keep only the 5 most recent entries per user.
VISIBLE_PREDICTION_RETENTION_LIMIT = 5
HIDDEN_LEARNING_RETENTION_LIMIT = 5
# REJECTED_PLAN/NO_TRADE are no longer saved into predictions after every analysis.
# This variable is kept only to filter legacy data from older DB versions.
HIDDEN_LEARNING_RESULTS = ("REJECTED_PLAN", "NO_TRADE")
TRADE_CANDIDATE_RETENTION_LIMIT = int(os.getenv("TRADE_CANDIDATE_RETENTION_LIMIT", "20"))
TRADE_CANDIDATE_EXPIRE_HOURS = int(os.getenv("TRADE_CANDIDATE_EXPIRE_HOURS", "24"))
VN_TZ = timezone(timedelta(hours=7))


# ─── DB ───────────────────────────────────────────────────────────────────────

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def init_prediction_db() -> None:
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
            ("reviewer_score", "REAL"),
            ("reviewer_verdict", "TEXT"),
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

        # a valid analysis is only saved into the draft/candidate table.
        # Only when the user taps "I traded this plan" does it get copied into predictions for auto-check.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_candidates (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id             INTEGER,
                chat_id             INTEGER,
                symbol              TEXT NOT NULL,
                mode                TEXT NOT NULL,
                created_at          TEXT NOT NULL,
                expires_at          TEXT NOT NULL,
                direction           TEXT NOT NULL,
                entry_low           REAL,
                entry_high          REAL,
                sl                  REAL,
                tp1                 REAL,
                tp2                 REAL,
                market_snapshot     TEXT,
                feature_snapshot    TEXT,
                reasoning_summary   TEXT,
                full_response       TEXT,
                status              TEXT NOT NULL DEFAULT 'DRAFT',
                confirmed_at        TEXT,
                confirmed_prediction_id INTEGER
            )
        """)
        for col, definition in [
            ("expires_at", "TEXT"),
            ("status", "TEXT NOT NULL DEFAULT 'DRAFT'"),
            ("confirmed_at", "TEXT"),
            ("confirmed_prediction_id", "INTEGER"),
        ]:
            try:
                conn.execute(f"ALTER TABLE trade_candidates ADD COLUMN {col} {definition}")
            except sqlite3.OperationalError:
                pass

        # Lightweight index for history/stats/learning/auto-check as the DB grows.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_user_id_id ON predictions(user_id, id DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_user_symbol_mode_id ON predictions(user_id, symbol, mode, id DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_result_next_check ON predictions(result, next_check_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_candidates_user_status_id ON trade_candidates(user_id, status, id DESC)")

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


def prune_trade_candidates(user_id: int | None = None) -> None:
    """Keep the draft table lean. A candidate is just an analysis waiting for user confirmation, not history."""
    now_s = iso(utc_now())
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            DELETE FROM trade_candidates
            WHERE status='DRAFT' AND expires_at < ?
            """,
            (now_s,),
        )
        if user_id is not None:
            conn.execute(
                """
                DELETE FROM trade_candidates
                WHERE user_id=?
                  AND id NOT IN (
                      SELECT id FROM trade_candidates
                      WHERE user_id=?
                      ORDER BY id DESC
                      LIMIT ?
                  )
                """,
                (user_id, user_id, TRADE_CANDIDATE_RETENTION_LIMIT),
            )
        conn.commit()


def save_trade_candidate(
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
) -> int:
    """Save a trackable draft. Does not appear in /history, /stats, or auto-check."""
    now = utc_now()
    expires_at = now + timedelta(hours=TRADE_CANDIDATE_EXPIRE_HOURS)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            INSERT INTO trade_candidates
                (user_id, chat_id, symbol, mode, created_at, expires_at, direction,
                 entry_low, entry_high, sl, tp1, tp2,
                 market_snapshot, feature_snapshot, reasoning_summary, full_response, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'DRAFT')
            """,
            (user_id, chat_id, symbol, mode, iso(now), iso(expires_at), direction,
             entry_low, entry_high, sl, tp1, tp2,
             market_snapshot, feature_snapshot, reasoning_summary, full_response),
        )
        candidate_id = cursor.lastrowid
        conn.commit()
    prune_trade_candidates(user_id)
    return int(candidate_id)


def get_trade_candidate(candidate_id: int, user_id: int | None = None) -> dict | None:
    init_prediction_db()
    clauses = ["id=?"]
    params: list = [candidate_id]
    if user_id is not None:
        clauses.append("user_id=?")
        params.append(user_id)
    where = " AND ".join(clauses)
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            f"""
            SELECT id, user_id, chat_id, symbol, mode, created_at, expires_at, direction,
                   entry_low, entry_high, sl, tp1, tp2,
                   market_snapshot, feature_snapshot, reasoning_summary, full_response,
                   status, confirmed_prediction_id
            FROM trade_candidates
            WHERE {where}
            """,
            params,
        ).fetchone()
    if not row:
        return None
    keys = [
        "id", "user_id", "chat_id", "symbol", "mode", "created_at", "expires_at", "direction",
        "entry_low", "entry_high", "sl", "tp1", "tp2",
        "market_snapshot", "feature_snapshot", "reasoning_summary", "full_response",
        "status", "confirmed_prediction_id",
    ]
    return dict(zip(keys, row))


def _candidate_entry_price(candidate: dict, live_price: float | None = None) -> float | None:
    low = candidate.get("entry_low")
    high = candidate.get("entry_high")
    direction = (candidate.get("direction") or "").upper()
    if low is None or high is None:
        return live_price
    low_f = min(float(low), float(high))
    high_f = max(float(low), float(high))
    if live_price is not None and low_f <= float(live_price) <= high_f:
        return float(live_price)
    # The user tapped "already traded" but the bot doesn't know the real fill price. Use the less favorable edge to score conservatively.
    if direction == "LONG":
        return high_f
    if direction == "SHORT":
        return low_f
    return (low_f + high_f) / 2.0


def confirm_trade_candidate(candidate_id: int, user_id: int | None = None) -> dict:
    """User confirmed they traded per the bot -> copy the candidate into predictions and start auto-check.

    Each confirm button is tied to exactly one trade_candidates.id.
    This function claims the candidate via UPDATE status='CONFIRMING' before save_prediction, to guard against double-click/race conditions.
    So a user tapping multiple times, or Telegram resending the same callback, will not create duplicate predictions.
    """
    init_prediction_db()
    candidate = get_trade_candidate(candidate_id, user_id=user_id)
    if not candidate:
        return {"ok": False, "message": "Không tìm thấy lệnh nháp này, hoặc lệnh không thuộc user hiện tại."}

    status = (candidate.get("status") or "DRAFT").upper()
    if status == "CONFIRMED" and candidate.get("confirmed_prediction_id"):
        return {
            "ok": True,
            "already_confirmed": True,
            "prediction_id": int(candidate["confirmed_prediction_id"]),
            "message": f"Lệnh nháp #{candidate_id} đã được lưu theo dõi trước đó. Mã theo dõi: #{candidate['confirmed_prediction_id']}.",
        }
    if status == "CONFIRMING":
        return {"ok": True, "message": f"Lệnh nháp #{candidate_id} đang được xử lý xác nhận. Vui lòng không bấm lại."}
    if status != "DRAFT":
        return {"ok": False, "message": f"Lệnh nháp #{candidate_id} không còn hiệu lực để lưu. Trạng thái hiện tại: {status}."}

    expires = parse_utc_datetime(candidate.get("expires_at"))
    if expires is not None and utc_now() > expires:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "UPDATE trade_candidates SET status='EXPIRED' WHERE id=? AND status='DRAFT'",
                (candidate_id,),
            )
            conn.commit()
        return {"ok": False, "message": f"Lệnh nháp #{candidate_id} đã quá hạn xác nhận. Hãy phân tích lại để có dữ liệu mới."}

    # Claim atomically before doing network/DB work. This prevents duplicate saves on double-click.
    with sqlite3.connect(DB_PATH) as conn:
        if user_id is None:
            cur = conn.execute(
                "UPDATE trade_candidates SET status='CONFIRMING' WHERE id=? AND status='DRAFT'",
                (candidate_id,),
            )
        else:
            cur = conn.execute(
                "UPDATE trade_candidates SET status='CONFIRMING' WHERE id=? AND user_id=? AND status='DRAFT'",
                (candidate_id, user_id),
            )
        conn.commit()
        claimed = cur.rowcount == 1

    if not claimed:
        latest = get_trade_candidate(candidate_id, user_id=user_id)
        latest_status = (latest or {}).get("status", "UNKNOWN")
        latest_pid = (latest or {}).get("confirmed_prediction_id")
        if str(latest_status).upper() == "CONFIRMED" and latest_pid:
            return {
                "ok": True,
                "already_confirmed": True,
                "prediction_id": int(latest_pid),
                "message": f"Lệnh nháp #{candidate_id} đã được lưu theo dõi trước đó. Mã theo dõi: #{latest_pid}.",
            }
        return {"ok": True, "message": f"Lệnh nháp #{candidate_id} đang được xử lý hoặc đã đổi trạng thái: {latest_status}."}

    try:
        live_price = get_current_price_raw(candidate["symbol"])
        entry_is_live = _price_in_entry_range(
            live_price,
            candidate.get("entry_low"),
            candidate.get("entry_high"),
        )
        entry_price = _candidate_entry_price(candidate, live_price) if entry_is_live else None

        prediction_id = save_prediction(
            symbol=candidate["symbol"],
            mode=candidate["mode"],
            direction=candidate["direction"],
            entry_low=candidate.get("entry_low"),
            entry_high=candidate.get("entry_high"),
            sl=candidate.get("sl"),
            tp1=candidate.get("tp1"),
            tp2=candidate.get("tp2"),
            market_snapshot=candidate.get("market_snapshot"),
            feature_snapshot=candidate.get("feature_snapshot"),
            reasoning_summary=candidate.get("reasoning_summary"),
            full_response=candidate.get("full_response"),
            user_id=candidate.get("user_id"),
            chat_id=candidate.get("chat_id"),
        )

        # the user tapping the button means "I placed the order / chose to track this plan".
        # If the current price isn't inside the Entry zone yet, keep it as PENDING_ENTRY so auto-check waits for a fill.
        # Only mark ENTRY_FILLED immediately if the live price is actually inside the Entry zone at confirmation time.
        if entry_price is not None:
            mark_entry_filled(prediction_id, float(entry_price), utc_now(), candidate["mode"])

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                UPDATE trade_candidates
                SET status='CONFIRMED', confirmed_at=?, confirmed_prediction_id=?
                WHERE id=?
                """,
                (iso(utc_now()), prediction_id, candidate_id),
            )
            conn.commit()

        entry_low = candidate.get("entry_low")
        entry_high = candidate.get("entry_high")
        entry_text = f"{fmt(entry_low)}–{fmt(entry_high)}" if entry_low is not None and entry_high is not None else "N/A"
        if entry_price is not None:
            message = (
                f"Đã lưu lệnh nháp #{candidate_id} thành lệnh theo dõi #{prediction_id}. "
                f"Giá hiện tại đang nằm trong vùng Entry nên bot đánh dấu ENTRY_FILLED tại {fmt(entry_price)}."
            )
        else:
            if live_price is None:
                relation = "Bot chưa lấy được giá hiện tại để kiểm tra khớp Entry."
            else:
                low_f, high_f = _range_low_high(entry_low, entry_high)
                if low_f is not None and high_f is not None:
                    if float(live_price) < low_f:
                        relation = f"Giá hiện tại {fmt(live_price)} còn thấp hơn vùng Entry {entry_text}."
                    elif float(live_price) > high_f:
                        relation = f"Giá hiện tại {fmt(live_price)} còn cao hơn vùng Entry {entry_text}."
                    else:
                        relation = f"Giá hiện tại {fmt(live_price)} đang ở gần vùng Entry {entry_text}."
                else:
                    relation = f"Giá hiện tại {fmt(live_price)}; vùng Entry không đủ dữ liệu."
            message = (
                f"Đã lưu lệnh nháp #{candidate_id} thành lệnh chờ #{prediction_id}. "
                f"Entry chưa khớp. {relation} Bot sẽ theo dõi đến khi giá chạm vùng Entry rồi mới tính WIN/LOSS."
            )

        return {
            "ok": True,
            "prediction_id": int(prediction_id),
            "entry_price": entry_price,
            "entry_status": "ENTRY_FILLED" if entry_price is not None else "PENDING_ENTRY",
            "message": message,
        }
    except Exception as exc:
        # If an error happens after claiming, reopen it as DRAFT so the user can retry once the transient error clears.
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "UPDATE trade_candidates SET status='DRAFT' WHERE id=? AND status='CONFIRMING'",
                (candidate_id,),
            )
            conn.commit()
        return {"ok": False, "message": f"Lưu lệnh nháp #{candidate_id} thất bại: {exc}"}


def discard_trade_candidate(candidate_id: int, user_id: int | None = None) -> dict:
    init_prediction_db()
    candidate = get_trade_candidate(candidate_id, user_id=user_id)
    if not candidate:
        return {"ok": False, "message": "Không tìm thấy lệnh nháp này, hoặc lệnh không thuộc user hiện tại."}
    if (candidate.get("status") or "").upper() != "DRAFT":
        return {"ok": True, "message": "Lệnh này không còn là nháp, không cần bỏ qua."}
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE trade_candidates SET status='DISCARDED' WHERE id=?", (candidate_id,))
        conn.commit()
    return {"ok": True, "message": "Đã bỏ qua lệnh này. Bot sẽ không lưu vào history và không theo dõi kết quả."}

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
) -> int:
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
                 entry_status, market_snapshot, feature_snapshot, reasoning_summary, full_response, result)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING_ENTRY', ?, ?, ?, ?, 'PENDING_ENTRY')
            """,
            (user_id, chat_id, symbol, mode, iso(now), entry_wait, entry_wait, max_hold,
             iso(next_check), direction, entry_low, entry_high, sl, tp1, tp2,
             market_snapshot, feature_snapshot, reasoning_summary, full_response),
        )
        prediction_id = cursor.lastrowid
        conn.commit()
    prune_prediction_history(user_id)
    return prediction_id




def save_no_trade_prediction(
    symbol: str,
    mode: str,
    market_snapshot: str | None,
    feature_snapshot: str | None,
    reasoning_summary: str | None,
    full_response: str | None,
    user_id: int | None = None,
    chat_id: int | None = None,
) -> int:
    """
    Save a NO_TRADE decision so the model can learn when it should stay out.

    This record is NOT auto-checked and NOT shown in /history, /stats, /dashboard.
    It's only used in the per-user learning context for future analyses.
    """
    now = utc_now()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions
                (user_id, chat_id, symbol, mode, created_at, check_after_hours, entry_wait_hours, max_hold_hours,
                 next_check_at, direction, entry_low, entry_high, sl, tp1, tp2,
                 entry_status, market_snapshot, feature_snapshot, reasoning_summary, full_response,
                 result, result_reason, result_checked_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, 'NO_TRADE', NULL, NULL, NULL, NULL, NULL,
                    'NO_TRADE', ?, ?, ?, ?, 'NO_TRADE', ?, ?)
            """,
            (user_id, chat_id, symbol, mode, iso(now),
             ENTRY_WAIT_HOURS.get(mode, 24), ENTRY_WAIT_HOURS.get(mode, 24), TRADE_MAX_HOLD_HOURS.get(mode, 72),
             market_snapshot, feature_snapshot, reasoning_summary or "Claude chọn NO TRADE.", full_response,
             "Claude chọn NO TRADE vì chưa có setup đủ rõ hoặc tỷ lệ lời/lỗ chưa đáng để tạo tín hiệu.", iso(now)),
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


def get_recent_predictions(
    symbol: str,
    mode: str,
    user_id: int | None = None,
    limit: int = PREDICTION_HISTORY_COUNT,
) -> list[dict]:
    """
    Get history to feed back to the model for learning.

    Privacy / per-user learning rules:
    - When analyzing for a given user, the AI only receives the trade history that same user has confirmed.
    - Another user's global history is never used, to avoid strategy noise and to avoid leaking data.
    - If user_id=None (e.g. legacy/manual calls), no learning history is included.
    """
    if user_id is None:
        return []

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT created_at, direction, entry_low, entry_high, sl, tp1, tp2,
                   reasoning_summary, full_response, result, result_price, result_reason,
                   market_snapshot, feature_snapshot
            FROM predictions
            WHERE symbol=? AND mode=? AND user_id=?
              AND result NOT IN ('REJECTED_PLAN', 'NO_TRADE')
            ORDER BY id DESC
            LIMIT ?
            """,
            (symbol, mode, user_id, limit),
        ).fetchall()

    return [
        {
            "created_at":        row[0],
            "direction":         row[1],
            "entry_low":         row[2],
            "entry_high":        row[3],
            "sl":                row[4],
            "tp1":               row[5],
            "tp2":               row[6],
            "reasoning_summary": row[7],
            "full_response":     row[8],
            "result":            row[9],
            "result_price":      row[10],
            "result_reason":     row[11],
            "market_snapshot":   row[12],
            "feature_snapshot":  row[13],
        }
        for row in rows
    ]


def get_open_signal_predictions(
    symbol: str,
    mode: str,
    user_id: int | None = None,
    limit: int = 2,
) -> list[dict]:
    """Get open plans so the model doesn't mistake a pending order for an opposite signal.

    Only fetched for the exact user + symbol + mode, to avoid leaking other users' data and to avoid
    bloating the prompt. Used for awareness when a user re-analyzes the same coin/mode while an
    older signal is still PENDING_ENTRY or ENTRY_FILLED.
    """
    if user_id is None:
        return []

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, direction, entry_low, entry_high, sl, tp1, tp2,
                   result, entry_status, entry_filled_at, entry_price, result_reason
            FROM predictions
            WHERE symbol=? AND mode=? AND user_id=?
              AND result IN ('PENDING_ENTRY', 'ENTRY_FILLED')
            ORDER BY id DESC
            LIMIT ?
            """,
            (symbol, mode, user_id, limit),
        ).fetchall()

    return [
        {
            "id": row[0],
            "created_at": row[1],
            "direction": row[2],
            "entry_low": row[3],
            "entry_high": row[4],
            "sl": row[5],
            "tp1": row[6],
            "tp2": row[7],
            "result": row[8],
            "entry_status": row[9],
            "entry_filled_at": row[10],
            "entry_price": row[11],
            "result_reason": row[12],
        }
        for row in rows
    ]


def _price_vs_entry_text(current_price: float | None, entry_low: float | None, entry_high: float | None) -> str:
    if current_price is None or entry_low is None or entry_high is None:
        return "không đủ dữ liệu để so với giá hiện tại"
    low = min(float(entry_low), float(entry_high))
    high = max(float(entry_low), float(entry_high))
    if low <= current_price <= high:
        return "giá hiện tại đang nằm trong vùng Entry cũ"
    if current_price < low:
        return "giá hiện tại đang thấp hơn vùng Entry cũ"
    return "giá hiện tại đang cao hơn vùng Entry cũ"


def format_open_signal_context(open_signals: list[dict], current_price: float | None) -> str:
    """Build a short awareness block for currently open plans.

    Main goal: avoid a situation where the model gives a LONG waiting for a pullback, the user
    re-analyzes, and the model then chases price, or the user mistakes the LONG Entry for a SHORT's TP.
    """
    if not open_signals:
        return "KẾ HOẠCH ĐANG MỞ: Không có kế hoạch đang chờ/đã khớp cho user này ở cùng coin và mode."

    lines = ["KẾ HOẠCH ĐANG MỞ CÙNG USER/COIN/MODE (CHỈ LÀ TRẠNG THÁI VẬN HÀNH, KHÔNG PHẢI BẰNG CHỨNG HƯỚNG):"]
    for p in open_signals:
        entry = f"{fmt(p.get('entry_low'))}-{fmt(p.get('entry_high'))}" if p.get("entry_low") is not None and p.get("entry_high") is not None else "N/A"
        status = p.get("result") or p.get("entry_status") or "N/A"
        relation = _price_vs_entry_text(current_price, p.get("entry_low"), p.get("entry_high"))
        extra = ""
        if status == "ENTRY_FILLED":
            extra = f"; đã khớp lúc {str(p.get('entry_filled_at') or '')[:16]}, giá khớp {fmt(p.get('entry_price'))}"
        elif status == "PENDING_ENTRY":
            extra = "; chưa khớp Entry"
        lines.append(
            f"- #{p.get('id')} {str(p.get('created_at') or '')[:16]} {p.get('direction')} {status}{extra}; "
            f"Entry {entry}, SL {fmt(p.get('sl'))}, TP1 {fmt(p.get('tp1'))}, TP2 {fmt(p.get('tp2'))}; {relation}."
        )

    lines.extend([
        "Cách dùng kế hoạch đang mở:",
        "- Nếu kế hoạch cũ là LONG chờ hồi, vùng Entry LONG KHÔNG phải TP cho lệnh SHORT ngược lại. Nếu kế hoạch cũ là SHORT chờ hồi, vùng Entry SHORT KHÔNG phải TP cho lệnh LONG ngược lại.",
        "- Hướng và mức giá cũ không được dùng làm bằng chứng. Chỉ giữ, hủy hoặc thay kế hoạch sau khi dữ liệu hiện tại tự xác nhận độc lập.",
        "- Entry mới gần giá hiện tại vẫn hợp lệ nếu nằm trong luận điểm cấu trúc hiện tại và có điểm vô hiệu rõ. Chỉ coi là đuổi giá khi giá đã rời vùng luận điểm và không còn đặt được SL hợp lý.",
    ])
    return "\n".join(lines)


# ─── Auto WIN/LOSS check ──────────────────────────────────────────────────────

def get_current_price_raw(symbol: str) -> float | None:
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": symbol},
            timeout=30,
        )
        r.raise_for_status()
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
    try:
        r = requests.get(
            BINANCE_API_URL,
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": int(start.timestamp() * 1000),
                "limit": limit,
            },
            timeout=120,
        )
        r.raise_for_status()
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
        pending_candles = candles[candles["close_time"] > pd.Timestamp(created)]
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
        normalized_symbol = symbol.upper() if symbol.upper().endswith("USDT") else f"{symbol.upper()}USDT"
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
    init_prediction_db()
    with sqlite3.connect(DB_PATH) as conn:
        visible_count = int(conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE result NOT IN ('REJECTED_PLAN', 'NO_TRADE')"
        ).fetchone()[0])
        total_prediction_count = int(conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
        try:
            draft_count = int(conn.execute("SELECT COUNT(*) FROM trade_candidates").fetchone()[0])
        except sqlite3.Error:
            draft_count = 0
        conn.execute("DELETE FROM predictions")
        try:
            conn.execute("DELETE FROM trade_candidates")
        except sqlite3.Error:
            pass
        try:
            conn.execute("DELETE FROM sqlite_sequence WHERE name='predictions'")
            conn.execute("DELETE FROM sqlite_sequence WHERE name='trade_candidates'")
        except sqlite3.Error:
            pass
        conn.commit()
    return {
        "visible_count": visible_count,
        "total_prediction_count": total_prediction_count,
        "draft_count": draft_count,
    }


def clear_trade_candidates(user_id: int | None = None) -> dict:
    """Delete only the draft/candidate table; does not touch predictions/history.

    - user_id != None: deletes all of that user's candidates.
    - user_id == None: admin deletes all candidates for every user.

    Note: a candidate is only a draft/confirmation layer. A confirmed trade has already been fully
    copied into the predictions table, so deleting candidates doesn't affect /history or auto-check.
    """
    init_prediction_db()
    with sqlite3.connect(DB_PATH) as conn:
        params: tuple = ()
        where = ""
        if user_id is not None:
            where = " WHERE user_id=?"
            params = (user_id,)

        def count_status(status: str) -> int:
            return int(conn.execute(
                f"SELECT COUNT(*) FROM trade_candidates{where}{' AND' if where else ' WHERE'} status=?",
                (*params, status),
            ).fetchone()[0])

        total = int(conn.execute(
            f"SELECT COUNT(*) FROM trade_candidates{where}",
            params,
        ).fetchone()[0])
        draft_count = count_status('DRAFT')
        expired_count = count_status('EXPIRED')
        discarded_count = count_status('DISCARDED')
        confirming_count = count_status('CONFIRMING')
        confirmed_count = count_status('CONFIRMED')

        conn.execute(f"DELETE FROM trade_candidates{where}", params)

        remaining = int(conn.execute("SELECT COUNT(*) FROM trade_candidates").fetchone()[0])
        if remaining == 0:
            try:
                conn.execute("DELETE FROM sqlite_sequence WHERE name='trade_candidates'")
            except sqlite3.Error:
                pass
        conn.commit()

    return {
        "deleted_count": total,
        "draft_count": draft_count,
        "expired_count": expired_count,
        "discarded_count": discarded_count,
        "confirming_count": confirming_count,
        "confirmed_count": confirmed_count,
        "history_untouched": True,
        "sequence_reset": remaining == 0,
        "scope": "all" if user_id is None else "user",
    }


# ─── Binance + Indicators ─────────────────────────────────────────────────────

def get_binance_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame | None:
    try:
        r = requests.get(
            BINANCE_API_URL,
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=120,
        )
        r.raise_for_status()
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
    return tr.rolling(period, min_periods=period).mean()


def add_indicators(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or len(df) < 60:
        return None
    r = df.copy()
    r["ema_7"],  r["ema_25"], r["ema_50"] = (
        calculate_ema(r["close"], 7),
        calculate_ema(r["close"], 25),
        calculate_ema(r["close"], 50),
    )
    r["rsi_6"],  r["rsi_14"] = calculate_rsi(r["close"], 6), calculate_rsi(r["close"], 14)
    r["macd_line"], r["macd_signal"], r["macd_hist"] = calculate_macd(r["close"])
    r["atr_14"] = calculate_atr(r, 14)
    r["atr_pct"] = (r["atr_14"] / r["close"]) * 100
    r["vol_ma20"]  = r["volume"].rolling(20).mean()
    r["vol_ratio"] = r["volume"] / r["vol_ma20"]
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


def _current_atr(df: pd.DataFrame | None) -> float | None:
    if df is None or df.empty or "atr_14" not in df.columns:
        return None
    row = _analysis_row(df) if "_analysis_row" in globals() else (df.iloc[-2] if len(df) >= 2 else df.iloc[-1])
    return _safe_float(row.get("atr_14"))


def _window_tail(df: pd.DataFrame | None, hours: int | None = None, max_candles: int | None = None) -> pd.DataFrame | None:
    """Get data over a time window instead of a fixed number of candles.

    Coinglass uses 12h/24h/48h as a *lookback window*, not necessarily meaning 12H/24H candles
    must be used. Teopard still uses smaller candles to keep resolution, but only considers
    candles that fall inside that time window.
    """
    if df is None or df.empty:
        return None
    data = df.copy()
    if hours is not None:
        time_col = "close_time" if "close_time" in data.columns else "timestamp"
        ref_time = data[time_col].max()
        start_time = ref_time - pd.Timedelta(hours=hours)
        data = data[data[time_col] >= start_time]
    if max_candles is not None and len(data) > max_candles:
        data = data.tail(max_candles)
    return data.reset_index(drop=True)


def _find_pivots(df: pd.DataFrame | None, side: str, lookback: int | None = 100, left: int = 2, right: int = 2) -> list[dict]:
    if df is None or df.empty:
        return []
    data = df.tail(lookback).reset_index(drop=True) if lookback else df.reset_index(drop=True)
    col = "high" if side == "high" else "low"
    pivots: list[dict] = []
    if len(data) < left + right + 1:
        return pivots
    for i in range(left, len(data) - right):
        val = float(data.loc[i, col])
        window = data.loc[i - left:i + right, col]
        if side == "high" and val >= float(window.max()):
            pivots.append({"price": val, "time": data.loc[i, "timestamp"], "index": i, "kind": "pivot", "weight": 1.0})
        elif side == "low" and val <= float(window.min()):
            pivots.append({"price": val, "time": data.loc[i, "timestamp"], "index": i, "kind": "pivot", "weight": 1.0})
    return pivots


def _cluster_zone(prices: list[float], current_price: float, side: str, atr: float | None) -> tuple[float | None, float | None, int]:
    """Legacy helper kept for a few old fallback paths if needed."""
    if not prices:
        return None, None, 0
    tol = max((atr or 0) * 0.25, current_price * 0.0012)
    buf = max((atr or 0) * 0.20, current_price * 0.0008)
    sorted_prices = sorted(prices)
    clusters: list[list[float]] = []
    cur = [sorted_prices[0]]
    for price in sorted_prices[1:]:
        if abs(price - sum(cur) / len(cur)) <= tol:
            cur.append(price)
        else:
            clusters.append(cur)
            cur = [price]
    clusters.append(cur)

    if side == "low":
        candidates = [c for c in clusters if sum(c) / len(c) <= current_price]
    else:
        candidates = [c for c in clusters if sum(c) / len(c) >= current_price]

    if not candidates:
        return None, None, 0
    candidates.sort(key=lambda c: (len(c), -abs(current_price - sum(c) / len(c))), reverse=True)
    best = candidates[0]
    low = min(best) - buf
    high = max(best) + buf
    return low, high, len(best)


def _candle_wick_stats(row) -> tuple[float, float, float, float]:
    open_ = float(row["open"])
    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])
    rng = max(high - low, 1e-12)
    upper = max(high - max(open_, close), 0.0) / rng
    lower = max(min(open_, close) - low, 0.0) / rng
    body = abs(close - open_) / rng
    return upper, lower, body, rng


def _collect_liquidity_points(
    window_df: pd.DataFrame | None,
    side: str,
    current_price: float,
    atr: float | None,
    role: str = "main",
) -> list[dict]:
    """Collect estimated liquidity points from pivots, equal highs/lows, and wick-sweep candles.

    The goal is to pick the most trade-relevant zone, not to force near/main/deep to be
    spread apart. If the same cluster gets touched repeatedly across multiple windows, that
    zone may reappear, but touch/sweep/volume stats will be attached so the AI understands
    the zone's true quality.
    """
    if window_df is None or window_df.empty:
        return []

    data = window_df.reset_index(drop=True)
    left_right = 1 if len(data) < 12 else 2
    points = _find_pivots(data, side, lookback=None, left=left_right, right=left_right)
    tol = _liquidity_tolerance(current_price, atr, role)

    col = "high" if side == "high" else "low"

    # Add wick-rejection/sweep points. These are usually where stops/liquidity get swept.
    for i, row in data.iterrows():
        upper_wick, lower_wick, body_pct, _rng = _candle_wick_stats(row)
        price = float(row[col])
        if side == "high":
            is_sweep_like = upper_wick >= 0.32 and upper_wick >= body_pct * 0.8
        else:
            is_sweep_like = lower_wick >= 0.32 and lower_wick >= body_pct * 0.8
        if is_sweep_like:
            points.append({
                "price": price,
                "time": row.get("timestamp"),
                "index": int(i),
                "kind": "wick_sweep",
                "weight": 1.25,
            })

    # Add equal-high/equal-low clusters: two touches close together within the tolerance.
    # Don't add every candle, to avoid turning this into a fake volume profile.
    values = [float(v) for v in data[col].tail(min(len(data), 80)).tolist()]
    offset = len(data) - len(values)
    for i in range(1, len(values)):
        prev = values[i - 1]
        cur = values[i]
        if abs(cur - prev) <= tol:
            row = data.iloc[offset + i]
            points.append({
                "price": cur,
                "time": row.get("timestamp"),
                "index": int(offset + i),
                "kind": "equal_touch",
                "weight": 0.85,
            })

    # Always add the window's extremes so an important high/low isn't missed when there are no pivots.
    if not data.empty:
        if side == "high":
            idx = int(data["high"].idxmax())
            price = float(data.loc[idx, "high"])
        else:
            idx = int(data["low"].idxmin())
            price = float(data.loc[idx, "low"])
        points.append({
            "price": price,
            "time": data.loc[idx, "timestamp"],
            "index": idx,
            "kind": "window_extreme",
            "weight": 0.9,
        })

    return points


def _zone_side_state(zone: tuple | None, current_price: float | None) -> str:
    if not zone or current_price is None:
        return "unknown"
    low = zone[0] if len(zone) > 0 else None
    high = zone[1] if len(zone) > 1 else None
    if low is None or high is None:
        return "unknown"
    if float(low) <= current_price <= float(high):
        return "touching"
    if float(high) < current_price:
        return "below"
    if float(low) > current_price:
        return "above"
    return "overlap"


def _zone_meta_default(role: str = "main") -> dict:
    return {"role": role, "touches": 0, "sweeps": 0, "vol_ratio": None, "score": 0.0}


def _cluster_zone_from_pivots(
    pivots: list[dict],
    current_price: float,
    side: str,
    atr: float | None,
    window_df: pd.DataFrame | None,
    role: str = "main",
) -> tuple[float | None, float | None, int, dict]:
    """Group liquidity points into zones and pick the highest-quality zone.

    The score favors a zone with more touches, wick sweeps, good volume, recency, and a
    distance that fits its role. Zones are not forced apart; if the market is genuinely
    trading around the same liquidity cluster, the near/main/deep zones may sit close
    together, but the metadata will clearly flag that they're touching price / overlapping roles.
    """
    if not pivots:
        return None, None, 0, _zone_meta_default(role)

    tol = _liquidity_tolerance(current_price, atr, role)
    buf = _liquidity_buffer(current_price, atr, role)
    sorted_pivots = sorted(pivots, key=lambda p: float(p["price"]))

    clusters: list[list[dict]] = []
    cur = [sorted_pivots[0]]
    for pivot in sorted_pivots[1:]:
        center = sum(float(p["price"]) for p in cur) / len(cur)
        if abs(float(pivot["price"]) - center) <= tol:
            cur.append(pivot)
        else:
            clusters.append(cur)
            cur = [pivot]
    clusters.append(cur)

    if side == "low":
        candidates = [c for c in clusters if (sum(float(p["price"]) for p in c) / len(c)) <= current_price]
    else:
        candidates = [c for c in clusters if (sum(float(p["price"]) for p in c) / len(c)) >= current_price]

    if not candidates:
        return None, None, 0, _zone_meta_default(role)

    data = window_df.reset_index(drop=True) if window_df is not None and not window_df.empty else None
    ref_time = data["timestamp"].max() if data is not None and "timestamp" in data.columns else None
    ref_atr = _liquidity_ref_atr(current_price, atr)

    def cluster_stats(cluster: list[dict]) -> dict:
        prices = [float(p["price"]) for p in cluster]
        raw_low, raw_high = min(prices), max(prices)
        low, high = raw_low - buf, raw_high + buf
        center = sum(prices) / len(prices)
        touch_count = 0
        sweep_count = 0
        vol_values: list[float] = []
        recent_touch_age_hours = None

        if data is not None:
            for _, row in data.iterrows():
                high_v = float(row["high"])
                low_v = float(row["low"])
                close_v = float(row["close"])
                open_v = float(row["open"])
                upper_wick, lower_wick, body_pct, _rng = _candle_wick_stats(row)
                price_v = high_v if side == "high" else low_v
                touched = (low - tol) <= price_v <= (high + tol)
                if touched:
                    touch_count += 1
                    vol_ratio = _safe_float(row.get("vol_ratio"))
                    if vol_ratio is not None and np.isfinite(vol_ratio):
                        vol_values.append(float(vol_ratio))
                    if ref_time is not None:
                        try:
                            age = max((pd.Timestamp(ref_time) - pd.Timestamp(row["timestamp"])).total_seconds() / 3600.0, 0.0)
                            recent_touch_age_hours = age if recent_touch_age_hours is None else min(recent_touch_age_hours, age)
                        except Exception:
                            pass

                if side == "high":
                    # Sweep up: pokes through the high/liquidity zone then closes back down with a clear upper wick.
                    swept = high_v >= low and close_v < center and upper_wick >= 0.25 and high_v > max(open_v, close_v)
                else:
                    # Sweep down: pokes below the low/liquidity zone then closes back up with a clear lower wick.
                    swept = low_v <= high and close_v > center and lower_wick >= 0.25 and low_v < min(open_v, close_v)
                if swept:
                    sweep_count += 1

        if touch_count == 0:
            touch_count = len(cluster)

        avg_vol = float(np.mean(vol_values)) if vol_values else None
        point_weight = sum(float(p.get("weight", 1.0)) for p in cluster)
        pivot_hits = sum(1 for p in cluster if p.get("kind") == "pivot")
        equal_hits = sum(1 for p in cluster if p.get("kind") == "equal_touch")
        wick_hits = sum(1 for p in cluster if p.get("kind") == "wick_sweep")

        distance_atr = abs(center - current_price) / max(ref_atr, 1e-12)
        if role == "near":
            distance_score = max(0.0, 1.0 - distance_atr / 4.0) * 1.8
        elif role == "deep":
            # "Deep" isn't forced to be far away, but zones too close to price aren't rewarded too heavily either.
            distance_score = max(0.0, min(distance_atr / 3.0, 1.0)) * 0.8
        else:
            distance_score = max(0.0, 1.0 - abs(distance_atr - 1.6) / 5.0) * 1.1

        recency_score = 0.0
        if recent_touch_age_hours is not None:
            recency_score = 1.2 / (1.0 + recent_touch_age_hours / 18.0)
        elif ref_time is not None:
            try:
                last_touch = max(pd.Timestamp(p["time"]) for p in cluster if p.get("time") is not None)
                age_hours = max((pd.Timestamp(ref_time) - last_touch).total_seconds() / 3600.0, 0.0)
                recency_score = 0.9 / (1.0 + age_hours / 24.0)
            except Exception:
                recency_score = 0.0

        vol_score = 0.0
        if avg_vol is not None:
            # High volume is good, but a single anomalous volume spike shouldn't dominate everything.
            vol_score = min(max(avg_vol - 0.8, 0.0), 1.8) * 0.8

        score = (
            min(point_weight, 8.0) * 1.1
            + min(touch_count, 8) * 0.9
            + min(sweep_count, 5) * 1.25
            + min(pivot_hits, 5) * 0.35
            + min(equal_hits, 5) * 0.25
            + min(wick_hits, 5) * 0.35
            + vol_score
            + recency_score
            + distance_score
        )

        return {
            "low": low,
            "high": high,
            "center": center,
            "hits": max(len(cluster), touch_count),
            "touches": touch_count,
            "sweeps": sweep_count,
            "vol_ratio": avg_vol,
            "score": score,
            "distance_atr": distance_atr,
            "pivot_hits": pivot_hits,
            "equal_hits": equal_hits,
            "wick_hits": wick_hits,
            "role": role,
        }

    scored = [cluster_stats(c) for c in candidates]
    best = max(scored, key=lambda m: m["score"])
    meta = {
        "role": role,
        "touches": int(best["touches"]),
        "sweeps": int(best["sweeps"]),
        "vol_ratio": best["vol_ratio"],
        "score": round(float(best["score"]), 2),
        "distance_atr": round(float(best["distance_atr"]), 2),
        "pivot_hits": int(best["pivot_hits"]),
        "equal_hits": int(best["equal_hits"]),
        "wick_hits": int(best["wick_hits"]),
    }
    return best["low"], best["high"], int(best["hits"]), meta


def _fallback_zone(
    df: pd.DataFrame | None,
    side: str,
    current_price: float,
    atr: float | None,
    window: int | None = None,
    role: str = "main",
) -> tuple[float | None, float | None, dict]:
    if df is None or df.empty:
        return None, None, _zone_meta_default(role)
    data = df.tail(window) if window else df
    if data.empty:
        return None, None, _zone_meta_default(role)
    buf = _liquidity_buffer(current_price, atr, role)
    if side == "low":
        idx = data["low"].idxmin()
        price = float(data.loc[idx, "low"])
    else:
        idx = data["high"].idxmax()
        price = float(data.loc[idx, "high"])

    low, high = price - buf, price + buf
    # If every fallback extreme ends up on the wrong side of the current price, report N/A.
    # Example: if price just broke above the 48h high, don't use an old high below price as the "upper zone".
    if side == "high" and high < current_price:
        return None, None, _zone_meta_default(role)
    if side == "low" and low > current_price:
        return None, None, _zone_meta_default(role)

    meta = _zone_meta_default(role)
    meta.update({"touches": 1, "score": 1.0, "fallback": True})
    return low, high, meta


# ─── Liquidity: fractal swing pools, not broad support/resistance bands ──────
# The goal is to estimate stop/liquidation pools from OHLCV:
# - Use swing/fractal highs-lows as liquidity levels.
# - The box sits OUTSIDE the swing: below the low for long liquidations, above the high for short liquidations.
# - M15/lower timeframes are only used to flag that a sweep occurred, not to build a wide box around price.
# - Don't merge levels that are far apart into one wide "liquidity cluster".

def _liq_role_params(role: str, mode: str = "short") -> dict:
    if mode == "short":
        params = {
            "near": {"tol_pct": 0.00028, "box_pct": 0.00075, "min_box_pct": 0.00028, "max_box_pct": 0.00105, "atr_mult": 0.12, "target_atr": 0.8},
            "main": {"tol_pct": 0.00036, "box_pct": 0.00105, "min_box_pct": 0.00035, "max_box_pct": 0.00145, "atr_mult": 0.16, "target_atr": 1.6},
            "deep": {"tol_pct": 0.00045, "box_pct": 0.00135, "min_box_pct": 0.00045, "max_box_pct": 0.00190, "atr_mult": 0.20, "target_atr": 2.6},
        }
    else:
        # SWING uses H4/D1/W1 so the box is allowed to be wider than scalp, but it's still a stop-pool
        # outside the swing high/low, not a sideways band around the current price.
        # near H4: the zone around the nearby swing for timing Entry; main D1: the main TP/SL zone; deep W1/D1: the far zone.
        params = {
            "near": {"tol_pct": 0.00055, "box_pct": 0.00180, "min_box_pct": 0.00070, "max_box_pct": 0.00320, "atr_mult": 0.16, "target_atr": 1.2},
            "main": {"tol_pct": 0.00085, "box_pct": 0.00350, "min_box_pct": 0.00110, "max_box_pct": 0.00650, "atr_mult": 0.22, "target_atr": 2.5},
            "deep": {"tol_pct": 0.00120, "box_pct": 0.00550, "min_box_pct": 0.00160, "max_box_pct": 0.01000, "atr_mult": 0.28, "target_atr": 4.0},
        }
    return params.get(role, params["main"])


def _liquidity_ref_atr(current_price: float, atr: float | None) -> float:
    # Lower fallback than before, so scalp doesn't build an overly wide zone when ATR is empty/large.
    return max(float(atr or 0), current_price * 0.0012)


def _liquidity_tolerance(current_price: float, atr: float | None, role: str = "main", mode: str = "short") -> float:
    params = _liq_role_params(role, mode)
    ref_atr = _liquidity_ref_atr(current_price, atr)
    tol = max(current_price * params["tol_pct"], ref_atr * 0.055)
    # This is the tolerance for detecting equal highs/lows, not the width of the zone.
    return min(tol, current_price * params["max_box_pct"] * 0.45)


def _liquidity_buffer(current_price: float, atr: float | None, role: str = "main", mode: str = "short") -> float:
    # Wrapper kept for fallback/legacy; the main flow now uses _liq_box_width.
    return _liq_box_width(current_price, atr, role, mode) * 0.50


def _liq_box_width(current_price: float, atr: float | None, role: str, mode: str = "short") -> float:
    params = _liq_role_params(role, mode)
    ref_atr = _liquidity_ref_atr(current_price, atr)
    raw = max(current_price * params["box_pct"], ref_atr * params["atr_mult"])
    return min(max(raw, current_price * params["min_box_pct"]), current_price * params["max_box_pct"])


def _fractal_swing_points(
    df: pd.DataFrame | None,
    side: str,
    lookback: int | None = None,
    m: int = 2,
) -> list[dict]:
    if df is None or df.empty:
        return []
    data = df.tail(lookback).reset_index(drop=True) if lookback else df.reset_index(drop=True)
    if len(data) < m * 2 + 1:
        return []
    col = "high" if side == "high" else "low"
    points: list[dict] = []
    for i in range(m, len(data) - m):
        price = float(data.loc[i, col])
        left = data.loc[i - m:i - 1, col]
        right = data.loc[i + 1:i + m, col]
        if side == "high":
            is_swing = price > float(left.max()) and price >= float(right.max())
        else:
            is_swing = price < float(left.min()) and price <= float(right.min())
        if not is_swing:
            continue
        vol_ratio = _safe_float(data.loc[i].get("vol_ratio"), 1.0) or 1.0
        points.append({
            "price": price,
            "index": int(i),
            "time": data.loc[i].get("timestamp"),
            "volume_ratio": float(vol_ratio) if np.isfinite(vol_ratio) else 1.0,
            "kind": "fractal",
        })
    return points


def _equal_touch_score(data: pd.DataFrame | None, side: str, level: float, tol: float) -> int:
    if data is None or data.empty:
        return 0
    col = "high" if side == "high" else "low"
    vals = data[col].astype(float)
    return int(((vals - level).abs() <= tol).sum())


def _sweep_stats_against_level(
    sweep_df: pd.DataFrame | None,
    side: str,
    level: float,
    tol: float,
) -> tuple[int, float | None]:
    if sweep_df is None or sweep_df.empty:
        return 0, None
    sweeps = 0
    vols: list[float] = []
    data = sweep_df.tail(120).reset_index(drop=True)
    for _, row in data.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        upper_wick, lower_wick, body_pct, _rng = _candle_wick_stats(row)
        vol_ratio = _safe_float(row.get("vol_ratio"), 1.0) or 1.0
        if side == "high":
            # Short-liq sweep: pokes above the swing high then closes back below the level.
            swept = high >= level + tol * 0.25 and close < level and upper_wick >= 0.28 and upper_wick >= body_pct * 0.7
        else:
            # Long-liq sweep: pokes below the swing low then closes back above the level.
            swept = low <= level - tol * 0.25 and close > level and lower_wick >= 0.28 and lower_wick >= body_pct * 0.7
        if swept:
            sweeps += 1
            if np.isfinite(vol_ratio):
                vols.append(float(vol_ratio))
    return sweeps, (float(np.mean(vols)) if vols else None)


def _cluster_liq_levels(points: list[dict], current_price: float, atr: float | None, role: str, mode: str) -> list[list[dict]]:
    if not points:
        return []
    tol = _liquidity_tolerance(current_price, atr, role, mode)
    clusters: list[list[dict]] = []
    for point in sorted(points, key=lambda p: float(p["price"])):
        if not clusters:
            clusters.append([point])
            continue
        cur = clusters[-1]
        center = sum(float(p["price"]) for p in cur) / len(cur)
        if abs(float(point["price"]) - center) <= tol:
            cur.append(point)
        else:
            clusters.append([point])
    return clusters


def _liq_zone_from_level(level_low: float, level_high: float, side: str, width: float) -> tuple[float, float]:
    # The liquidity zone sits OUTSIDE the level, not wrapped around the current price like support/resistance.
    if side == "low":
        top = level_high
        return top - width, top
    bottom = level_low
    return bottom, bottom + width


def _score_liq_cluster(
    cluster: list[dict],
    data: pd.DataFrame | None,
    sweep_df: pd.DataFrame | None,
    current_price: float,
    atr: float | None,
    side: str,
    role: str,
    mode: str,
) -> dict:
    prices = [float(p["price"]) for p in cluster]
    level_low, level_high = min(prices), max(prices)
    level = sum(prices) / len(prices)
    tol = _liquidity_tolerance(current_price, atr, role, mode)
    width = _liq_box_width(current_price, atr, role, mode)

    # If a cluster is abnormally wide, don't turn the whole cluster into one wide zone.
    # Only take the outer edge closest to the stop-pool, to avoid an output like 62,620-62,820.
    max_level_span = max(tol * 1.65, current_price * _liq_role_params(role, mode)["max_box_pct"] * 0.35)
    if (level_high - level_low) > max_level_span:
        if side == "low":
            level_low = level_high = max(prices)  # the highest low closer to price is the nearest stop-pool below
        else:
            level_low = level_high = min(prices)  # the lowest high closer to price is the nearest stop-pool above
        level = level_low

    zone_low, zone_high = _liq_zone_from_level(level_low, level_high, side, width)
    center = (zone_low + zone_high) / 2.0
    ref_atr = _liquidity_ref_atr(current_price, atr)
    distance_atr = abs(level - current_price) / max(ref_atr, 1e-12)

    touches = _equal_touch_score(data, side, level, tol)
    sweep_count, sweep_vol = _sweep_stats_against_level(sweep_df, side, level, tol)
    vol_values = [float(p.get("volume_ratio", 1.0)) for p in cluster if np.isfinite(float(p.get("volume_ratio", 1.0)))]
    avg_vol = float(np.mean(vol_values)) if vol_values else None
    if sweep_vol is not None:
        avg_vol = max(avg_vol or 0.0, sweep_vol)

    latest_idx = max(int(p.get("index", 0)) for p in cluster)
    total_len = len(data) if data is not None and not data.empty else latest_idx + 1
    age_ratio = max((total_len - 1 - latest_idx) / max(total_len, 1), 0.0)
    recency_score = 1.25 * (1.0 - min(age_ratio, 1.0))

    params = _liq_role_params(role, mode)
    target = params["target_atr"]
    if role == "near":
        distance_score = max(0.0, 1.0 - distance_atr / 3.5) * 1.6
    elif role == "deep":
        distance_score = min(distance_atr / max(target, 0.1), 1.4) * 0.9
    else:
        distance_score = max(0.0, 1.0 - abs(distance_atr - target) / 3.5) * 1.1

    side_ok = level <= current_price if side == "low" else level >= current_price
    if not side_ok:
        distance_score -= 3.0

    vol_score = 0.0
    if avg_vol is not None:
        vol_score = min(max(avg_vol - 0.8, 0.0), 2.2) * 0.7

    score = (
        min(len(cluster), 4) * 0.90
        + min(touches, 6) * 0.55
        + min(sweep_count, 4) * 1.15
        + vol_score
        + recency_score
        + distance_score
    )

    strength = "mạnh" if (avg_vol or 0) >= 1.5 or sweep_count >= 2 or touches >= 4 else "vừa"
    if touches <= 1 and sweep_count == 0 and (avg_vol or 1.0) < 1.1:
        strength = "yếu"

    return {
        "low": zone_low,
        "high": zone_high,
        "center": center,
        "level": level,
        "score": score,
        "hits": max(len(cluster), touches),
        "touches": touches,
        "sweeps": sweep_count,
        "vol_ratio": avg_vol,
        "distance_atr": distance_atr,
        "strength": strength,
        "role": role,
        "level_span": level_high - level_low,
        "width": zone_high - zone_low,
    }


def _zone_for_liq_pools(
    level_df: pd.DataFrame | None,
    sweep_df: pd.DataFrame | None,
    current_price: float,
    atr: float | None,
    side: str,
    role: str,
    mode: str,
    lookback: int | None,
    m: int = 2,
) -> tuple[float | None, float | None, int, dict]:
    data = level_df.tail(lookback).reset_index(drop=True) if level_df is not None and not level_df.empty else None
    if data is None or data.empty:
        return None, None, 0, _zone_meta_default(role)

    points = _fractal_swing_points(data, side, lookback=None, m=m)
    if not points:
        # Fallback only takes one extreme still on the correct side, and still builds the box outside that extreme.
        col = "low" if side == "low" else "high"
        idx = int(data[col].idxmin() if side == "low" else data[col].idxmax())
        points = [{
            "price": float(data.loc[idx, col]),
            "index": idx,
            "time": data.loc[idx].get("timestamp"),
            "volume_ratio": _safe_float(data.loc[idx].get("vol_ratio"), 1.0) or 1.0,
            "kind": "extreme_fallback",
        }]

    # Only keep the level on the correct side. The lower zone is the stop pool below the swing low; the upper zone is the stop pool above the swing high.
    if side == "low":
        points = [p for p in points if float(p["price"]) <= current_price]
    else:
        points = [p for p in points if float(p["price"]) >= current_price]
    if not points:
        return None, None, 0, _zone_meta_default(role)

    clusters = _cluster_liq_levels(points, current_price, atr, role, mode)
    scored = [_score_liq_cluster(c, data, sweep_df, current_price, atr, side, role, mode) for c in clusters]
    # Drop zones that are still abnormally wide due to noisy data. For BTC scalp, a width above the cap is discarded.
    max_width = current_price * _liq_role_params(role, mode)["max_box_pct"] * 1.10
    scored = [s for s in scored if (s["high"] - s["low"]) <= max_width]
    if not scored:
        return None, None, 0, _zone_meta_default(role)

    best = max(scored, key=lambda x: x["score"])
    meta = {
        "role": role,
        "touches": int(best["touches"]),
        "sweeps": int(best["sweeps"]),
        "vol_ratio": best["vol_ratio"],
        "score": round(float(best["score"]), 2),
        "distance_atr": round(float(best["distance_atr"]), 2),
        "strength": best["strength"],
        "swing_level": round(float(best["level"]), 2),
        "zone_width": round(float(best["width"]), 2),
        "method": "fractal_swing_pool",
    }
    meta["side_state"] = _zone_side_state((best["low"], best["high"], int(best["hits"]), meta), current_price)
    return best["low"], best["high"], int(best["hits"]), meta


def _first_valid_df(*dfs: pd.DataFrame | None) -> pd.DataFrame | None:
    for df in dfs:
        if df is not None and not df.empty:
            return df
    return None


def _zone_gap_to_price(zone: tuple | None, current_price: float, side: str) -> float:
    """Distance from the current price to the zone's inner edge.

    side="lower": the zone sits below price, gap = current - high.
    side="upper": the zone sits above price, gap = low - current.
    If the zone is already touching/wrapping price, gap = 0.
    """
    if not zone or len(zone) < 2 or zone[0] is None or zone[1] is None:
        return float("inf")
    low, high = float(zone[0]), float(zone[1])
    if low <= current_price <= high:
        return 0.0
    if side == "lower":
        return max(current_price - high, 0.0)
    return max(low - current_price, 0.0)


def _liq_zone_overlap_ratio(a: tuple | None, b: tuple | None) -> float:
    if not a or not b or a[0] is None or a[1] is None or b[0] is None or b[1] is None:
        return 0.0
    a_low, a_high = float(a[0]), float(a[1])
    b_low, b_high = float(b[0]), float(b[1])
    overlap = max(0.0, min(a_high, b_high) - max(a_low, b_low))
    smaller = max(min(a_high - a_low, b_high - b_low), 1e-12)
    return overlap / smaller


def _liq_zone_external_gap(a: tuple | None, b: tuple | None) -> float:
    """The empty gap between two liquidity boxes; 0 if they overlap or touch."""
    if not a or not b or a[0] is None or a[1] is None or b[0] is None or b[1] is None:
        return float("inf")
    a_low, a_high = float(a[0]), float(a[1])
    b_low, b_high = float(b[0]), float(b[1])
    if a_high < b_low:
        return b_low - a_high
    if b_high < a_low:
        return a_low - b_high
    return 0.0


def _liq_zone_width(zone: tuple | None) -> float:
    if not zone or zone[0] is None or zone[1] is None:
        return 0.0
    return max(float(zone[1]) - float(zone[0]), 0.0)


def _mark_zone_merged_pool(zone: tuple, merged_with: str | None = None) -> tuple:
    """Internal marker for when near/main/deep get merged because they belong to the same liquidity cluster."""
    if not zone or len(zone) < 4 or not isinstance(zone[3], dict):
        return zone
    meta = dict(zone[3])
    meta["merged_pool"] = True
    if merged_with:
        roles = set(str(meta.get("merged_roles", "")).split("/")) if meta.get("merged_roles") else set()
        roles.add(str(meta.get("role", "")))
        roles.add(str(merged_with))
        roles = {r for r in roles if r}
        meta["merged_roles"] = "/".join(sorted(roles))
    return (zone[0], zone[1], zone[2], meta)


def _liq_zones_same_pool(a: tuple | None, b: tuple | None, current_price: float, mode: str) -> bool:
    """Avoid printing the same pool as separate near/main/deep zones.

    This was the main bug in earlier versions: two boxes that don't overlap much but sit only a
    tiny distance apart were still assigned as different near/main/deep zones. For scalp, if two
    zones are less than about 0.10% of price apart, or less than ~1 box-width apart, they're
    treated as the same stop/liquidity pool and not printed as separate targets.
    """
    if not a or not b:
        return False

    if _liq_zone_overlap_ratio(a, b) >= 0.20:
        return True

    gap = _liq_zone_external_gap(a, b)
    width_a = _liq_zone_width(a)
    width_b = _liq_zone_width(b)
    max_width = max(width_a, width_b, 1e-12)
    avg_width = max((width_a + width_b) / 2.0, 1e-12)

    # This threshold is for merging nearby roles, NOT for forcing distant zones together.
    # Scalp needs tight merging to avoid a near/main/deep output that only differs by a few USDT.
    gap_pct = 0.0010 if mode == "short" else 0.0022
    close_gap_threshold = max(current_price * gap_pct, avg_width * 0.85)
    if gap <= close_gap_threshold:
        return True

    ma = a[3] if len(a) > 3 and isinstance(a[3], dict) else {}
    mb = b[3] if len(b) > 3 and isinstance(b[3], dict) else {}
    la = ma.get("swing_level")
    lb = mb.get("swing_level")
    if la is None or lb is None:
        return False

    level_pct = 0.0013 if mode == "short" else 0.0028
    level_threshold = max(current_price * level_pct, max_width * 1.35)
    return abs(float(la) - float(lb)) <= level_threshold


def _copy_zone_with_assigned_role(zone: tuple, role: str) -> tuple:
    if not zone or len(zone) < 4 or not isinstance(zone[3], dict):
        return zone
    meta = dict(zone[3])
    meta["assigned_role"] = role
    meta["role"] = role
    return (zone[0], zone[1], zone[2], meta)


def _normalize_liquidity_role_order(zones: dict, current_price: float, mode: str) -> None:
    """Normalize near/main/deep after each candidate has been computed independently.

    Reason: the same H4/D1 data can produce multiple candidates, but if each role is chosen
    independently you can get "deep" ending up closer than "main", or the same zone printed twice.
    This function doesn't invent new zones; it only reorders the existing candidates by real distance:
    - lower side: the closer to price, the higher the swing low.
    - upper side: the closer to price, the lower the swing high.
    - zones in the same pool are dropped.
    - if the second candidate is too far for scalp, it's assigned to "deep" and "main" becomes N/A.
    """
    far_pct_cut = 0.025 if mode == "short" else 0.060
    far_atr_cut = 10.0 if mode == "short" else 12.0

    for side in ("lower", "upper"):
        raw: list[tuple] = []
        for role in ("near", "main", "deep"):
            z = zones.get(f"{side}_{role}")
            if z and z[0] is not None and z[1] is not None:
                raw.append(z)

        # Sort by gap first, then by descending score, so the closer candidate is always considered first.
        raw.sort(
            key=lambda z: (
                _zone_gap_to_price(z, current_price, side),
                -float((z[3] if len(z) > 3 and isinstance(z[3], dict) else {}).get("score", 0.0)),
            )
        )

        unique: list[tuple] = []
        for z in raw:
            if any(_liq_zones_same_pool(z, kept, current_price, mode) for kept in unique):
                # If candidates overlap or sit too close to the same pool, keep only one; don't print them as separate near/main/deep.
                for i, kept in enumerate(unique):
                    if _liq_zones_same_pool(z, kept, current_price, mode):
                        mz = z[3] if len(z) > 3 and isinstance(z[3], dict) else {}
                        mk = kept[3] if len(kept) > 3 and isinstance(kept[3], dict) else {}
                        wz = abs(float(z[1]) - float(z[0]))
                        wk = abs(float(kept[1]) - float(kept[0]))
                        z_role = str(mz.get("role", ""))
                        kept_role = str(mk.get("role", ""))
                        # Within the same pool, prefer the narrower box so scalp doesn't print a meaninglessly wide zone.
                        # If the widths are nearly equal, use the score to pick the higher-quality candidate.
                        choose_z = wz < wk * 0.92 or (
                            abs(wz - wk) <= wk * 0.08
                            and float(mz.get("score", 0.0)) > float(mk.get("score", 0.0))
                        )
                        if choose_z:
                            unique[i] = _mark_zone_merged_pool(z, kept_role)
                        else:
                            unique[i] = _mark_zone_merged_pool(kept, z_role)
                        break
                continue
            unique.append(z)

        assigned = {"near": None, "main": None, "deep": None}
        if unique:
            assigned["near"] = _copy_zone_with_assigned_role(unique[0], "near")

        for z in unique[1:]:
            gap = _zone_gap_to_price(z, current_price, side)
            gap_pct = gap / max(current_price, 1e-12)
            meta = z[3] if len(z) > 3 and isinstance(z[3], dict) else {}
            distance_atr = float(meta.get("distance_atr", 0.0) or 0.0)
            is_far = gap_pct >= far_pct_cut or distance_atr >= far_atr_cut

            if is_far:
                if assigned["deep"] is None:
                    assigned["deep"] = _copy_zone_with_assigned_role(z, "deep")
                # If a deep zone already exists, drop farther/weaker candidates to keep the prompt from getting diluted.
                continue

            if assigned["main"] is None:
                assigned["main"] = _copy_zone_with_assigned_role(z, "main")
            elif assigned["deep"] is None:
                assigned["deep"] = _copy_zone_with_assigned_role(z, "deep")

        # If there's no main zone but a deep zone happens to be very close (only 2 candidates), don't promote deep to main.
        # If both main and deep exist, make sure deep is actually farther out than main.
        if assigned["main"] is not None and assigned["deep"] is not None:
            main_gap = _zone_gap_to_price(assigned["main"], current_price, side)
            deep_gap = _zone_gap_to_price(assigned["deep"], current_price, side)
            if deep_gap < main_gap:
                assigned["main"], assigned["deep"] = assigned["deep"], assigned["main"]
                assigned["main"] = _copy_zone_with_assigned_role(assigned["main"], "main")
                assigned["deep"] = _copy_zone_with_assigned_role(assigned["deep"], "deep")

        for role in ("near", "main", "deep"):
            zones[f"{side}_{role}"] = assigned[role]

    # After role normalization, the label should be the trading role, not the old timeframe-based label.
    zones["label_near"] = "gần"
    zones["label_main"] = "chính"
    zones["label_deep"] = "sâu"
    zones["liquidity_method"] = "fractal_swing_pool_v13_longer_lookback_tp_guard"


def _fmt_zone_tuple(zone: tuple | None, current_price: float | None = None) -> str:
    if not zone:
        return "N/A"
    low = zone[0] if len(zone) > 0 else None
    high = zone[1] if len(zone) > 1 else None
    hits = zone[2] if len(zone) > 2 else 0
    meta = zone[3] if len(zone) > 3 and isinstance(zone[3], dict) else {}
    if low is None or high is None:
        return "N/A"

    details: list[str] = []
    touches = meta.get("touches")
    sweeps = meta.get("sweeps")
    vol_ratio = meta.get("vol_ratio")
    distance_atr = meta.get("distance_atr")
    strength = meta.get("strength")
    swing_level = meta.get("swing_level")
    zone_width = meta.get("zone_width")

    if strength:
        details.append(f"{strength}")
    if swing_level is not None:
        details.append(f"level {fmt(swing_level)}")
    if touches:
        details.append(f"{int(touches)} chạm")
    elif hits:
        details.append(f"{int(hits)} điểm")
    if sweeps:
        details.append(f"quét {int(sweeps)}")
    if vol_ratio is not None and np.isfinite(vol_ratio):
        details.append(f"vol {float(vol_ratio):.2f}x")
    if distance_atr is not None and np.isfinite(distance_atr):
        details.append(f"~{float(distance_atr):.1f}ATR")
    if zone_width is not None and np.isfinite(zone_width):
        details.append(f"rộng {fmt(zone_width)}")
    if current_price is not None and float(low) <= current_price <= float(high):
        details.append("đang chạm giá")
    if meta.get("fallback"):
        details.append("fallback cực trị")

    detail_text = f" ({', '.join(details)})" if details else ""
    return f"{fmt(low)}–{fmt(high)}{detail_text}"

def _zones_have_meaningful_overlap(a: tuple | None, b: tuple | None) -> bool:
    if not a or not b:
        return False
    a_low, a_high = a[0], a[1]
    b_low, b_high = b[0], b[1]
    if a_low is None or a_high is None or b_low is None or b_high is None:
        return False
    overlap = max(0.0, min(float(a_high), float(b_high)) - max(float(a_low), float(b_low)))
    width = max(min(float(a_high) - float(a_low), float(b_high) - float(b_low)), 1e-12)
    return overlap / width >= 0.55


def _liquidity_overlap_note(zones: dict, side: str) -> str:
    pairs = [
        ("gần", "chính", zones.get(f"{side}_near"), zones.get(f"{side}_main")),
        ("chính", "sâu", zones.get(f"{side}_main"), zones.get(f"{side}_deep")),
        ("gần", "sâu", zones.get(f"{side}_near"), zones.get(f"{side}_deep")),
    ]
    overlapped = [f"{a}/{b}" for a, b, za, zb in pairs if _zones_have_meaningful_overlap(za, zb)]
    if not overlapped:
        return ""
    return f" | Lưu ý: vùng {', '.join(overlapped)} đang trùng mạnh, xem là cùng một cụm thanh khoản thay vì 3 mục tiêu riêng."


def _structure_info(df: pd.DataFrame | None, current_price: float | None) -> dict:
    df = _closed_candles(df)
    if df is None or df.empty:
        return {}
    data_recent = df.tail(60)
    data_major = df.tail(120 if len(df) >= 120 else len(df))
    if data_recent.empty or data_major.empty:
        return {}
    recent_high = float(data_recent["high"].max())
    recent_low = float(data_recent["low"].min())
    major_high = float(data_major["high"].max())
    major_low = float(data_major["low"].min())
    swing_low, swing_high = recent_low, recent_high
    span = max(swing_high - swing_low, 0.0)
    fibs = {}
    if span > 0:
        fibs = {
            "0.382": swing_low + span * 0.382,
            "0.5": swing_low + span * 0.5,
            "0.618": swing_low + span * 0.618,
        }
    first_close = float(data_recent.iloc[0]["close"])
    last_close = float(data_recent.iloc[-1]["close"])
    if last_close > first_close * 1.003:
        trend = "TĂNG"
    elif last_close < first_close * 0.997:
        trend = "GIẢM"
    else:
        trend = "ĐI NGANG"
    pivot_highs = _find_pivots(df, "high", 80)
    pivot_lows = _find_pivots(df, "low", 80)
    recent_pivot_high = pivot_highs[-1]["price"] if pivot_highs else recent_high
    recent_pivot_low = pivot_lows[-1]["price"] if pivot_lows else recent_low
    return {
        "trend": trend,
        "recent_high": recent_high,
        "recent_low": recent_low,
        "major_high": major_high,
        "major_low": major_low,
        "recent_pivot_high": recent_pivot_high,
        "recent_pivot_low": recent_pivot_low,
        "fib": fibs,
    }


def _consecutive_candles(df: pd.DataFrame | None) -> str:
    data = _closed_candles(df) if "_closed_candles" in globals() else df
    if data is None or len(data) < 2:
        return "Không đủ dữ liệu"
    count = 0
    last_dir = None
    for _, row in data.tail(12).iloc[::-1].iterrows():
        direction = "xanh" if float(row["close"]) > float(row["open"]) else "đỏ" if float(row["close"]) < float(row["open"]) else "doji"
        if last_dir is None:
            last_dir = direction
            count = 1
        elif direction == last_dir:
            count += 1
        else:
            break
    return f"{count} nến {last_dir} liên tiếp"


def _wick_body_info(df: pd.DataFrame | None) -> str:
    data = _closed_candles(df) if "_closed_candles" in globals() else df
    if data is None or data.empty:
        return "Không đủ dữ liệu"
    row = data.iloc[-1]
    high, low, open_, close = map(float, [row["high"], row["low"], row["open"], row["close"]])
    rng = max(high - low, 1e-12)
    body = abs(close - open_)
    upper = high - max(open_, close)
    lower = min(open_, close) - low
    return f"nến đã đóng: thân {body / rng * 100:.0f}%, râu trên {upper / rng * 100:.0f}%, râu dưới {lower / rng * 100:.0f}%"


def _mode_labels(mode: str) -> tuple[str, str, str]:
    # main/structure/big are the timeframes used for decisions, not necessarily the smallest trigger timeframe.
    # SCALP: 1H decides the setup, 4H confirms the trend, 1D is the macro context. 15M is timing only.
    if mode == "short":
        return "1H", "4H", "1D"
    # SWING: 4H is the setup/entry zone, 1D decides the main trend, 1W is macro. 1H is secondary timing only.
    return "4H", "1D", "1W"


def _mode_trigger_label(mode: str) -> str:
    return "15M" if mode == "short" else "1H"


def _mode_role_text(mode: str) -> str:
    if mode == "short":
        return (
            "SCALP roles: 4H quyết định xu hướng/cấu trúc chính; 1H là khung thiết kế setup, vùng Entry, điểm vô hiệu SL và TP gần; "
            "1D chỉ là bối cảnh lớn; 15M chỉ dùng để xác nhận timing, sweep/râu nến và không được quyết định hướng, độ rộng Entry, SL hoặc TP."
        )
    return (
        "SWING roles: 1D quyết định xu hướng/cấu trúc chính; 4H là khung thiết kế setup, vùng Entry, điểm vô hiệu SL và TP gần; "
        "1W là bối cảnh lớn và mục tiêu mở rộng; 1H chỉ dùng để tinh chỉnh timing, không được quyết định hướng, độ rộng Entry, SL hoặc TP."
    )


def _closed_candles(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Use closed candles to find swings/invalidation levels, avoiding an unfinished realtime candle as the SL basis."""
    if df is None or df.empty:
        return None
    if len(df) >= 3:
        return df.iloc[:-1].copy()
    return df.copy()


# Scoring: only one final score, self-graded by the final model: Signal Score /100.
# The new name takes priority; the old name is kept as a fallback so old DB/Railway setups don't break.
MIN_SIGNAL_SCORE = _env_float(
    "TEOPARD_MIN_SIGNAL_SCORE",
    _env_float("TEOPARD_MIN_SCALP_CONFIDENCE", 62.0),
)
MIN_ACTION_CONFIDENCE_SCALP = MIN_SIGNAL_SCORE
MIN_ACTION_CONFIDENCE_SWING = _env_float("TEOPARD_MIN_SWING_CONFIDENCE", MIN_SIGNAL_SCORE)
MIN_SETUP_STRENGTH = _env_float("TEOPARD_MIN_SETUP_STRENGTH", MIN_SIGNAL_SCORE)
MIN_REVERSAL_CONFIDENCE_SCALP = _env_float("TEOPARD_MIN_REVERSAL_CONFIDENCE", 50.0)
MIN_REVERSAL_CONFIDENCE_WITH_BAD_MOMENTUM = _env_float("TEOPARD_MIN_REVERSAL_BAD_MOMENTUM_CONFIDENCE", 52.0)

# Final 100-point rubric. The model scores itself; Python only parses the total and gates on the Signal Score.
SIGNAL_SCORE_WEIGHTS = {
    "huong_boi_canh_da_khung": 30.0,
    "entry_timing": 20.0,
    "chat_luong_ke_hoach": 25.0,
    "mau_thuan_rui_ro_nhieu": 15.0,
    "thuc_thi_thuc_te": 10.0,
}

# Legacy weights kept for internal debug/data_support and for reading old outputs if needed.


def _analysis_row(df: pd.DataFrame | None):
    """
    Use the most recently closed candle to read indicators/volume.

    Binance usually also returns the currently running candle; its volume is very low right after
    the candle opens, which can make the model mistakenly read it as weak liquidity and choose NO TRADE.
    So indicator/regime/snapshot logic uses candle -2 whenever there's enough data.
    """
    if df is None or df.empty:
        return None
    return df.iloc[-2] if len(df) >= 2 else df.iloc[-1]


REGIME_LABEL_VI = {
    "EMA_TANG": "EMA nghiêng tăng",
    "EMA_GIAM": "EMA nghiêng giảm",
    "EMA_DAN_XEN": "EMA đan xen",
    "TRENDING_UP": "xu hướng tăng rõ",
    "TRENDING_DOWN": "xu hướng giảm rõ",
    "RANGE_CHOPPY": "đi ngang/nhiễu",
    "MIXED_TRANSITION": "trạng thái chuyển pha",
    "BEAR_TREND": "xu hướng giảm",
    "BULL_TREND": "xu hướng tăng",
    "MIXED_UNCLEAR": "chưa rõ xu hướng",
    "HIGH_VOLATILITY": "biến động mạnh",
    "LOW_VOLATILITY": "biến động thấp",
    "NORMAL_VOLATILITY": "biến động bình thường",
    "HIGH_VOLUME": "khối lượng cao",
    "LOW_VOLUME": "khối lượng thấp",
    "NORMAL_VOLUME": "khối lượng bình thường",
    "LOW_LIQUIDITY_RISK": "rủi ro thanh khoản thấp",
    "HIGH_VOLATILITY_RISK": "rủi ro biến động mạnh",
    "LOWER_TIMEFRAME_PULLBACK_AGAINST_STRUCTURE": "khung nhỏ đang hồi ngược cấu trúc lớn",
}


def _label_vi(code: str) -> str:
    return REGIME_LABEL_VI.get(code, code)


def _ema_state_from_last(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "N/A"
    last = _analysis_row(df)
    if last is None:
        return "N/A"
    if last["ema_7"] > last["ema_25"] > last["ema_50"]:
        return "EMA_TANG"
    if last["ema_7"] < last["ema_25"] < last["ema_50"]:
        return "EMA_GIAM"
    return "EMA_DAN_XEN"


def _timeframe_regime_details(label: str, df: pd.DataFrame | None) -> dict:
    if df is None or df.empty:
        return {
            "label": label,
            "trend_tag": "N/A",
            "vol_tag": "N/A",
            "volume_tag": "N/A",
            "ema_state": "N/A",
            "text": f"{label}: không đủ dữ liệu",
        }
    last = _analysis_row(df)
    if last is None:
        return {
            "label": label,
            "trend_tag": "N/A",
            "vol_tag": "N/A",
            "volume_tag": "N/A",
            "ema_state": "N/A",
            "text": f"{label}: không đủ dữ liệu",
        }
    close = float(last["close"])
    ema_state = _ema_state_from_last(df)
    rsi = _safe_float(last.get("rsi_14"), 50.0) or 50.0
    vol_ratio = _safe_float(last.get("vol_ratio"), 1.0) or 1.0
    ema_spread_pct = abs(float(last["ema_7"]) - float(last["ema_50"])) / max(close, 1e-12) * 100

    if ema_state == "EMA_TANG" and close >= float(last["ema_25"]) and rsi >= 52:
        trend_tag = "TRENDING_UP"
    elif ema_state == "EMA_GIAM" and close <= float(last["ema_25"]) and rsi <= 48:
        trend_tag = "TRENDING_DOWN"
    elif ema_spread_pct < 0.20 or 42 <= rsi <= 58:
        trend_tag = "RANGE_CHOPPY"
    else:
        trend_tag = "MIXED_TRANSITION"
    vol_tag = "N/A"

    if vol_ratio >= 1.50:
        volume_tag = "HIGH_VOLUME"
    elif vol_ratio <= 0.55:
        volume_tag = "LOW_VOLUME"
    else:
        volume_tag = "NORMAL_VOLUME"

    return {
        "label": label,
        "trend_tag": trend_tag,
        "vol_tag": vol_tag,
        "volume_tag": volume_tag,
        "ema_state": ema_state,
        "text": (
            f"{label}: {_label_vi(trend_tag)}, {_label_vi(volume_tag)}; "
            f"EMA={_label_vi(ema_state)}, RSI14={_fmt_metric(rsi,1)}, Vol={fmt(vol_ratio,2)}x"
        ),
    }


def build_market_regime_block(timeframe_data: dict[str, pd.DataFrame | None], mode: str) -> str:
    main_label, structure_label, big_label = _mode_labels(mode)
    main_state = _timeframe_regime_details(main_label, timeframe_data.get(main_label))
    structure_state = _timeframe_regime_details(structure_label, timeframe_data.get(structure_label))
    big_state = _timeframe_regime_details(big_label, timeframe_data.get(big_label))

    states = [main_state, structure_state, big_state]
    down_count = sum(s["trend_tag"] == "TRENDING_DOWN" for s in states)
    up_count = sum(s["trend_tag"] == "TRENDING_UP" for s in states)
    range_count = sum(s["trend_tag"] == "RANGE_CHOPPY" for s in states)
    low_volume_count = sum(s["volume_tag"] == "LOW_VOLUME" for s in states)

    if down_count >= 2:
        overall_code = "BEAR_TREND"
    elif up_count >= 2:
        overall_code = "BULL_TREND"
    elif range_count >= 2:
        overall_code = "RANGE_CHOPPY"
    else:
        overall_code = "MIXED_UNCLEAR"

    modifiers = []
    if low_volume_count >= 2:
        modifiers.append("LOW_LIQUIDITY_RISK")
    if (main_state["trend_tag"] == "TRENDING_UP" and structure_state["trend_tag"] == "TRENDING_DOWN") or (
        main_state["trend_tag"] == "TRENDING_DOWN" and structure_state["trend_tag"] == "TRENDING_UP"
    ):
        modifiers.append("LOWER_TIMEFRAME_PULLBACK_AGAINST_STRUCTURE")
    modifier_text = ", ".join(_label_vi(m) for m in modifiers) if modifiers else "không có ghi chú rủi ro lớn"

    return "\n".join([
        "Phân loại thị trường do Python:",
        f"- Xu hướng chính: {_label_vi(overall_code)}; ghi chú: {modifier_text}",
        f"- {main_state['text']}",
        f"- {structure_state['text']}",
        f"- {big_state['text']}",
        "- Cách dùng: đi ngang/nhiễu, chưa rõ xu hướng hoặc thanh khoản thấp là cảnh báo rủi ro. Không cố tạo LONG/SHORT nếu lợi thế không rõ; chỉ dùng lệnh chờ khi vùng Entry thật sự đẹp và có lý do kỹ thuật rõ ràng.",
    ])


def _format_candle_compact(row) -> str:
    open_ = float(row["open"])
    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])
    volume = float(row.get("volume", 0) or 0)
    rng = max(high - low, 1e-12)
    body_pct = abs(close - open_) / rng * 100
    upper_pct = (high - max(open_, close)) / rng * 100
    lower_pct = (min(open_, close) - low) / rng * 100
    direction = "xanh" if close > open_ else "đỏ" if close < open_ else "doji"
    taker_ratio = None
    try:
        if volume > 0:
            taker_ratio = float(row.get("taker_buy_volume", 0) or 0) / volume * 100
    except Exception:
        taker_ratio = None
    taker_text = f" TakerBuy:{fmt(taker_ratio,1)}%" if taker_ratio is not None else ""
    return (
        f"{str(row['timestamp'])[:16]} {direction} "
        f"O:{fmt(open_)} H:{fmt(high)} L:{fmt(low)} C:{fmt(close)} "
        f"Body:{body_pct:.0f}% U:{upper_pct:.0f}% D:{lower_pct:.0f}% "
        f"Vol:{fmt(row.get('vol_ratio'),2)}x{taker_text}"
    )


def build_raw_candle_context(timeframe_data: dict[str, pd.DataFrame | None], mode: str) -> str:
    """Send raw closed candles. The still-running candle is kept separate so the model doesn't treat it as confirmation."""
    main_label, structure_label, big_label = _mode_labels(mode)
    trigger_label = _mode_trigger_label(mode)
    if mode == "short":
        ordered = [("15M", 24, "trigger/timing"), ("1H", 24, "setup chính"), ("4H", 12, "trend filter"), ("1D", 8, "macro context")]
    else:
        ordered = [("1H", 12, "trigger phụ"), ("4H", 24, "setup"), ("1D", 18, "decision chính"), ("1W", 12, "macro context")]

    blocks = ["RAW_CANDLE_CONTEXT_CHON_LOC — CHỈ NẾN ĐÃ ĐÓNG:"]
    blocks.append(f"- Vai trò khung: {_mode_role_text(mode)}")
    for label, n, role in ordered:
        df = timeframe_data.get(label)
        closed_df = _closed_candles(df)
        if closed_df is None or closed_df.empty:
            blocks.append(f"- {label} ({role}): Không đủ dữ liệu nến đã đóng.")
            continue
        rows = ["  " + _format_candle_compact(row) for _, row in closed_df.tail(n).iterrows()]
        blocks.append(f"- {label} ({role}): {min(n, len(closed_df))} nến đã đóng gần nhất, dùng để đọc phá giả/rút râu/đuối lực:")
        blocks.extend(rows)
    return "\n".join(blocks)


def build_live_candle_context(timeframe_data: dict[str, pd.DataFrame | None], mode: str) -> str:
    """Separate the still-running candle from closed candles so it's used only for reference, not confirmation."""
    if mode == "short":
        labels = ["15M", "1H", "4H", "1D"]
    else:
        labels = ["1H", "4H", "1D", "1W"]
    blocks = ["LIVE_CANDLE_CONTEXT — NẾN ĐANG CHẠY, CHỈ THAM KHẢO:"]
    blocks.append("- Không dùng nến đang chạy để xác nhận entry/đảo chiều. Chỉ dùng để biết giá hiện tại đang di chuyển ra sao so với nến đã đóng.")
    for label in labels:
        df = timeframe_data.get(label)
        if df is None or df.empty or len(df) < 2:
            blocks.append(f"- {label}: Không đủ dữ liệu nến đang chạy.")
            continue
        row = df.iloc[-1]
        blocks.append(f"- {label} live: {_format_candle_compact(row)}")
    return "\n".join(blocks)


_TIMEFRAME_SECONDS_BY_LABEL = {
    "15M": 15 * 60,
    "1H": 60 * 60,
    "4H": 4 * 60 * 60,
    "1D": 24 * 60 * 60,
    "1W": 7 * 24 * 60 * 60,
}


def _fmt_metric(value, decimals: int = 2) -> str:
    number = _safe_float(value)
    if number is None or not np.isfinite(number):
        return "N/A"
    return f"{number:.{max(0, int(decimals))}f}"


def _pct_delta(new_value, old_value) -> float | None:
    new_num = _safe_float(new_value)
    old_num = _safe_float(old_value)
    if new_num is None or old_num is None or abs(old_num) <= 1e-12:
        return None
    return (new_num - old_num) / abs(old_num) * 100.0


def _closed_metric_delta(df: pd.DataFrame | None, column: str, bars: int) -> float | None:
    data = _closed_candles(df)
    if data is None or len(data) <= bars or column not in data.columns:
        return None
    return _safe_float(data.iloc[-1].get(column)) - _safe_float(data.iloc[-1 - bars].get(column)) \
        if _safe_float(data.iloc[-1].get(column)) is not None and _safe_float(data.iloc[-1 - bars].get(column)) is not None else None


def _closed_return_pct(df: pd.DataFrame | None, bars: int) -> float | None:
    data = _closed_candles(df)
    if data is None or len(data) <= bars:
        return None
    return _pct_delta(data.iloc[-1].get("close"), data.iloc[-1 - bars].get("close"))


def _closed_ema_slope_pct(df: pd.DataFrame | None, column: str, bars: int = 3) -> float | None:
    data = _closed_candles(df)
    if data is None or len(data) <= bars or column not in data.columns:
        return None
    return _pct_delta(data.iloc[-1].get(column), data.iloc[-1 - bars].get(column))


def _taker_buy_ratio(row) -> float | None:
    if row is None:
        return None
    volume = _safe_float(row.get("volume"))
    taker = _safe_float(row.get("taker_buy_volume"))
    if volume is None or taker is None or volume <= 0:
        return None
    return taker / volume * 100.0


def _taker_ratio_average(df: pd.DataFrame | None, bars: int) -> float | None:
    data = _closed_candles(df)
    if data is None or data.empty:
        return None
    values = [_taker_buy_ratio(row) for _, row in data.tail(bars).iterrows()]
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def _last_pivot_values(df: pd.DataFrame | None, side: str, count: int = 3) -> list[float]:
    data = _closed_candles(df)
    if data is None or data.empty:
        return []
    try:
        pivots = _find_pivots(data, side, lookback=min(120, len(data)), left=2, right=2)
    except Exception:
        pivots = []
    values: list[float] = []
    key = "high" if side == "high" else "low"
    for item in pivots[-count:]:
        value = _safe_float(item.get("price") if isinstance(item, dict) else None)
        if value is None and isinstance(item, dict):
            value = _safe_float(item.get(key))
        if value is not None:
            values.append(value)
    if values:
        return values[-count:]
    # Fallback: use rolling local extrema so the model still receives an ordered sequence.
    series = data[key].astype(float)
    local = []
    for idx in range(2, max(2, len(series) - 2)):
        window = series.iloc[idx - 2: idx + 3]
        value = float(series.iloc[idx])
        if (side == "high" and value >= float(window.max())) or (side == "low" and value <= float(window.min())):
            local.append(value)
    return local[-count:]


def _sequence_shape(values: list[float], high_side: bool) -> str:
    if len(values) < 2:
        return "N/A"
    eps = max(abs(values[-1]) * 1e-5, 1e-9)
    deltas = [values[i] - values[i - 1] for i in range(1, len(values))]
    if all(d > eps for d in deltas):
        return "đỉnh cao dần" if high_side else "đáy cao dần"
    if all(d < -eps for d in deltas):
        return "đỉnh thấp dần" if high_side else "đáy thấp dần"
    return "đan xen"


def _format_values(values: list[float]) -> str:
    return "→".join(fmt(v) for v in values) if values else "N/A"


def _live_candle_progress(row, label: str) -> float | None:
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
        duration = float(_TIMEFRAME_SECONDS_BY_LABEL.get(label, 0) or 0)
        return None if duration <= 0 else 0.0


def _ema_interaction_text(row, ema_column: str, current_price: float, atr: float | None) -> str:
    ema = _safe_float(row.get(ema_column)) if row is not None else None
    open_ = _safe_float(row.get("open")) if row is not None else None
    high = _safe_float(row.get("high")) if row is not None else None
    low = _safe_float(row.get("low")) if row is not None else None
    if None in (ema, open_, high, low) or ema is None or ema <= 0:
        return f"{ema_column.upper().replace('_', '')}:N/A"
    distance_pct = (current_price - ema) / ema * 100.0
    atr_num = _safe_float(atr, 0.0) or 0.0
    distance_atr = (current_price - ema) / atr_num if atr_num > 0 else None
    tol = max(abs(ema) * 0.00025, atr_num * 0.04, 1e-9)
    touched = low - tol <= ema <= high + tol
    state = "trên" if current_price > ema + tol else "dưới" if current_price < ema - tol else "sát"
    if touched:
        if open_ < ema - tol and current_price < ema - tol and high >= ema - tol:
            state = "test từ dưới rồi quay lại dưới"
        elif open_ > ema + tol and current_price > ema + tol and low <= ema + tol:
            state = "test từ trên rồi quay lại trên"
        elif open_ < ema - tol and current_price > ema + tol:
            state = "xuyên lên và đang giữ trên"
        elif open_ > ema + tol and current_price < ema - tol:
            state = "xuyên xuống và đang giữ dưới"
        else:
            state = "đang chạm"
    dist_text = f"{distance_pct:+.2f}%"
    if distance_atr is not None:
        dist_text += f"/{distance_atr:+.2f}ATR"
    return f"{ema_column.upper().replace('_', '')} {fmt(ema)} ({state}; dist {dist_text})"


def _lower_confirmation_text(
    timeframe_data: dict[str, pd.DataFrame | None],
    lower_label: str | None,
    reference_level: float | None,
    bars: int = 3,
) -> str:
    if not lower_label or reference_level is None:
        return ""
    lower = _closed_candles(timeframe_data.get(lower_label))
    if lower is None or lower.empty:
        return f"; {lower_label} giữ level: N/A"
    closes = [float(v) for v in lower.tail(bars)["close"].tolist()]
    above = sum(v > reference_level for v in closes)
    below = sum(v < reference_level for v in closes)
    return f"; {lower_label} {len(closes)} close gần nhất: trên {above}, dưới {below}"


# ─── Format helpers ───────────────────────────────────────────────────────────

def fmt(v, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if abs(v) >= 100:
        return f"{v:,.{decimals}f}"
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


def summarize_timeframe(label: str, df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return f"\nKHUNG {label}: Không đủ dữ liệu.\n"

    last = _analysis_row(df)
    if last is None:
        return f"\nKHUNG {label}: Không đủ dữ liệu.\n"
    last_pos = df.index.get_loc(last.name) if hasattr(last, "name") else len(df) - 1
    prev = df.iloc[max(0, int(last_pos) - 1)] if len(df) >= 2 else last
    ema7, ema25, ema50 = last["ema_7"], last["ema_25"], last["ema_50"]

    if ema7 > ema25 > ema50:
        ema_align = "TĂNG (EMA7>EMA25>EMA50)"
    elif ema7 < ema25 < ema50:
        ema_align = "GIẢM (EMA7<EMA25<EMA50)"
    else:
        ema_align = "TRUNG TÍNH (đan xen)"

    macd_dir = "TĂNG" if last["macd_hist"] > 0 else "GIẢM"
    macd_cross = ""
    if prev["macd_hist"] < 0 <= last["macd_hist"]:
        macd_cross = " — VỪA CROSS BULLISH"
    elif prev["macd_hist"] > 0 >= last["macd_hist"]:
        macd_cross = " — VỪA CROSS BEARISH"

    vol_lbl = "CAO" if last["vol_ratio"] > 1.5 else ("THẤP" if last["vol_ratio"] < 0.7 else "BÌNH THƯỜNG")

    closed_df = _closed_candles(df)
    if closed_df is None or closed_df.empty:
        closed_df = df
    window    = closed_df.tail(50)
    key_high  = window["high"].max()
    key_low   = window["low"].min()

    candles = "\n".join(
        f"  {str(row['timestamp'])[:16]} O:{fmt(row['open'])} H:{fmt(row['high'])} "
        f"L:{fmt(row['low'])} C:{fmt(row['close'])} "
        f"RSI14:{fmt(row['rsi_14'],1)} Vol:{fmt(row['vol_ratio'],2)}x"
        for _, row in closed_df.tail(6).iterrows()
    )

    return "\n".join([
        f"\nKHUNG {label}:",
        f"  Giá: {fmt(last['close'])} | Nến trước: {fmt(prev['close'])}",
        f"  EMA7={fmt(ema7)} EMA25={fmt(ema25)} EMA50={fmt(ema50)} → {ema_align}",
        f"  RSI(6)={fmt(last['rsi_6'],1)} RSI(14)={fmt(last['rsi_14'],1)}",
        f"  MACD={fmt(last['macd_line'],4)} Signal={fmt(last['macd_signal'],4)}; {macd_momentum_text(last['macd_hist'])} → {macd_dir}{macd_cross}",
        f"  Volume={fmt(last['vol_ratio'],2)}x → {vol_lbl}",
        f"  Nến đã đóng: {_consecutive_candles(df)} | {_wick_body_info(df)}",
        f"  High/Low 50 nến: {fmt(key_high)} / {fmt(key_low)}",
        f"  6 nến đã đóng gần nhất:",
        candles,
    ])


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
        ema_align = "mixed"
        if last["ema_7"] > last["ema_25"] > last["ema_50"]:
            ema_align = "bullish"
        elif last["ema_7"] < last["ema_25"] < last["ema_50"]:
            ema_align = "bearish"

        lines.append(
            f"{label}: close={fmt(last['close'])}, EMA={ema_align} "
            f"(7={fmt(last['ema_7'])},25={fmt(last['ema_25'])},50={fmt(last['ema_50'])}), "
            f"RSI14={fmt(last['rsi_14'], 1)}, {macd_momentum_text(last['macd_hist'])}, "
            f"vol={fmt(last['vol_ratio'], 2)}x"
        )

    return " | ".join(lines)


def get_current_price_str(symbol: str) -> tuple[str, float | None]:
    price = get_current_price_raw(symbol)
    if price is None:
        return "Giá hiện tại: không có dữ liệu", None
    return f"Giá hiện tại: {fmt(price)} USDT", price


# ─── History formatter ────────────────────────


def format_prediction_history(history: list[dict]) -> str:
    """Learning context without old price anchors or directional win-rate bias."""
    if not history:
        return "No previous traded outcome for this symbol/mode."

    selected = list(history or [])[:PREDICTION_HISTORY_COUNT]
    lines = [
        f"USER-SPECIFIC RECENT OUTCOME LESSONS ({len(selected)} confirmed trades):",
        "- This history is diagnostic only. Do not prefer LONG/SHORT and do not reuse any old Entry/SL/TP from it.",
    ]
    for i, item in enumerate(selected, 1):
        outcome = item.get("result") or "PENDING"
        reason = str(item.get("result_reason") or "Outcome detail unavailable.").strip().replace("\n", " ")
        if len(reason) > 220:
            reason = reason[:217] + "..."
        decision_reason = str(item.get("reasoning_summary") or "").strip().replace("\n", " ")
        if len(decision_reason) > 260:
            decision_reason = decision_reason[:257] + "..."
        lines.append(
            f"- #{i} Outcome={outcome}. Original thesis summary: {decision_reason or 'N/A'}. Outcome note: {reason}"
        )
    lines.append("Use only to avoid repeated analytical mistakes; current OHLCV must determine direction and all new levels.")
    return "\n".join(lines)


def format_deepseek_history_compact(history: list[dict], limit: int = 3) -> str:
    """Outcome-only history for prefilter; excludes old directions and price levels."""
    selected = list(history or [])[:max(0, int(limit))]
    if not selected:
        return "Không có lịch sử đã trade cho coin/mode này."
    lines = [
        f"{len(selected)} kết quả gần nhất (chỉ để tránh lặp lỗi, không dùng để nghiêng LONG/SHORT):"
    ]
    for index, pred in enumerate(selected, 1):
        outcome_reason = str(pred.get("result_reason") or "").strip().replace("\n", " ")
        if len(outcome_reason) > 150:
            outcome_reason = outcome_reason[:147] + "..."
        lines.append(f"- #{index} Kết quả={pred.get('result') or 'N/A'}; ghi chú={outcome_reason or 'N/A'}.")
    return "\n".join(lines)


def _json_safe_value(value):
    """Convert numpy/pandas values into strict JSON-safe Python primitives."""
    try:
        if value is None:
            return None
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            value = float(value)
        if isinstance(value, float):
            return value if np.isfinite(value) else None
        if isinstance(value, (datetime, pd.Timestamp)):
            return value.isoformat()
    except Exception:
        pass
    return value


def _truncate_text(text: str | None, limit: int = 600) -> str | None:
    if not text:
        return None
    text = str(text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


# ─── Select current AI provider/API key/model (multi-provider config) ───



def get_ai_model_name() -> str:
    return DEEPSEEK_FINAL_MODEL


def get_ai_provider_label() -> str:
    return "deepseek"


def ensure_ai_config() -> None:
    if not DEEPSEEK_FINAL_API_KEY:
        raise RuntimeError(
            "Missing DeepSeek API key. Set DEEPSEEK_FINAL_API_KEY or DEEPSEEK_API_KEY in Railway variables."
        )


def _deepseek_create_once(
    system: str | None,
    messages: list,
    max_tokens: int,
    timeout: int | None = None,
    model: str | None = None,
    temperature: float | None = None,
    response_format: dict | None = None,
    reasoning_effort: str | None = None,
) -> dict:
    """Call DeepSeek Flash/reviewer via the native Chat Completions endpoint.

    This helper is intentionally separate from the Pro planner helper because
    prefilter/reviewer may use another model, JSON mode and temperature.
    """
    api_key = DEEPSEEK_API_KEY or DEEPSEEK_FINAL_API_KEY
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY for Flash prefilter/reviewer.")

    payload_messages = []
    if system:
        payload_messages.append({"role": "system", "content": system})
    payload_messages.extend(messages or [])

    effective_model = (model or DEEPSEEK_MODEL or "").strip()
    if not effective_model:
        raise RuntimeError("Missing DEEPSEEK_MODEL for Flash prefilter/reviewer.")

    payload = {
        "model": effective_model,
        "messages": payload_messages,
        "max_tokens": int(max_tokens),
    }
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if response_format:
        payload["response_format"] = response_format

    effort_norm = (reasoning_effort or "").strip().lower()
    if effort_norm in {"", "off", "none", "false", "0", "disabled"}:
        payload["thinking"] = {"type": "disabled"}
        effective_effort = "off"
    else:
        effort = "max" if effort_norm in {"max", "xhigh"} else "high"
        payload["thinking"] = {"type": "enabled"}
        payload["reasoning_effort"] = effort
        effective_effort = effort

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    request_timeout = int(timeout or DEEPSEEK_TIMEOUT_SECONDS)
    r = requests.post(
        f"{DEEPSEEK_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=request_timeout,
    )
    try:
        r.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"DeepSeek Flash API error: {r.status_code} - {r.text[:1000]}") from exc

    data = r.json()
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning_content = message.get("reasoning_content") or message.get("reasoning") or ""
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    if isinstance(reasoning_content, list):
        reasoning_content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in reasoning_content
        )

    return {
        "text": str(content or ""),
        "reasoning_text": str(reasoning_content or ""),
        "stop_reason": choice.get("finish_reason"),
        "usage": data.get("usage"),
        "effort": effective_effort,
        "model": effective_model,
    }

def _openrouter_reviewer_create_once(
    system: str | None,
    messages: list,
    max_tokens: int,
    timeout: int | None = None,
    model: str | None = None,
    temperature: float | None = None,
    response_format: dict | None = None,
) -> dict:
    """Call OpenRouter for the reviewer step using the OpenAI-compatible Chat Completions API."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Missing OPENROUTER_API_KEY. Set it in Railway variables.")

    payload_messages = []
    if system:
        payload_messages.append({"role": "system", "content": system})
    payload_messages.extend(messages or [])

    effective_model = (model or OPENROUTER_REVIEWER_MODEL or "").strip()
    if not effective_model:
        raise RuntimeError("Missing OPENROUTER_REVIEWER_MODEL.")

    payload: dict = {
        "model": effective_model,
        "messages": payload_messages,
        "max_tokens": int(max_tokens),
    }
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if response_format:
        payload["response_format"] = response_format

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    request_timeout = int(timeout or OPENROUTER_REVIEWER_TIMEOUT_SECONDS)
    r = requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=request_timeout,
    )
    try:
        r.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"OpenRouter reviewer API error: {r.status_code} - {r.text[:1000]}") from exc

    data = r.json()
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return {
        "text": str(content or ""),
        "reasoning_text": "",
        "stop_reason": choice.get("finish_reason"),
        "usage": data.get("usage"),
        "model": effective_model,
    }


def _deepseek_final_create_once(
    system: str | None,
    messages: list,
    max_tokens: int,
    timeout: int,
    reasoning_effort: str | None = None,
) -> dict:
    """Call native DeepSeek V4 Pro for the final analysis via Chat Completions."""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_FINAL_API_KEY}",
        "Content-Type": "application/json",
    }

    payload_messages = []
    if system:
        payload_messages.append({"role": "system", "content": system})
    payload_messages.extend(messages)

    if reasoning_effort is None:
        effective_reasoning_effort = (DEEPSEEK_FINAL_REASONING_EFFORT or "max").strip().lower()
    else:
        effective_reasoning_effort = (reasoning_effort or "").strip().lower()

    payload = {
        "model": DEEPSEEK_FINAL_MODEL,
        "messages": payload_messages,
        "max_tokens": max_tokens,
    }

    if effective_reasoning_effort in {"", "off", "none", "false", "0", "disabled"}:
        payload["thinking"] = {"type": "disabled"}
        effective_effort_for_log = "off"
    else:
        # The DeepSeek V4 API supports high/max; old values are mapped to these two valid levels.
        if effective_reasoning_effort in {"max", "xhigh"}:
            effort = "max"
        else:
            effort = "high"
        payload["thinking"] = {"type": "enabled"}
        payload["reasoning_effort"] = effort
        effective_effort_for_log = effort

    r = requests.post(
        f"{DEEPSEEK_FINAL_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    try:
        r.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"DeepSeek API error: {r.status_code} - {r.text[:1000]}") from exc

    data = r.json()
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}

    def _flatten_text(value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = []
            for part in value:
                if isinstance(part, dict):
                    parts.append(str(part.get("text") or part.get("content") or part.get("value") or ""))
                else:
                    parts.append(str(part))
            return "".join(parts)
        if isinstance(value, dict):
            return str(value.get("text") or value.get("content") or value.get("value") or "")
        return str(value)

    content = _flatten_text(message.get("content"))
    if not content.strip():
        # Some OpenAI-compatible responses return the final text in an alternate field.
        content = _flatten_text(message.get("final")) or _flatten_text(message.get("output_text"))
    reasoning_content = (
        _flatten_text(message.get("reasoning_content"))
        or _flatten_text(message.get("reasoning"))
        or _flatten_text(message.get("analysis"))
    )
    finish_reason = choice.get("finish_reason")
    usage = data.get("usage")

    if not content.strip():
        print(
            f"[DEEPSEEK_EMPTY_FINAL] model={DEEPSEEK_FINAL_MODEL} "
            f"effort={effective_effort_for_log} finish_reason={finish_reason} "
            f"content_chars={len(content)} reasoning_chars={len(reasoning_content)} "
            f"usage={usage} response_keys={list(data.keys())} "
            f"message_keys={list(message.keys())}",
            flush=True,
        )
        raise RuntimeError(
            "DeepSeek returned empty final content (retryable). "
            f"finish_reason={finish_reason}; reasoning_chars={len(reasoning_content)}"
        )

    return {
        "text": str(content),
        "reasoning_text": str(reasoning_content),
        "stop_reason": finish_reason,
        "usage": usage,
        "effort": effective_effort_for_log,
    }

def llm_create_once(
    system: str | None,
    messages: list,
    max_tokens: int,
    timeout: int,
    reasoning_effort: str | None = None,
) -> dict:
    ensure_ai_config()
    return _deepseek_final_create_once(system, messages, max_tokens, timeout, reasoning_effort=reasoning_effort)


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
    max_tokens: int = LLM_MAX_OUTPUT_TOKENS,
    timeout: int = LLM_MAIN_TIMEOUT_SECONDS,
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
    max_attempts = LLM_MAX_CONTINUATIONS + 1 if allow_continuation else 1
    retry_count = max(0, LLM_API_RETRIES)
    if call_type in ("main", "main_json"):
        # Don't let the old LLM_API_RETRIES=2/3 Railway variable make manual analysis hang for 9-20 minutes.
        retry_count = min(retry_count, max(0, LLM_MAIN_RETRY_LIMIT))
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
                effective_timeout = max(30, min(timeout, LLM_RETRY_TIMEOUT_SECONDS))
                effective_reasoning_effort = DEEPSEEK_FINAL_RETRY_REASONING_EFFORT or "max"
            try:
                effort_for_log = effective_reasoning_effort or DEEPSEEK_FINAL_REASONING_EFFORT or "max"
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
                    time.sleep(max(0.0, LLM_RETRY_SLEEP_SECONDS) * (retry_idx + 1))
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


# ─── JSON model output layer ────────────────────────────────────────────────

JSON_OUTPUT_CONTRACT = r"""

TRẢ VỀ JSON NỘI BỘ BẮT BUỘC:
- Chỉ trả về 1 JSON object hợp lệ.
- Không markdown, không ```json, không giải thích ngoài JSON.
- User sẽ KHÔNG thấy JSON này; Python sẽ render lại format cũ cho Telegram.
- decision chỉ được là "LONG", "SHORT" hoặc "NO_TRADE".
- Nếu decision là LONG/SHORT: entry_low, entry_high, sl, tp1, tp2 bắt buộc là số; TP2 phải có mục tiêu cấu trúc thực sự. Nếu không bảo vệ được TP2 bằng dữ liệu hiện tại, chọn NO_TRADE thay vì bịa TP2.
- Nếu decision là NO_TRADE: entry_low, entry_high, sl, tp1, tp2 để null.
- current_price nên copy đúng từ JSON input; nếu thiếu, Python sẽ tự chèn giá hiện tại lấy từ Binance.

Schema:
{
  "symbol": "BTCUSDT",
  "mode": "SCALP hoặc SWING",
  "decision": "LONG | SHORT | NO_TRADE",
  "confidence": 55,
  "current_price": 61266.4,
  "entry_low": 61250.0,
  "entry_high": 61350.0,
  "sl": 59884.11,
  "tp1": 62064.0,
  "tp2": 62750.0,
  "risk_text": "~1,465.89 USDT",
  "activation": "Có thể vào ngay... hoặc Lệnh chờ, chưa vào ngay...",
  "risk_note": "Rủi ro chính và điều kiện hủy lệnh ngắn gọn."
}
"""


def request_json_analysis(system_prompt: str, user_prompt: str) -> str:
    """Call the model and request internal JSON. The user never sees raw JSON."""
    return create_with_continuation(
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt + JSON_OUTPUT_CONTRACT}],
        timeout=LLM_MAIN_TIMEOUT_SECONDS,
        call_type="main_json",
    )


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


def _rubric_item_score(raw: float, maximum: float) -> float:
    """Clamp a rubric item and normalize it to an integer within the allowed range.

    The model is allowed to score any integer from 0 up to the item's max, instead of being
    forced into 20% steps. If the provider returns a decimal, Python rounds it to the
    nearest integer before summing the total.
    """
    maximum = max(float(maximum), 0.0)
    if maximum <= 0:
        return 0.0
    value = min(max(float(raw), 0.0), maximum)
    return float(min(int(value + 0.5), int(maximum)))


def _rubric_total(
    breakdown: dict | None,
    weights: dict[str, float],
) -> float | None:
    """Require every item to be present, clamp each one, then sum the 0-100 total."""
    if not isinstance(breakdown, dict):
        return None
    total = 0.0
    for key, maximum in weights.items():
        raw = _num_or_none(breakdown.get(key))
        if raw is None:
            return None
        total += _rubric_item_score(float(raw), float(maximum))
    return min(max(total, 0.0), 100.0)


def _direction_multiplier(direction: str) -> int:
    return 1 if str(direction).upper() == "LONG" else -1


def _extract_signal_rubric_breakdown(output: str | None) -> dict:
    """Parser: one final model-scored rubric named SIGNAL."""
    text = output or ""
    match = re.search(
        r"\[\[TEOPARD_RUBRIC\]\]([\s\S]*?)\[\[/TEOPARD_RUBRIC\]\]",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return {}
    breakdown: dict[str, float] = {}
    aliases = {
        "huong_boi_canh_da_khung": "huong_boi_canh_da_khung",
        "huong_va_boi_canh_da_khung": "huong_boi_canh_da_khung",
        "direction_context": "huong_boi_canh_da_khung",
        "entry_timing": "entry_timing",
        "entry_va_timing": "entry_timing",
        "chat_luong_ke_hoach": "chat_luong_ke_hoach",
        "plan_quality": "chat_luong_ke_hoach",
        "sl_tp_rr": "chat_luong_ke_hoach",
        "sltp_rr": "chat_luong_ke_hoach",
        "mau_thuan_rui_ro_nhieu": "mau_thuan_rui_ro_nhieu",
        "mau_thuan_va_rui_ro_nhieu": "mau_thuan_rui_ro_nhieu",
        "contradiction_noise": "mau_thuan_rui_ro_nhieu",
        "thuc_thi_thuc_te": "thuc_thi_thuc_te",
        "execution": "thuc_thi_thuc_te",
    }
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        m = re.fullmatch(
            r"(?:SIGNAL|SCORE|FINAL)\s+([a-z0-9_]+)\s*=\s*(-?[0-9]+(?:\.[0-9]+)?)",
            line,
            flags=re.IGNORECASE,
        )
        if not m:
            continue
        raw_key = m.group(1).lower()
        key = aliases.get(raw_key, raw_key)
        if key in SIGNAL_SCORE_WEIGHTS:
            breakdown[key] = float(m.group(2))
    return breakdown


def _remove_rubric_block(output: str | None) -> str:
    return re.sub(
        r"\n?\s*\[\[TEOPARD_RUBRIC\]\][\s\S]*?\[\[/TEOPARD_RUBRIC\]\]\s*",
        "\n",
        output or "",
        flags=re.IGNORECASE,
    ).strip()


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


def _insert_public_signal_score(output: str, signal_score: float | None) -> str:
    """Insert exactly one public line: Signal Score: x/100, right below the DECISION line."""
    text = output or ""
    text = re.sub(
        r"(^\s*🏆\s*QUYẾT\s+ĐỊNH\s*:\s*(?:LONG|SHORT|NO\s+TRADE))\s*[—\-]\s*[0-9]+(?:\.[0-9]+)?\s*%\s*$",
        r"\1",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    text = re.sub(
        r"(^\s*(?:📈\s*LONG|📉\s*SHORT))\s*[—\-]\s*[0-9]+(?:\.[0-9]+)?\s*%\s*$",
        r"\1",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    text = re.sub(
        r"^\s*(?:Độ\s+mạnh\s+setup|Chất\s+lượng\s+kế\s+hoạch|Độ\s+chắc\s+chắn|Điểm\s+chắc\s+chắn|Điểm\s+tin\s+cậy\s+AI|Điểm\s+tín\s+hiệu)\s*:[^\n]*\n?",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    score_text = f"Điểm tín hiệu: {signal_score:.0f}/100" if signal_score is not None else "Điểm tín hiệu: N/A"

    lines = text.splitlines()
    for index, line in enumerate(lines):
        if re.search(r"QUYẾT\s+ĐỊNH\s*:", line, flags=re.IGNORECASE):
            lines[index + 1:index + 1] = [score_text]
            return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return "\n".join([score_text, text]).strip()


def finalize_model_scoring_output(output: str | None) -> tuple[str, dict]:
    """The model self-scores one SIGNAL rubric; Python only sums the total and hides the block."""
    raw_text = output or ""
    has_rubric_block = bool(re.search(
        r"\[\[TEOPARD_RUBRIC\]\][\s\S]*?\[\[/TEOPARD_RUBRIC\]\]",
        raw_text,
        flags=re.IGNORECASE,
    ))
    signal_breakdown = _extract_signal_rubric_breakdown(raw_text)
    signal_score = _rubric_total(signal_breakdown, SIGNAL_SCORE_WEIGHTS)
    if signal_score is None and not has_rubric_block:
        signal_score = _extract_legacy_confidence(raw_text)

    clean = _remove_rubric_block(raw_text)
    clean = _insert_public_signal_score(clean, signal_score)
    return clean, {
        "signal_score": signal_score,
        "confidence": signal_score,
        "signal_score_breakdown": signal_breakdown,
    }


def _clean_decision(value: str | None) -> str:
    raw = str(value or "").upper().replace("-", "_").replace(" ", "_")
    if raw in {"NO_TRADE", "NOTRADE", "NO__TRADE", "KHONG_VAO_LENH", "KHÔNG_VÀO_LỆNH"}:
        return "NO_TRADE"
    if raw in {"LONG", "SHORT"}:
        return raw
    return "WAIT"


def parse_prediction_from_json_payload(payload: dict | None) -> dict:
    if not isinstance(payload, dict):
        return {"direction": "WAIT", "confidence": None, "entry_low": None, "entry_high": None, "sl": None, "tp1": None, "tp2": None}
    decision = _clean_decision(payload.get("decision"))
    signal_score = _rubric_total(payload.get("signal_score_breakdown"), SIGNAL_SCORE_WEIGHTS)
    if signal_score is None:
        signal_score = _num_or_none(payload.get("signal_score"))
    if signal_score is None:
        signal_score = _num_or_none(payload.get("confidence"))
    setup_strength = None
    confidence = signal_score
    return {
        "direction": decision,
        "signal_score": signal_score,
        "setup_strength": setup_strength,
        "confidence": confidence,
        "entry_low": _num_or_none(payload.get("entry_low")),
        "entry_high": _num_or_none(payload.get("entry_high")),
        "sl": _num_or_none(payload.get("sl")),
        "tp1": _num_or_none(payload.get("tp1")),
        "tp2": _num_or_none(payload.get("tp2")),
    }


def render_user_output_from_json_payload(payload: dict, fallback_symbol: str, mode: str, fallback_current_price: float | None = None) -> str:
    """Render the internal JSON into the old text format so the change is invisible to the user."""
    mode_label = "SCALP" if mode == "short" else "SWING"
    symbol = str(payload.get("symbol") or fallback_symbol).upper()
    decision = _clean_decision(payload.get("decision"))
    signal_score = _rubric_total(payload.get("signal_score_breakdown"), SIGNAL_SCORE_WEIGHTS)
    if signal_score is None:
        signal_score = _num_or_none(payload.get("signal_score"))
    if signal_score is None:
        signal_score = _num_or_none(payload.get("confidence"))
    setup_strength = None
    confidence = signal_score
    current_price = _num_or_none(payload.get("current_price"))
    if current_price is None:
        current_price = fallback_current_price
    current_price_line = f"Giá hiện tại: {fmt(current_price)} USDT" if current_price is not None else "Giá hiện tại: N/A"

    activation = str(payload.get("activation") or "").strip()
    risk_note = str(payload.get("risk_note") or "").strip()
    risk_text = str(payload.get("risk_text") or "").strip()

    lines = [
        f"🎯 {symbol} — {mode_label}",
        f"🏆 QUYẾT ĐỊNH: {decision.replace('_', ' ')}",
        f"Điểm tín hiệu: {confidence:.0f}/100" if confidence is not None else "Điểm tín hiệu: N/A",
        current_price_line,
    ]

    if decision in ("LONG", "SHORT"):
        emoji = "📈" if decision == "LONG" else "📉"
        entry_low = _num_or_none(payload.get("entry_low"))
        entry_high = _num_or_none(payload.get("entry_high"))
        sl = _num_or_none(payload.get("sl"))
        tp1 = _num_or_none(payload.get("tp1"))
        tp2 = _num_or_none(payload.get("tp2"))
        lines += [
            "",
            f"{emoji} {decision}",
            f"Entry: {fmt(entry_low)}–{fmt(entry_high)}",
            f"SL: {fmt(sl)}",
            f"TP1: {fmt(tp1)}",
            f"TP2: {fmt(tp2)}",
        ]
        if risk_text:
            lines.append(f"Rủi ro mỗi lệnh: {risk_text}")
        if activation:
            lines.append(f"Kích hoạt: {activation}")
    else:
        if activation:
            lines += ["", f"Kích hoạt: {activation}"]

    if risk_note:
        lines += ["", f"⚠️ Rủi ro: {risk_note}"]

    return sanitize_user_output("\n".join(lines).strip())


def model_output_to_user_text_and_pred(raw_output: str, symbol: str, mode: str, current_price: float | None = None) -> tuple[str, dict, dict | None]:
    """Prefer parsing JSON; if that fails, fall back to the old regex text parser so the bot doesn't crash."""
    payload = _extract_json_object(raw_output)
    if payload is not None:
        user_text = render_user_output_from_json_payload(payload, symbol, mode, fallback_current_price=current_price)
        pred = parse_prediction_from_json_payload(payload)
        return user_text, pred, payload
    user_text = sanitize_user_output(raw_output)
    return user_text, parse_prediction_from_output(user_text), None

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
    m = re.search(r"QUYẾT ĐỊNH[:\s]+(LONG|SHORT|NO[_\s-]?TRADE|KHÔNG\s+VÀO\s+LỆNH|KHONG\s+VAO\s+LENH)", output, re.IGNORECASE)
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
    em = re.search(r"Entry[:\s]+([0-9,\.]+)(?:\s*[–\-]\s*([0-9,\.]+))?", selected_output, re.IGNORECASE)
    if em:
        try:
            entry_low  = float(em.group(1).replace(",", ""))
            entry_high = float(em.group(2).replace(",", "")) if em.group(2) else entry_low
        except Exception:
            pass

    sl  = find_price([r"SL[:\s]+([0-9,\.]+)"], selected_output)
    tp1 = find_price([r"TP1[:\s]+([0-9,\.]+)"], selected_output)
    tp2 = find_price([r"TP2[:\s]+([0-9,\.]+)"], selected_output)

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

    if "entry_low" in values and "entry_high" in values and values["entry_low"] > values["entry_high"]:
        errors.append("Entry thấp lớn hơn Entry cao.")

    return errors


def _guarded_no_trade_output(
    symbol: str,
    mode: str,
    current_price: float | None,
    errors: list[str],
    pred: dict | None = None,
    timeframe_data: dict[str, pd.DataFrame | None] | None = None,
) -> str:
    """Render a NO TRADE caused by the Python guard, while still keeping the direction the model preferred.

    The DECISION is still NO TRADE because the trade failed the guard. However, the user should still
    see whether the original plan leaned LONG or SHORT, while distinguishing that trade direction from
    the structural trend of the confirmation timeframe.
    """
    mode_label = "SCALP" if mode == "short" else "SWING"
    price_text = f" Giá hiện tại {fmt(current_price)} USDT." if current_price is not None else ""
    reason = errors[0] if errors else "Kế hoạch LONG/SHORT bị bộ lọc rủi ro từ chối."
    pred_data = pred or {}
    signal_score = _num_or_none(pred_data.get("signal_score"))
    if signal_score is None:
        signal_score = _num_or_none(pred_data.get("confidence"))
    signal_text = f"{signal_score:.0f}/100" if signal_score is not None else "N/A"

    rejected_direction = str(pred_data.get("direction") or "").upper()
    direction_line = ""
    if rejected_direction in ("LONG", "SHORT"):
        direction_emoji = "📈" if rejected_direction == "LONG" else "📉"
        direction_line = f"Hướng ưu tiên bị từ chối: {rejected_direction} {direction_emoji}\n"

    structure_line = ""
    if timeframe_data:
        _main_label, structure_label, _big_label = _mode_labels(mode)
        structure = _structure_info(timeframe_data.get(structure_label), current_price)
        structure_trend = str(structure.get("trend") or "").upper()
        if structure_trend in ("TĂNG", "GIẢM", "ĐI NGANG"):
            structure_line = f"Xu hướng cấu trúc ({structure_label}): {structure_trend}\n"

    return sanitize_user_output(
        f"🎯 {symbol} — {mode_label}\n"
        f"🏆 QUYẾT ĐỊNH: NO TRADE\n"
        f"{direction_line}"
        f"{structure_line}"
        f"Điểm tín hiệu: {signal_text}\n"
        f"Giá hiện tại: {fmt(current_price)} USDT\n"
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
    text = re.sub(r"\brisk\b", "rủi ro", text, flags=re.IGNORECASE)
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
    price_line = f"Giá hiện tại: {fmt(current_price)} USDT" if current_price is not None else "Giá hiện tại: N/A"
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


def build_no_trade_summary(output: str) -> str:
    text = (output or "").strip().replace("\n", " ")
    if not text:
        return "Claude chọn NO TRADE nhưng không có lý do rõ."
    return "NO TRADE: " + text[:600]


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


def load_prefilter_system_prompt() -> str:
    return _load_prompt_file("prefilter_system_prompt.txt")


def load_reviewer_system_prompt() -> str:
    return _load_prompt_file("reviewer_system_prompt.txt")


def load_timeframe_data(binance_symbol: str, interval: str, limit: int) -> pd.DataFrame | None:
    """Sync helper: fetch Binance candles then calculate indicators."""
    return add_indicators(get_binance_klines(binance_symbol, interval, limit))


def request_claude_analysis(system_prompt: str, user_prompt: str) -> str:
    """Sync helper: calls the main model; output is short so there's no continuation."""
    max_tokens = max(800, min(LLM_MAX_OUTPUT_TOKENS, LLM_MAIN_OUTPUT_TOKEN_CAP))
    return create_with_continuation(
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=max_tokens,
        timeout=LLM_MAIN_TIMEOUT_SECONDS,
        allow_continuation=False,
        call_type="main",
    )


# ─── Objective market packet + independent Flash reviewer ─────────────────

def _mode_frame_roles(mode: str) -> tuple[str, str, str, str]:
    """Return timing, setup/plan, trend/structure, macro labels."""
    if mode == "short":
        return "15M", "1H", "4H", "1D"
    return "1H", "4H", "1D", "1W"

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
    # The last Binance row is usually the still-running candle.
    return df.iloc[:-1].copy() if len(df) >= 2 else df.copy()


def _v50_raw_limit(mode: str, label: str) -> int:
    # Raw history is long enough for the model to see structure instead of just the nearest high/low.
    # SCALP: 1 day of 15M, 7 days of 1H, 20 days of 4H, 60 days of 1D.
    # SWING: 4 days of 1H, 28 days of 4H, 120 days of 1D, 1 year of 1W.
    limits = {
        "short": {"15M": 96, "1H": 168, "4H": 120, "1D": 60},
        "long": {"1H": 96, "4H": 168, "1D": 120, "1W": 52},
    }
    return limits.get(mode, {}).get(label, 16)


def _v50_raw_candles(label: str, df: pd.DataFrame | None, mode: str) -> str:
    closed = _v50_closed_df(df)
    if closed is None or closed.empty:
        return f"{label}: N/A"
    rows = closed.tail(_v50_raw_limit(mode, label))
    out = [f"{label} — {len(rows)} nến đã đóng gần nhất (time,O,H,L,C,V):"]
    for _, row in rows.iterrows():
        out.append(
            f"{_v50_time_value(row)} | "
            f"{fmt(_safe_float(row.get('open')))} | {fmt(_safe_float(row.get('high')))} | "
            f"{fmt(_safe_float(row.get('low')))} | {fmt(_safe_float(row.get('close')))} | "
            f"{fmt(_safe_float(row.get('volume')))}"
        )
    return "\n".join(out)


def _v50_pivots(df: pd.DataFrame | None, lookback: int = 80, wing: int = 2) -> list[dict]:
    closed = _v50_closed_df(df)
    if closed is None or len(closed) < wing * 2 + 3:
        return []
    sample = closed.tail(lookback)
    highs = pd.to_numeric(sample["high"], errors="coerce").to_numpy()
    lows = pd.to_numeric(sample["low"], errors="coerce").to_numpy()
    rows = list(sample.iterrows())
    pivots: list[dict] = []
    for i in range(wing, len(sample) - wing):
        if np.isfinite(highs[i]) and highs[i] >= np.nanmax(highs[i-wing:i+wing+1]):
            ts = _v50_timestamp_value(rows[i][1])
            pivots.append({"type": "HIGH", "price": float(highs[i]), "time": _v50_time_value(rows[i][1]), "time_utc": ts.isoformat() if ts is not None else None, "index": i})
        if np.isfinite(lows[i]) and lows[i] <= np.nanmin(lows[i-wing:i+wing+1]):
            ts = _v50_timestamp_value(rows[i][1])
            pivots.append({"type": "LOW", "price": float(lows[i]), "time": _v50_time_value(rows[i][1]), "time_utc": ts.isoformat() if ts is not None else None, "index": i})
    return pivots[-12:]


def _v50_pivot_followup(df: pd.DataFrame | None, pivot: dict) -> dict:
    """Objective stats around the exact pivot price, without constructing an artificial % zone."""
    closed = _v50_closed_df(df)
    if closed is None or closed.empty:
        return {}
    price = float(pivot["price"])
    post = closed.copy()
    try:
        pivot_time = pd.to_datetime(pivot.get("time_utc"), utc=True)
        time_values = pd.to_datetime(post.get("open_time", post.index), utc=True, errors="coerce")
        post = post.loc[time_values > pivot_time]
    except Exception:
        pass
    wick_tests = closes_beyond = 0
    last_test = None
    for _, row in post.iterrows():
        low = _safe_float(row.get("low"))
        high = _safe_float(row.get("high"))
        close = _safe_float(row.get("close"))
        if None in (low, high, close):
            continue
        if low <= price <= high:
            wick_tests += 1
            last_test = _v50_time_value(row)
        if pivot["type"] == "LOW" and close < price:
            closes_beyond += 1
        elif pivot["type"] == "HIGH" and close > price:
            closes_beyond += 1
    return {
        "wick_tests": wick_tests,
        "closes_beyond": closes_beyond,
        "last_test": last_test or "N/A",
    }


def _v50_swing_zone_block(label: str, df: pd.DataFrame | None) -> str:
    """An objective swing sequence; no +/-% zones are constructed and no fresh/tested/weakened labels are attached."""
    pivots = _v50_pivots(df, lookback=240 if label in {"1H", "4H"} else 180, wing=2 if label in {"15M", "1H"} else 3)
    if not pivots:
        return f"{label}: không đủ pivot khách quan."
    highs = [p for p in pivots if p["type"] == "HIGH"][-8:]
    lows = [p for p in pivots if p["type"] == "LOW"][-8:]
    lines = [
        f"{label} chuỗi swing khách quan (chỉ là pivot có timestamp, không phải vùng Entry/SL/TP bắt buộc):",
        "- Swing highs theo thời gian: " + (" → ".join(f"{fmt(p['price'])} ({p['time']})" for p in highs) if highs else "N/A"),
        "- Swing lows theo thời gian: " + (" → ".join(f"{fmt(p['price'])} ({p['time']})" for p in lows) if lows else "N/A"),
        "- Kiểm tra sau pivot (đúng giá pivot, không dùng tolerance phần trăm):",
    ]
    for pivot in pivots[-10:]:
        stats = _v50_pivot_followup(df, pivot)
        lines.append(
            f"  • {pivot['type']} {fmt(pivot['price'])} tại {pivot['time']}; "
            f"wick-test={stats.get('wick_tests', 0)}, close-vượt={stats.get('closes_beyond', 0)}, "
            f"lần kiểm tra gần nhất={stats.get('last_test', 'N/A')}."
        )
    return "\n".join(lines)


def _v50_live_line(label: str, df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return f"{label} live: N/A"
    row = df.iloc[-1]
    progress = None
    try:
        progress = (_live_candle_progress(row, label) * 100.0) if _live_candle_progress(row, label) is not None else None
    except Exception:
        progress = None
    return (
        f"{label} live ({fmt(progress, 1) if progress is not None else 'N/A'}%): "
        f"time={_v50_time_value(row)}, O={fmt(_safe_float(row.get('open')))}, "
        f"H={fmt(_safe_float(row.get('high')))}, L={fmt(_safe_float(row.get('low')))}, "
        f"C={fmt(_safe_float(row.get('close')))}, V={fmt(_safe_float(row.get('volume')))}. "
        "Đây là nến đang chạy, không phải xác nhận đóng nến."
    )


def build_feature_engineering_block(
    timeframe_data: dict[str, pd.DataFrame | None],
    mode: str,
    current_price: float | None,
) -> str:
    """Build the objective packet used by both Manual and Auto Scan planner.

    Every timeframe receives the same categories of evidence. The timing frame
    uses a compact pivot sequence so it informs trigger quality without
    overwhelming the setup/trend structure.
    """
    trigger, setup, trend, big = _mode_frame_roles(mode)
    labels = [trigger, setup, trend, big]
    lines = [
        "OBJECTIVE_MARKET_PACKET",
        "Múi giờ của mọi timestamp trong packet: giờ Việt Nam (UTC+7), hậu tố VN.",
        f"Giá hiện tại: {fmt(current_price)}",
        "Python chỉ chuẩn bị dữ kiện khách quan; không kết luận hướng và không dựng Entry/SL/TP.",
        "Packet không có ATR, Fibonacci, market-regime label hay trend label.",
    ]
    for label in labels:
        df = timeframe_data.get(label)
        if df is None or df.empty:
            lines.append(f"{label}: không có dữ liệu.")
            continue
        row = _analysis_row(df)
        if row is not None:
            lines.append(
                f"{label} chỉ báo nến đóng gần nhất: close={fmt(_safe_float(row.get('close')))}, "
                f"EMA7={fmt(_safe_float(row.get('ema_7')))}, EMA25={fmt(_safe_float(row.get('ema_25')))}, "
                f"EMA50={fmt(_safe_float(row.get('ema_50')))}, RSI14={fmt(_safe_float(row.get('rsi_14')),1)}, "
                f"MACD line={fmt(_safe_float(row.get('macd_line')))}, signal={fmt(_safe_float(row.get('macd_signal')))}, "
                f"histogram={fmt(_safe_float(row.get('macd_hist')))}, volume={fmt(_safe_float(row.get('volume')))}, "
                f"takerBuy={fmt(_taker_buy_ratio(row),1)}%."
            )
        if ANALYSIS_DATA_VARIANT in {"B", "C"}:
            if label == trigger:
                pivots = _v50_pivots(df, lookback=96, wing=2)
                highs = [x for x in pivots if x.get("type") == "HIGH"][-4:]
                lows = [x for x in pivots if x.get("type") == "LOW"][-4:]
                lines.append(
                    f"{label} swing timing rút gọn: highs="
                    + (" → ".join(f"{fmt(x['price'])} ({x['time']})" for x in highs) if highs else "N/A")
                    + "; lows="
                    + (" → ".join(f"{fmt(x['price'])} ({x['time']})" for x in lows) if lows else "N/A")
                    + "."
                )
            else:
                lines.append(_v50_swing_zone_block(label, df))
    return "\n".join(lines)


def build_feature_snapshot(
    timeframe_data: dict[str, pd.DataFrame | None],
    mode: str,
    current_price: float | None,
) -> str:
    """Compact but structure-aware packet for Flash prefilter.

    It includes several recent closed candles and compact swing sequences for
    the trend/setup frames, allowing the rubric's structure score to be based
    on more than one candle and EMA values.
    """
    trigger, setup, trend, big = _mode_frame_roles(mode)
    lines = [f"Mode={'SCALP' if mode == 'short' else 'SWING'}; price={fmt(current_price)}"]
    # Increases structural coverage for the prefilter while keeping the packet lighter than Pro.
    # SCALP: timing 12, setup 24, trend 16, macro 6.
    # SWING uses the same allocation, mapped to the corresponding roles.
    recent_counts = {trigger: 12, setup: 24, trend: 16, big: 6}
    for label in (trend, setup, trigger, big):
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
            f"RSI14={fmt(_safe_float(row.get('rsi_14')),1)},MACDline={fmt(_safe_float(row.get('macd_line')))},"
            f"signal={fmt(_safe_float(row.get('macd_signal')))},hist={fmt(_safe_float(row.get('macd_hist')))},"
            f"V={fmt(_safe_float(row.get('volume')))},takerBuy={fmt(_taker_buy_ratio(row),1)}%"
        )
        if closed is not None and not closed.empty:
            compact=[]
            for _, candle in closed.tail(recent_counts[label]).iterrows():
                compact.append(
                    f"{_v50_time_value(candle)} O={fmt(_safe_float(candle.get('open')))} "
                    f"H={fmt(_safe_float(candle.get('high')))} L={fmt(_safe_float(candle.get('low')))} "
                    f"C={fmt(_safe_float(candle.get('close')))} V={fmt(_safe_float(candle.get('volume')))}"
                )
            lines.append(f"{label} recent closed ({len(compact)}): " + " || ".join(compact))
        # Every timeframe gets a compact objective swing sequence.
        # Setup/trend get a deeper look to reduce the chance of missing older structure zones;
        # timing/macro stay compact to keep prefilter cost reasonable.
        if label == setup:
            pivot_count, pivot_lookback = 6, 240
        elif label == trend:
            pivot_count, pivot_lookback = 5, 220
        elif label == trigger:
            pivot_count, pivot_lookback = 4, 144
        else:
            pivot_count, pivot_lookback = 3, 120
        pivot_wing = 2 if label in {trigger, setup} else 3
        pivots = _v50_pivots(df, lookback=pivot_lookback, wing=pivot_wing)
        highs = [x for x in pivots if x.get('type') == 'HIGH'][-pivot_count:]
        lows = [x for x in pivots if x.get('type') == 'LOW'][-pivot_count:]
        lines.append(
            f"{label} recent swings: highs="
            + (" → ".join(f"{fmt(x['price'])} ({x['time']})" for x in highs) if highs else "N/A")
            + "; lows="
            + (" → ".join(f"{fmt(x['price'])} ({x['time']})" for x in lows) if lows else "N/A")
        )
    return "\n".join(lines)


def build_synchronized_decision_snapshot(
    timeframe_data: dict[str, pd.DataFrame | None],
    mode: str,
    current_price: float | None,
) -> str:
    trigger, setup, trend, big = _mode_frame_roles(mode)
    lines = ["SYNCHRONIZED_DECISION_SNAPSHOT", "Mọi timestamp bên dưới dùng giờ Việt Nam (UTC+7), hậu tố VN."]
    lines.append(f"Roles: timing={trigger}; setup/plan={setup}; trend/structure={trend}; macro={big}.")
    lines.append(_v50_live_line(setup, timeframe_data.get(setup)))
    lines.append(_v50_live_line(trend, timeframe_data.get(trend)))
    lines.append(_v50_live_line(trigger, timeframe_data.get(trigger)))
    lines.append(_v50_live_line(big, timeframe_data.get(big)))
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
) -> str:
    """Data-first planner prompt; analytical rules live only in system prompt."""
    mode_label = "SCALP" if mode == "short" else "SWING"
    trigger, setup, trend, big = _mode_frame_roles(mode)
    raw_sections = [_v50_raw_candles(label, timeframe_data.get(label), mode) for label in (trigger, setup, trend, big)]
    return "\n".join([
        f"PHÂN TÍCH {symbol} — {mode_label}",
        f"Thời điểm tạo packet: {utc_now().astimezone(VN_TZ).strftime('%Y-%m-%d %H:%M:%S VN')}",
        current_price_str,
        f"Vai trò khung: {trend}=hướng/cấu trúc; {setup}=setup và Entry/SL/TP; {trigger}=timing; {big}=bối cảnh lớn.",
        "Không có kế hoạch đang mở, Fear & Greed, ATR, Fibonacci hoặc hướng ưu tiên.",
        "",
        feature_block or "OBJECTIVE_MARKET_PACKET: N/A",
        "",
        decision_snapshot or "LIVE SNAPSHOT: N/A",
        "",
        "RAW OHLCV:",
        "\n\n".join(raw_sections),
        "",
        "Tuân thủ toàn bộ quy trình phân tích và tự phản biện trong system prompt. Chỉ xuất bản FINAL; không xuất bản nháp hoặc reasoning nội bộ.",
        "",
        "OUTPUT PUBLIC:",
        f"🎯 {symbol} — {mode_label}",
        "🏆 QUYẾT ĐỊNH: LONG | SHORT | NO TRADE",
        "Trạng thái: READY_TO_ENTER | SETUP_WAITING_TRIGGER | NO_TRADE",
        "Giá hiện tại: ... USDT",
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
        "Không tự in Điểm tín hiệu; Flash reviewer sẽ chấm độc lập.",
    ])


def _extract_setup_status(output: str | None) -> str:
    """Read an explicit planner status; never infer READY_TO_ENTER from prose."""
    text = output or ""
    m = re.search(r"Trạng\s*thái\s*:\s*(READY_TO_ENTER|SETUP_WAITING_TRIGGER|NO_TRADE)", text, flags=re.I)
    if m:
        return m.group(1).upper()
    return "STATUS_PARSE_ERROR"


def _reviewer_json_candidates(raw: str) -> list[str]:
    """Return likely JSON snippets from a reviewer response."""
    candidates: list[str] = []
    text = (raw or "").strip()
    if not text:
        return candidates
    candidates.append(text)
    for match in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.I | re.S):
        candidates.append(match.group(1).strip())
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first:last + 1].strip())
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(candidates))


def _normalize_reviewer_verdict(value) -> str | None:
    text = str(value or "").strip().upper()
    if text in {"APPROVE", "APPROVED", "ACCEPT", "ACCEPTED", "PASS", "PASSED", "CHẤP NHẬN", "CHAP NHAN", "ĐẠT", "DAT"}:
        return "APPROVE"
    if text in {"REJECT", "REJECTED", "DENY", "DENIED", "FAIL", "FAILED", "TỪ CHỐI", "TU CHOI", "KHÔNG ĐẠT", "KHONG DAT"}:
        return "REJECT"
    return None


def _reviewer_score_value(value) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            match = re.search(r"-?[0-9]+(?:\.[0-9]+)?", value.replace(",", "."))
            if not match:
                return None
            value = match.group(0)
        return min(100.0, max(0.0, float(value)))
    except Exception:
        return None


def _parse_reviewer_output(text: str | None) -> dict:
    """Parse reviewer output without asking Python to evaluate the trade.

    Accepted forms include JSON, SCORE/VERDICT/REASON lines, Vietnamese labels,
    light Markdown, bullets, ``67/100`` and compact one-line responses.
    """
    raw = str(text or "").strip()
    score = None
    verdict = None
    reason = ""
    parsed_format = None
    validity = {}

    # 1) Prefer structured JSON when available.
    for candidate in _reviewer_json_candidates(raw):
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        lowered = {str(k).strip().lower(): v for k, v in payload.items()}
        score = _reviewer_score_value(
            lowered.get("score", lowered.get("point", lowered.get("diem", lowered.get("điểm"))))
        )
        verdict = _normalize_reviewer_verdict(
            lowered.get("verdict", lowered.get("decision", lowered.get("ket_luan", lowered.get("kết luận"))))
        )
        reason_value = lowered.get("reason", lowered.get("ly_do", lowered.get("lý do", lowered.get("comment"))))
        reason = str(reason_value or "").strip()
        for key in ("direction_valid", "entry_valid", "sl_valid", "tp_valid", "trigger_valid", "status_valid"):
            value = lowered.get(key)
            if isinstance(value, bool):
                validity[key] = value
            elif isinstance(value, str) and value.strip().lower() in {"true", "false"}:
                validity[key] = value.strip().lower() == "true"
        if score is not None or verdict is not None or reason:
            parsed_format = "json"
            break

    # 2) Flexible text parser. It is intentionally not anchored to line starts.
    parse_text = raw.replace("**", "").replace("__", "").replace("`", "")
    if score is None:
        score_patterns = [
            r"(?i)(?:SCORE|FINAL\s*SCORE|REVIEW(?:ER)?\s*SCORE|ĐIỂM(?:\s*ĐÁNH\s*GIÁ)?|DIEM(?:\s*DANH\s*GIA)?)\s*[:=\-]\s*([0-9]+(?:[\.,][0-9]+)?)\s*(?:/\s*100)?",
            r"(?i)\b([0-9]+(?:[\.,][0-9]+)?)\s*/\s*100\b",
        ]
        for pattern in score_patterns:
            match = re.search(pattern, parse_text)
            if match:
                score = _reviewer_score_value(match.group(1))
                if score is not None:
                    parsed_format = parsed_format or "text"
                    break

    if verdict is None:
        verdict_patterns = [
            r"(?i)(?:VERDICT|KẾT\s*LUẬN|KET\s*LUAN|DECISION)\s*[:=\-]\s*([^\n;,]+)",
            r"(?i)\b(APPROVE(?:D)?|REJECT(?:ED)?|ACCEPT(?:ED)?|PASS(?:ED)?|FAIL(?:ED)?|CHẤP\s*NHẬN|CHAP\s*NHAN|TỪ\s*CHỐI|TU\s*CHOI|KHÔNG\s*ĐẠT|KHONG\s*DAT)\b",
        ]
        for pattern in verdict_patterns:
            match = re.search(pattern, parse_text)
            if match:
                verdict = _normalize_reviewer_verdict(match.group(1))
                if verdict:
                    parsed_format = parsed_format or "text"
                    break

    if not reason:
        reason_match = re.search(
            r"(?is)(?:REASON|NHẬN\s*XÉT|NHAN\s*XET|LÝ\s*DO|LY\s*DO|COMMENT)\s*[:=\-]\s*(.+?)(?=\n\s*(?:SCORE|VERDICT|KẾT\s*LUẬN|KET\s*LUAN|ĐIỂM|DIEM)\s*[:=\-]|\Z)",
            parse_text,
        )
        if reason_match:
            reason = " ".join(reason_match.group(1).strip().split())
            parsed_format = parsed_format or "text"

    # Verdict may be inferred solely from the reviewer's own score. Python is
    # not assessing market quality here; it only applies the configured gate.
    if verdict is None and score is not None:
        verdict = "APPROVE" if score >= FINAL_REVIEW_MIN_SIGNAL_SCORE else "REJECT"
    if verdict is None:
        verdict = "REJECT"

    return {
        "score": score,
        "verdict": verdict,
        "breakdown": {},
        **validity,
        "reason": reason,
        "parse_ok": score is not None,
        "parsed_format": parsed_format,
    }


def _reviewer_format_repair(raw_output: str) -> dict:
    """Ask Flash to reformat an existing answer; no market re-analysis."""
    raw = (raw_output or "").strip()
    if not raw:
        return {"score": None, "verdict": "REJECT", "reason": "", "parse_ok": False, "raw": ""}
    repair_prompt = "\n".join([
        "Chỉ định dạng lại kết quả reviewer bên dưới. Không phân tích lại thị trường, không đổi điểm hoặc kết luận.",
        "Trả đúng một JSON object hợp lệ, không markdown:",
        '{"score": 0, "verdict": "REJECT", "direction_valid": false, "entry_valid": false, "sl_valid": false, "tp_valid": false, "trigger_valid": false, "status_valid": false, "reason": "..."}',
        "score phải là số 0..100; verdict chỉ APPROVE hoặc REJECT.",
        "reason bắt buộc viết bằng tiếng Việt; nếu reason gốc là tiếng Anh thì dịch sang tiếng Việt nhưng không đổi ý.",
        "Nếu nội dung gốc không có điểm rõ ràng, dùng score=null và verdict=REJECT.",
        "",
        "NỘI DUNG GỐC:",
        raw[:12000],
    ])
    result = _deepseek_create_once(
        system="Bạn là bộ sửa định dạng JSON. Chỉ định dạng và dịch reason sang tiếng Việt; không phân tích lại hoặc đổi điểm/kết luận.",
        messages=[{"role": "user", "content": repair_prompt}],
        timeout=DEEPSEEK_TIMEOUT_SECONDS,
        model=DEEPSEEK_REVIEW_MODEL,
        max_tokens=min(3000, max(1200, DEEPSEEK_REVIEW_MAX_OUTPUT_TOKENS)),
        temperature=0,
        response_format={"type": "json_object"},
        reasoning_effort="off",
    )
    repair_raw = (result.get("text") or result.get("reasoning_text") or "").strip()
    parsed = _parse_reviewer_output(repair_raw)
    parsed["raw"] = repair_raw
    return parsed


def review_trade_plan_with_flash(
    market_packet: str,
    planner_output: str,
    mode: str,
    minimum_score: float,
) -> dict:
    """Flash independently reviews the immutable planner plan and self-scores it."""
    system_prompt = load_reviewer_system_prompt()
    prompt = "\n".join([
        f"MODE: {'SCALP' if mode == 'short' else 'SWING'}",
        f"NGƯỠNG THỰC THI CỦA PIPELINE: {float(minimum_score):g}/100.",
        "APPROVE chỉ khi score đạt ngưỡng trên và không có lỗi nghiêm trọng ở direction, Entry, SL, TP, trigger hoặc trạng thái.",
        "",
        "MARKET PACKET:",
        market_packet,
        "",
        "PLANNER OUTPUT — NGUYÊN VĂN:",
        planner_output,
    ])
    result = _openrouter_reviewer_create_once(
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
        timeout=OPENROUTER_REVIEWER_TIMEOUT_SECONDS,
        model=OPENROUTER_REVIEWER_MODEL,
        max_tokens=max(4000, OPENROUTER_REVIEWER_MAX_OUTPUT_TOKENS),
        temperature=OPENROUTER_REVIEWER_TEMPERATURE,
        response_format={"type": "json_object"},
    )
    content_raw = (result.get("text") or "").strip()
    reasoning_raw = (result.get("reasoning_text") or "").strip()
    primary_raw = content_raw or reasoning_raw
    parsed = _parse_reviewer_output(primary_raw)
    repair_raw = ""

    if not parsed.get("parse_ok"):
        if primary_raw:
            # Non-empty output with invalid JSON: repair format only.
            repaired = _reviewer_format_repair(primary_raw)
            repair_raw = repaired.get("raw") or ""
            if repaired.get("parse_ok"):
                parsed = repaired
        else:
            # Retry when the provider returned no usable text.
            retry_prompt = prompt + "\n\nIMPORTANT: Output only the final JSON now."
            retry_result = _openrouter_reviewer_create_once(
                system=system_prompt,
                messages=[{"role": "user", "content": retry_prompt}],
                timeout=OPENROUTER_REVIEWER_TIMEOUT_SECONDS,
                model=OPENROUTER_REVIEWER_MODEL,
                max_tokens=max(6000, OPENROUTER_REVIEWER_MAX_OUTPUT_TOKENS),
                temperature=OPENROUTER_REVIEWER_TEMPERATURE,
                response_format={"type": "json_object"},
            )
            retry_content = (retry_result.get("text") or "").strip()
            retry_reasoning = (retry_result.get("reasoning_text") or "").strip()
            retry_raw = retry_content or retry_reasoning
            retry_parsed = _parse_reviewer_output(retry_raw)
            if retry_parsed.get("parse_ok"):
                parsed = retry_parsed
                primary_raw = retry_raw
                content_raw = retry_content
                reasoning_raw = retry_reasoning
            elif retry_raw:
                repaired = _reviewer_format_repair(retry_raw)
                repair_raw = repaired.get("raw") or ""
                if repaired.get("parse_ok"):
                    parsed = repaired

    parsed["raw"] = primary_raw
    parsed["raw_content"] = content_raw
    parsed["raw_reasoning"] = reasoning_raw
    parsed["repair_raw"] = repair_raw
    parsed["empty_response"] = not bool(primary_raw)
    if not parsed.get("parse_ok"):
        parsed["verdict"] = "REJECT"
        if not parsed.get("reason"):
            parsed["reason"] = "Flash reviewer trả response rỗng." if not primary_raw else "Không đọc được điểm reviewer sau một lần sửa định dạng."
    return parsed


def _apply_reviewer_score(output: str, review: dict) -> str:
    """Attach the reviewer's actual score to the plan, even when the reviewer's verdict is REJECT.

    The verdict and threshold decide pass/fail; the real score must never be replaced with 0.
    """
    clean = _remove_rubric_block(output or "")
    return _insert_public_signal_score(clean, review.get("score"))


def _review_passed(review: dict, minimum_score: float) -> bool:
    """Gate by reviewer verdict, score, and all required validity checks."""
    score = review.get("score")
    required_flags = (
        "direction_valid",
        "entry_valid",
        "sl_valid",
        "tp_valid",
        "trigger_valid",
        "status_valid",
    )
    flags_ok = all(review.get(key) is True for key in required_flags)
    return (
        review.get("verdict") == "APPROVE"
        and score is not None
        and float(score) >= float(minimum_score)
        and flags_ok
    )


def _review_breakdown_text(review: dict) -> str:
    breakdown = review.get("breakdown") or {}
    labels = [
        ("THESIS", "Luận điểm đa khung"),
        ("SETUP", "Cấu trúc setup"),
        ("ENTRY", "Bằng chứng Entry"),
        ("SL", "Điểm vô hiệu/SL"),
        ("TARGET", "Bằng chứng mục tiêu"),
        ("TRIGGER", "Trigger/timing"),
    ]
    parts = []
    for key, label in labels:
        if key in breakdown:
            cap = {"THESIS":20,"SETUP":20,"ENTRY":20,"SL":15,"TARGET":15,"TRIGGER":10}[key]
            parts.append(f"- {label}: {float(breakdown[key]):g}/{cap}")
    return "\n".join(parts)


def _manual_review_rejection_output(
    symbol: str, mode: str, current_price: float, planner_pred: dict, review: dict, minimum_score: float
) -> str:
    mode_label = "SCALP" if mode == "short" else "SWING"
    direction = _clean_decision(planner_pred.get("direction"))
    score = review.get("score")
    score_text = f"{float(score):g}/100" if score is not None else "Không đọc được"
    verdict = review.get("verdict") or "REJECT"
    reason = review.get("reason") or (
        "Flash reviewer trả response rỗng." if review.get("empty_response")
        else "Không đọc được điểm reviewer sau một lần sửa định dạng." if score is None
        else "Kế hoạch chưa được dữ liệu hỗ trợ đủ."
    )
    breakdown = _review_breakdown_text(review)
    breakdown_block = f"\n\nChi tiết chấm điểm:\n{breakdown}" if breakdown else ""
    direction_line = f"Hướng planner đề xuất: {direction} {'📈' if direction == 'LONG' else '📉' if direction == 'SHORT' else ''}\n"
    return sanitize_user_output(
        f"🎯 {symbol} — {mode_label}\n"
        f"🏆 QUYẾT ĐỊNH: NO TRADE\n"
        f"{direction_line}"
        f"Giá hiện tại: {fmt(current_price)} USDT\n\n"
        f"🔍 FLASH REVIEWER\n"
        f"Điểm đánh giá: {score_text}\n"
        f"Kết luận: {verdict}\n"
        f"Ngưỡng Manual: {float(minimum_score):g}/100\n"
        f"Nhận xét reviewer: {reason}"
        f"{breakdown_block}\n\n"
        "Kế hoạch planner không được bot lưu."
    )


def review_and_gate_plan(
    market_packet: str, planner_output: str, mode: str, minimum_score: float
) -> dict:
    """Shared Manual/Auto reviewer gate with strict technical status parsing."""
    if _extract_setup_status(planner_output) == "STATUS_PARSE_ERROR":
        return {
            "score": None,
            "verdict": "REJECT",
            "reason": "Planner thiếu hoặc trả sai nhãn Trạng thái bắt buộc; bot không tự suy đoán READY_TO_ENTER.",
            "parse_ok": False,
            "passed": False,
            "minimum_score": float(minimum_score),
            "raw": "",
            "status_valid": False,
        }
    review = review_trade_plan_with_flash(market_packet, planner_output, mode, minimum_score)
    review["minimum_score"] = float(minimum_score)
    review["passed"] = _review_passed(review, minimum_score)
    return review


def _review_market_packet(user_prompt: str) -> str:
    """Keep complete market context and remove only planner output instructions.

    Avoids blind character truncation that could cut raw OHLCV mid-packet.
    """
    text = str(user_prompt or "").strip()
    marker = "\nOUTPUT PUBLIC:"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip()
    return text


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
                prefilter_output TEXT,
                planner_input TEXT,
                planner_output TEXT,
                reviewer_output TEXT,
                reviewer_score REAL,
                reviewer_verdict TEXT,
                setup_status TEXT,
                current_price REAL,
                outcome TEXT DEFAULT 'SETUP_CREATED',
                mae REAL,
                mfe REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS auto_scan_bias_state (
                user_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                mode TEXT NOT NULL,
                direction TEXT,
                confirmations INTEGER NOT NULL DEFAULT 0,
                recent_snapshots TEXT NOT NULL DEFAULT '[]',
                updated_at TEXT NOT NULL,
                PRIMARY KEY(user_id, symbol, mode)
            )
        """)
        try:
            conn.execute("ALTER TABLE auto_scan_bias_state ADD COLUMN recent_snapshots TEXT NOT NULL DEFAULT '[]'")
        except sqlite3.OperationalError:
            pass
        for table in ("predictions", "trade_candidates"):
            for col, definition in [
                ("setup_status", "TEXT"),
                ("reviewer_score", "REAL"),
                ("reviewer_verdict", "TEXT"),
                ("mae", "REAL"),
                ("mfe", "REAL"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {definition}")
                except sqlite3.OperationalError:
                    pass


def _save_analysis_snapshot(**kwargs) -> None:
    """Save the full case when Pro is called; the older history and Auto log still serve their own separate UI."""
    try:
        _ensure_v50_tables()
        planner_output = kwargs.get("planner_output") or ""
        public_output = kwargs.get("public_output") or planner_output
        parsed = parse_prediction_from_output(public_output)
        direction = (parsed.get("direction") or "NO_TRADE").upper()
        status = kwargs.get("setup_status") or _extract_setup_status(public_output)
        reviewer_verdict = kwargs.get("reviewer_verdict")
        reviewer_score = kwargs.get("reviewer_score")
        if direction == "NO_TRADE":
            phase, final_result = "PLANNER_NO_TRADE", "NO_TRADE"
        elif str(reviewer_verdict or "").upper() != "APPROVE":
            phase, final_result = "REVIEWER_REJECTED", "REJECTED_PLAN"
        elif status == "SETUP_WAITING_TRIGGER":
            phase, final_result = "APPROVED_WAITING_TRIGGER", direction
        elif status == "READY_TO_ENTER":
            phase, final_result = "APPROVED_READY", direction
        else:
            phase, final_result = "PLANNER_PARSE_ERROR", "PARSE_ERROR"

        prefilter = {}
        try:
            prefilter = json.loads(kwargs.get("prefilter_output") or "{}")
        except Exception:
            prefilter = {}
        save_evaluation_case(
            user_id=kwargs.get("user_id"), chat_id=kwargs.get("chat_id"), source=kwargs.get("source") or "unknown",
            symbol=kwargs.get("symbol"), mode=kwargs.get("mode"), pipeline_phase=phase, final_result=final_result,
            current_price=kwargs.get("current_price"), prefilter_long_score=prefilter.get("long_score"),
            prefilter_short_score=prefilter.get("short_score"), prefilter_direction=prefilter.get("best_direction"),
            bias_window=kwargs.get("bias_window"), planner_direction=direction, planner_status=status,
            reviewer_score=reviewer_score, reviewer_verdict=reviewer_verdict, entry_low=parsed.get("entry_low"),
            entry_high=parsed.get("entry_high"), sl=parsed.get("sl"), tp1=parsed.get("tp1"), tp2=parsed.get("tp2"),
            market_packet=kwargs.get("planner_input"), planner_output=planner_output, reviewer_output=kwargs.get("reviewer_output"),
            public_output=public_output, planner_prompt_hash=prompt_hash(load_system_prompt()),
            reviewer_prompt_hash=prompt_hash(load_reviewer_system_prompt()),
            prefilter_prompt_hash=prompt_hash(load_prefilter_system_prompt()),
        )
        cleanup_evaluation_data()
    except Exception as exc:
        print(f"[SNAPSHOT_SAVE_ERROR] {exc}", flush=True)


def _record_auto_scan_bias_snapshot(
    user_id: int,
    symbol: str,
    mode: str,
    direction: str,
    qualified: bool,
) -> dict:
    """Keep a rolling 3-snapshot bias window.

    A qualified snapshot counts toward planner confirmation. A same-direction
    snapshot below the score threshold is neutral: it occupies one slot but
    does not erase prior confirmation. A strong opposite snapshot naturally
    shifts the rolling window toward the opposite direction.
    """
    _ensure_v50_tables()
    direction = str(direction or "NEUTRAL").upper()
    if direction not in {"LONG", "SHORT"}:
        direction = "NEUTRAL"
    now = iso(utc_now())
    item = direction if qualified and direction in {"LONG", "SHORT"} else f"NEUTRAL_{direction}"
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT recent_snapshots FROM auto_scan_bias_state WHERE user_id=? AND symbol=? AND mode=?",
            (user_id, symbol, mode),
        ).fetchone()
        try:
            history = json.loads(row[0] or "[]") if row else []
        except Exception:
            history = []
        if not isinstance(history, list):
            history = []
        history = [str(x) for x in history[-2:]] + [item]
        long_count = sum(1 for x in history if x == "LONG")
        short_count = sum(1 for x in history if x == "SHORT")
        if long_count > short_count:
            dominant, confirmations = "LONG", long_count
        elif short_count > long_count:
            dominant, confirmations = "SHORT", short_count
        else:
            dominant, confirmations = (direction if qualified else "NEUTRAL"), max(long_count, short_count)
        conn.execute(
            """INSERT INTO auto_scan_bias_state(user_id,symbol,mode,direction,confirmations,recent_snapshots,updated_at)
               VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(user_id,symbol,mode) DO UPDATE SET
               direction=excluded.direction, confirmations=excluded.confirmations,
               recent_snapshots=excluded.recent_snapshots, updated_at=excluded.updated_at""",
            (user_id, symbol, mode, dominant, confirmations, json.dumps(history), now),
        )
    return {
        "direction": dominant,
        "confirmations": confirmations,
        "history": history,
        "qualified_for_direction": (
            direction in {"LONG", "SHORT"}
            and sum(1 for x in history if x == direction) >= AUTO_SCAN_DIRECTION_CONFIRMATIONS
        ),
    }


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

    system_prompt, fear_greed_info, price_tuple = await asyncio.gather(
        asyncio.to_thread(load_system_prompt),
        asyncio.to_thread(lambda: "Không sử dụng Fear & Greed trong phân tích."),
        asyncio.to_thread(get_current_price_str, binance_symbol),
    )
    current_price_str, current_price = price_tuple
    feature_block = build_feature_engineering_block(timeframe_data, mode, current_price)
    feature_snapshot = build_feature_snapshot(timeframe_data, mode, current_price)
    decision_snapshot = build_synchronized_decision_snapshot(timeframe_data, mode, current_price)
    # do not send LONG/SHORT support scorecard into model prompts.
    # This prevents Python from anchoring the model direction.
    direction_scorecard_payload = None
    direction_scorecard = None
    market_snapshot = build_market_snapshot(timeframe_data, fear_greed_info, current_price_str)
    open_signal_context = None
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
    }


async def analyze_symbol(symbol: str, mode: str, user_id: int | None = None, chat_id: int | None = None) -> dict:
    """
    Async entry point used by Telegram handlers.

    Never call requests.get(), a synchronous AI API, or SQLite directly on the event loop.
    Blocking I/O is offloaded to a worker thread via asyncio.to_thread().
    """
    ensure_ai_config()

    await asyncio.to_thread(init_prediction_db)

    binance_symbol = f"{symbol.upper()}USDT"
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
    planner_clean = _remove_rubric_block(raw_output)
    planner_pred = parse_prediction_from_output(planner_clean)
    if (planner_pred.get("direction") or "").upper() in {"LONG", "SHORT"}:
        review = await asyncio.to_thread(
            review_and_gate_plan, _review_market_packet(user_prompt), planner_clean, mode, MIN_SIGNAL_SCORE
        )
        output = ensure_current_price_line(
            sanitize_user_output(_apply_reviewer_score(planner_clean, review)), current_price
        )
    else:
        review = {"score": None, "verdict": "REJECT", "raw": "", "reason": "Planner chọn NO TRADE."}
        output = ensure_current_price_line(
            sanitize_user_output(_insert_public_signal_score(planner_clean, None)), current_price
        )
    pred = parse_prediction_from_output(output)
    await asyncio.to_thread(
        _save_analysis_snapshot,
        user_id=user_id, chat_id=chat_id, symbol=binance_symbol, mode=mode, source="manual",
        model=get_ai_model_name(), planner_input=user_prompt, planner_output=planner_clean,
        reviewer_output=review.get("raw"), reviewer_score=review.get("score"),
        reviewer_verdict=review.get("verdict"), setup_status=_extract_setup_status(output),
        current_price=current_price, public_output=output,
    )
    if (planner_pred.get("direction") or "").upper() in {"LONG", "SHORT"} and not review.get("passed"):
        rejected = _manual_review_rejection_output(
            binance_symbol, mode, current_price, planner_pred, review, MIN_SIGNAL_SCORE
        )
        return {"text": rejected, "candidate_id": None}

    # Model-authoritative flow:
    # - The model alone chooses and is responsible for all of Entry/SL/TP.
    # - Python keeps the model's numbers exactly as returned.
    # - The only gate is the Signal Score; Python does not reject based on RR/ATR/structure/geometry.
    direction = (pred.get("direction") or "").upper()

    if direction == "NO_TRADE":
        # NO TRADE is not saved into predictions/history; only trades the user confirms are tracked.
        return {"text": _strip_public_evidence_for_user(output), "candidate_id": None}

    guard_errors = _validate_actionable_trade_plan(pred, timeframe_data, mode, current_price, output)
    if guard_errors:
        guarded_output = _guarded_no_trade_output(binance_symbol, mode, current_price, guard_errors, pred, timeframe_data)
        log_hidden_rejection(binance_symbol, mode, pred, guard_errors, output)
        # rejected plans are no longer saved into predictions/history.
        return {"text": guarded_output, "candidate_id": None}

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

    print(
        f"[MANUAL_DONE] symbol={binance_symbol} mode={mode} elapsed={loop.time() - manual_started:.1f}s",
        flush=True,
    )
    return {"text": _strip_public_evidence_for_user(output) + tracking_note, "candidate_id": None}


# ─── Auto Scan Mode: DeepSeek prefilter → GLM full analysis ──────────────────

def init_auto_scan_db() -> None:
    """Separate DB for auto scan, kept apart from manual mode/drafts."""
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
                pre_direction     TEXT,
                pre_confidence    INTEGER,
                pre_long_score    INTEGER,
                pre_short_score   INTEGER,
                pre_gap           INTEGER,
                final_direction   TEXT,
                final_confidence  INTEGER,
                reviewer_verdict  TEXT,
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
        for col, definition in [
            ("pre_long_score", "INTEGER"),
            ("pre_short_score", "INTEGER"),
            ("pre_gap", "INTEGER"),
            ("reviewer_verdict", "TEXT"),
        ]:
            try:
                conn.execute(f"ALTER TABLE auto_scan_logs ADD COLUMN {col} {definition}")
            except sqlite3.OperationalError:
                pass

        conn.execute("CREATE INDEX IF NOT EXISTS idx_auto_scan_settings_enabled ON auto_scan_settings(enabled)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_auto_scan_signals_user_symbol_mode ON auto_scan_signals(user_id, symbol, mode, sent_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_auto_scan_logs_user_id ON auto_scan_logs(user_id, id DESC)")

        # Keep a lightweight log over time to evaluate the prefilter; the UI still only shows the 5 most recent rows.
        log_cutoff = iso(utc_now() - timedelta(days=AUTO_SCAN_LOG_RETENTION_DAYS))
        conn.execute("DELETE FROM auto_scan_logs WHERE scanned_at < ?", (log_cutoff,))
        conn.commit()


def _auto_scan_quota_day_key(now: datetime | None = None) -> str:
    """The Auto Scan quota day runs from 07:00 VN to 06:59 VN the next day."""
    local_now = (now or utc_now()).astimezone(VN_TZ)
    wake_hour = max(0, min(23, int(AUTO_SCAN_WAKE_HOUR_VN)))
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
        quota_blocked = bool(enabled and calls >= AUTO_SCAN_MAX_GLM_CALLS_PER_DAY)
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
        conn.commit()
    return {
        "enabled": effective_enabled,
        "quota_blocked": quota_blocked,
        "glm_calls_today": calls,
        "glm_calls_remaining": max(0, AUTO_SCAN_MAX_GLM_CALLS_PER_DAY - calls),
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
        "glm_calls_remaining": max(0, AUTO_SCAN_MAX_GLM_CALLS_PER_DAY - calls),
        "glm_calls_limit": AUTO_SCAN_MAX_GLM_CALLS_PER_DAY, "updated_at": row[8],
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
    sleep_hour = max(0, min(23, int(AUTO_SCAN_SLEEP_HOUR_VN)))
    wake_hour = max(0, min(23, int(AUTO_SCAN_WAKE_HOUR_VN)))
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
        exhausted = calls >= AUTO_SCAN_MAX_GLM_CALLS_PER_DAY
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
        "remaining": max(0, AUTO_SCAN_MAX_GLM_CALLS_PER_DAY - calls),
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
        if calls >= AUTO_SCAN_MAX_GLM_CALLS_PER_DAY:
            conn.execute(
                "UPDATE auto_scan_settings SET enabled=0, quota_resume=1, glm_calls_today=?, glm_calls_day=?, updated_at=? WHERE user_id=?",
                (calls, day_key, iso(utc_now()), user_id),
            )
            conn.commit()
            return {"allowed": False, "used": calls, "remaining": 0, "exhausted": True}
        new_calls = calls + 1
        exhausted = new_calls >= AUTO_SCAN_MAX_GLM_CALLS_PER_DAY
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
        "remaining": max(0, AUTO_SCAN_MAX_GLM_CALLS_PER_DAY - new_calls),
        "exhausted": exhausted,
    }

def get_auto_scan_enabled_users() -> list[dict]:
    init_auto_scan_db()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT user_id, chat_id, symbols FROM auto_scan_settings WHERE enabled=1 AND chat_id IS NOT NULL ORDER BY user_id"
        ).fetchall()
    return [{"user_id": int(r[0]), "chat_id": int(r[1]), "symbols": r[2] or ""} for r in rows]


def _normalize_auto_scan_modes() -> list[str]:
    result = []
    for m in AUTO_SCAN_MODES or ["short"]:
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
    s = (symbol or "").strip().lstrip("/").upper()
    if not s:
        return ""
    return s if s.endswith("USDT") else f"{s}USDT"


def _auto_scan_recently_sent(user_id: int, symbol: str, mode: str, direction: str | None = None) -> bool:
    cooldown = max(0, AUTO_SCAN_SIGNAL_COOLDOWN_MINUTES)
    if cooldown <= 0:
        return False
    cutoff = utc_now() - timedelta(minutes=cooldown)
    clauses = ["user_id=?", "symbol=?", "mode=?", "sent_at>=?"]
    params: list = [user_id, symbol, mode, iso(cutoff)]
    if direction:
        clauses.append("direction=?")
        params.append(direction)
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            f"SELECT 1 FROM auto_scan_signals WHERE {' AND '.join(clauses)} LIMIT 1",
            params,
        ).fetchone()
    return row is not None


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
    return max(60, int(AUTO_SCAN_INTERVAL_SECONDS or 900))


def _auto_scan_slot_info(now: datetime | None = None) -> dict:
    now = (now or utc_now()).astimezone(timezone.utc)
    interval = _auto_scan_interval_seconds()
    delay = max(0, int(AUTO_SCAN_CANDLE_CLOSE_DELAY_SECONDS or 0))
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
    pre_direction: str | None = None,
    pre_confidence: int | None = None,
    pre_long_score: int | None = None,
    pre_short_score: int | None = None,
    pre_gap: int | None = None,
    final_direction: str | None = None,
    final_confidence: int | None = None,
    reviewer_verdict: str | None = None,
    prediction_id: int | None = None,
) -> None:
    init_auto_scan_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO auto_scan_logs
                (user_id, chat_id, symbol, mode, scan_slot, scanned_at, stage, status,
                 pre_direction, pre_confidence, pre_long_score, pre_short_score, pre_gap,
                 final_direction, final_confidence, reviewer_verdict, reason, prediction_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, chat_id, symbol, mode, scan_slot, iso(utc_now()), stage, status,
             pre_direction, pre_confidence, pre_long_score, pre_short_score, pre_gap,
             final_direction, final_confidence, reviewer_verdict, reason, prediction_id),
        )
        log_cutoff = iso(utc_now() - timedelta(days=AUTO_SCAN_LOG_RETENTION_DAYS))
        conn.execute("DELETE FROM auto_scan_logs WHERE scanned_at < ?", (log_cutoff,))
        conn.commit()
    if AUTO_SCAN_DEBUG:
        print(
            f"[AUTO_SCAN] log user={user_id} symbol={symbol} mode={mode} stage={stage} "
            f"status={status} pre={pre_direction}/{pre_confidence} final={final_direction}/{final_confidence} reason={reason}",
            flush=True,
        )


def get_auto_scan_logs(user_id: int, limit: int | None = None) -> list[dict]:
    init_auto_scan_db()
    limit = max(1, min(AUTO_SCAN_LOG_LIMIT, int(limit or AUTO_SCAN_LOG_LIMIT)))
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT scanned_at, symbol, mode, stage, status, pre_direction, pre_confidence,
                   pre_long_score, pre_short_score, pre_gap,
                   final_direction, final_confidence, reviewer_verdict, reason, prediction_id
            FROM auto_scan_logs
            WHERE user_id=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    keys = [
        "scanned_at", "symbol", "mode", "stage", "status",
        "pre_direction", "pre_confidence", "pre_long_score", "pre_short_score", "pre_gap",
        "final_direction", "final_confidence", "reviewer_verdict", "reason", "prediction_id",
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
        "sleep_hour_vn": int(window.get("sleep_hour", AUTO_SCAN_SLEEP_HOUR_VN)),
        "wake_hour_vn": int(window.get("wake_hour", AUTO_SCAN_WAKE_HOUR_VN)),
    }


_DEEPSEEK_MINI_RUBRIC_WEIGHTS = {
    "trend": 25,
    "structure": 25,
    "momentum": 20,
    "confirmation": 15,
    "setup_room": 15,
}


def _prefilter_key_variants(key: str) -> list[str]:
    """Accepted labels for the DeepSeek Flash mini-rubric parser.

    Flash is a cheap prefilter model, so sometimes it returns small label variants
    despite being asked for exact text. These variants keep the system robust while
    still requiring real LONG/SHORT numeric evidence instead of silently turning a
    parse failure into 0/100.
    """
    k = (key or "").strip().lower()
    variants = {
        "trend": ["trend", "xu_huong", "xu hướng", "huong", "hướng"],
        "structure": ["structure", "cau_truc", "cấu trúc", "vi_tri", "vị trí", "price_structure"],
        "momentum": ["momentum", "dong_luong", "động lượng", "macd", "rsi"],
        "confirmation": ["confirmation", "xac_nhan", "xác nhận", "volume", "nen", "nến"],
        "setup_room": ["setup_room", "setup room", "setup", "room", "kha_nang", "khả năng", "setup_potential"],
    }
    return variants.get(k, [k])


def _read_number_after_label(text: str, label_pattern: str, maximum: int) -> int | None:
    patterns = [
        rf"{label_pattern}\s*[:=]\s*(-?\d+(?:\.\d+)?)",
        rf"{label_pattern}\s+(-?\d+(?:\.\d+)?)\s*/\s*{maximum}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if not match:
            continue
        try:
            return max(0, min(maximum, int(round(float(match.group(1))))))
        except Exception:
            return None
    return None


def _extract_prefilter_side_block(raw: str, side: str) -> str:
    # Matches formats such as:
    # LONG:
    # TREND: 12
    # STRUCTURE: 20
    pattern = rf"^\s*{side}\s*[:=]\s*(.*?)(?=^\s*(?:LONG|SHORT|BEST|CALL_GLM|REASON)\s*[:=]|\Z)"
    match = re.search(pattern, raw, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    return match.group(1) if match else ""


def _normalize_prefilter_direction(value) -> str:
    text = str(value or "").strip().upper().replace(" ", "_")
    if text in {"LONG", "SHORT", "NEUTRAL"}:
        return text
    return "NEUTRAL"


def _normalize_prefilter_verdict(value) -> str | None:
    text = str(value or "").strip().upper().replace(" ", "_")
    if text in {"CALL_PLANNER", "CALL_FINAL", "CALL_AI", "CALL_GLM", "YES", "APPROVE"}:
        return "CALL_PLANNER"
    if text in {"SKIP", "NO", "REJECT", "NEUTRAL"}:
        return "SKIP"
    return None


def _prefilter_score_value(value) -> int | None:
    try:
        if isinstance(value, str):
            match = re.search(r"-?[0-9]+(?:[\.,][0-9]+)?", value)
            if not match:
                return None
            value = match.group(0).replace(",", ".")
        return max(0, min(100, int(round(float(value)))))
    except Exception:
        return None


def _parse_deepseek_prefilter_text(text: str | None) -> dict:
    """Parse Flash prefilter totals only; Python never scores rubric items."""
    raw = str(text or "").strip()
    long_score = None
    short_score = None
    best = "NEUTRAL"
    verdict = None
    reason = ""
    parsed_format = None

    payload = _extract_json_object(raw)
    if isinstance(payload, dict):
        lowered = {str(k).strip().lower(): v for k, v in payload.items()}
        long_score = _prefilter_score_value(lowered.get("long_score", lowered.get("long")))
        short_score = _prefilter_score_value(lowered.get("short_score", lowered.get("short")))
        best = _normalize_prefilter_direction(lowered.get("best_direction", lowered.get("best")))
        verdict = _normalize_prefilter_verdict(lowered.get("verdict", lowered.get("decision")))
        reason = str(lowered.get("reason", lowered.get("comment", "")) or "").strip()
        if long_score is not None or short_score is not None:
            parsed_format = "json"

    clean = raw.replace("**", "").replace("__", "").replace("`", "")
    if long_score is None:
        m = re.search(r"(?i)(?:LONG_SCORE|LONG\s*SCORE|LONG|ĐIỂM\s*LONG|DIEM\s*LONG)\s*[:=\-]\s*([0-9]+(?:[\.,][0-9]+)?)", clean)
        if m:
            long_score = _prefilter_score_value(m.group(1)); parsed_format = parsed_format or "text"
    if short_score is None:
        m = re.search(r"(?i)(?:SHORT_SCORE|SHORT\s*SCORE|SHORT|ĐIỂM\s*SHORT|DIEM\s*SHORT)\s*[:=\-]\s*([0-9]+(?:[\.,][0-9]+)?)", clean)
        if m:
            short_score = _prefilter_score_value(m.group(1)); parsed_format = parsed_format or "text"
    if best == "NEUTRAL":
        m = re.search(r"(?i)(?:BEST_DIRECTION|BEST|HƯỚNG\s*TỐT\s*NHẤT|HUONG\s*TOT\s*NHAT)\s*[:=\-]\s*(LONG|SHORT|NEUTRAL)", clean)
        if m:
            best = _normalize_prefilter_direction(m.group(1))
    if verdict is None:
        m = re.search(r"(?i)(?:VERDICT|DECISION|KẾT\s*LUẬN|KET\s*LUAN)\s*[:=\-]\s*([A-Z_ ]+)", clean)
        if m:
            verdict = _normalize_prefilter_verdict(m.group(1))
    if not reason:
        m = re.search(r"(?is)(?:REASON|LÝ\s*DO|LY\s*DO|NHẬN\s*XÉT|NHAN\s*XET)\s*[:=\-]\s*(.+)$", clean)
        if m:
            reason = " ".join(m.group(1).strip().split())

    parse_ok = long_score is not None and short_score is not None
    if parse_ok:
        # Model may return inconsistent BEST/VERDICT. Scores are authoritative inputs;
        # Python only performs arithmetic and gate checks, not market analysis.
        if long_score > short_score:
            computed_best = "LONG"
        elif short_score > long_score:
            computed_best = "SHORT"
        else:
            computed_best = "NEUTRAL"
        best = computed_best
        if verdict is None:
            verdict = "CALL_PLANNER" if best != "NEUTRAL" else "SKIP"
    else:
        verdict = "SKIP"

    return {
        "long_score": long_score,
        "short_score": short_score,
        "best_direction": best,
        "model_verdict": verdict,
        "reason": reason,
        "parse_ok": parse_ok,
        "parsed_format": parsed_format,
        "raw_text": raw[:2000],
        "rubric_complete": parse_ok,
        "used_legacy_format": parsed_format == "text",
        "used_total_score_fallback": False,
    }


def _evaluate_deepseek_prefilter_gate(prefilter: dict | None) -> dict:
    """Apply thresholds to model-provided final LONG/SHORT scores.

    Flash performs all qualitative scoring. Python only validates 0..100 values,
    computes the numeric gap, chooses the larger score, and applies configured gates.
    """
    payload = prefilter if isinstance(prefilter, dict) else {}
    long_score = _prefilter_score_value(payload.get("long_score"))
    short_score = _prefilter_score_value(payload.get("short_score"))
    parse_ok = bool(payload.get("parse_ok") and long_score is not None and short_score is not None)

    if not parse_ok:
        raw_preview = str(payload.get("raw_text") or payload.get("reason") or "").replace("\n", " ").strip()
        if len(raw_preview) > 160:
            raw_preview = raw_preview[:160] + "..."
        reason = "Không đọc được điểm LONG/SHORT cuối từ Flash prefilter."
        if raw_preview:
            reason += f" Raw đầu: {raw_preview}"
        return {
            "long_score": None, "short_score": None, "direction": "NEUTRAL",
            "raw_direction": "NEUTRAL", "best_score": None, "gap": None,
            "should_call_glm": False, "reason": reason, "rubric_complete": False,
            "parse_ok": False, "used_total_score_fallback": False,
        }

    gap = abs(long_score - short_score)
    best_score = max(long_score, short_score)
    if long_score > short_score:
        raw_direction = "LONG"
    elif short_score > long_score:
        raw_direction = "SHORT"
    else:
        raw_direction = "NEUTRAL"

    neutral_by_gap = raw_direction == "NEUTRAL" or gap < AUTO_SCAN_PREFILTER_MIN_DIRECTION_GAP
    direction = "NEUTRAL" if neutral_by_gap else raw_direction
    above_threshold = best_score >= AUTO_SCAN_MIN_PREFILTER_CONFIDENCE
    should_call_glm = bool(above_threshold and not neutral_by_gap)

    if neutral_by_gap:
        gate_reason = (
            f"Flash prefilter gần cân bằng: LONG {long_score}/100, SHORT {short_score}/100; "
            f"chênh {gap} điểm, cần tối thiểu {AUTO_SCAN_PREFILTER_MIN_DIRECTION_GAP} điểm."
        )
    elif not above_threshold:
        gate_reason = (
            f"{raw_direction} đạt {best_score}/100, dưới ngưỡng lọc nhanh "
            f"{AUTO_SCAN_MIN_PREFILTER_CONFIDENCE}/100."
        )
    else:
        gate_reason = (
            f"{raw_direction} đạt {best_score}/100, hướng đối diện "
            f"{min(long_score, short_score)}/100, chênh {gap} điểm; gọi planner."
        )

    return {
        "long_score": long_score, "short_score": short_score,
        "direction": direction, "raw_direction": raw_direction,
        "best_score": best_score, "gap": gap,
        "should_call_glm": should_call_glm, "reason": gate_reason,
        "rubric_complete": True, "parse_ok": True,
        "used_total_score_fallback": False,
    }


def _prefilter_format_repair(raw_output: str) -> dict:
    raw = str(raw_output or "").strip()
    if not raw:
        return {"long_score": None, "short_score": None, "parse_ok": False, "raw_text": ""}
    prompt = "\n".join([
        "Chỉ định dạng lại kết quả prefilter bên dưới. Không phân tích lại và không đổi điểm.",
        "Trả đúng một JSON object hợp lệ, không markdown:",
        '{"long_score": 0, "short_score": 0, "best_direction": "NEUTRAL", "verdict": "SKIP", "reason": "..."}',
        "Nếu nội dung gốc không có đủ hai điểm, dùng null cho điểm bị thiếu.",
        "",
        "NỘI DUNG GỐC:",
        raw[:10000],
    ])
    result = _deepseek_create_once(
        system="Bạn là bộ sửa định dạng JSON. Chỉ định dạng lại, không phân tích hoặc đổi dữ liệu.",
        messages=[{"role": "user", "content": prompt}],
        timeout=DEEPSEEK_TIMEOUT_SECONDS,
        model=DEEPSEEK_MODEL,
        max_tokens=max(1200, min(3000, DEEPSEEK_MAX_OUTPUT_TOKENS)),
        temperature=0,
        response_format={"type": "json_object"},
        reasoning_effort="off",
    )
    repaired_raw = (result.get("text") or result.get("reasoning_text") or "").strip()
    parsed = _parse_deepseek_prefilter_text(repaired_raw)
    parsed["raw_text"] = repaired_raw[:2000]
    return parsed


def request_deepseek_prefilter(prefilter_text: str) -> dict:
    """Flash self-scores LONG/SHORT and returns only final totals."""
    system_prompt = load_prefilter_system_prompt()
    prompt = prefilter_text
    retry_count = max(0, LLM_API_RETRIES)
    last_exc = None
    for retry_idx in range(retry_count + 1):
        try:
            result = _deepseek_create_once(
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                model=DEEPSEEK_MODEL,
                max_tokens=max(2000, DEEPSEEK_MAX_OUTPUT_TOKENS),
                temperature=DEEPSEEK_TEMPERATURE,
                response_format={"type": "json_object"},
                reasoning_effort=DEEPSEEK_PREFILTER_REASONING_EFFORT,
            )
            raw = (result.get("text") or result.get("reasoning_text") or "").strip()
            parsed = _parse_deepseek_prefilter_text(raw)
            parsed["usage"] = result.get("usage")
            parsed["stop_reason"] = result.get("stop_reason")
            if parsed.get("parse_ok"):
                return parsed
            repaired = _prefilter_format_repair(raw)
            repaired["usage"] = result.get("usage")
            if repaired.get("parse_ok"):
                return repaired
            return parsed
        except Exception as exc:
            last_exc = exc
            if retry_idx >= retry_count or not _is_transient_llm_error(exc):
                raise
            try:
                import time
                time.sleep(max(0.0, LLM_RETRY_SLEEP_SECONDS) * (retry_idx + 1))
            except Exception:
                pass
    if last_exc:
        raise last_exc
    return {
        "long_score": None, "short_score": None, "best_direction": "NEUTRAL",
        "model_verdict": "SKIP", "reason": "Flash không trả được kết quả.",
        "parse_ok": False, "raw_text": "",
    }


def _auto_scan_text_header(symbol: str, mode: str) -> str:
    mode_label = "SCALP" if mode == "short" else "SWING"
    return f"🤖 AUTO SCAN — {symbol} — {mode_label}\n"


def _strip_public_evidence_for_user(output: str) -> str:
    """Hide the Evidence blocks from every public message (both Manual and Auto Scan).

    The planner still returns the full content; the reviewer and DB still receive/save the full,
    unedited full_response. Only the final text sent to Telegram is trimmed, from right after
    Activation straight to Risk.
    """
    lines = (output or "").splitlines()
    kept: list[str] = []
    skipping = False
    for line in lines:
        normalized = line.strip().lower()
        if not skipping and normalized.startswith("bằng chứng entry"):
            skipping = True
            continue
        if skipping:
            if normalized.startswith("⚠️ rủi ro") or normalized.startswith("rủi ro"):
                skipping = False
                kept.append(line)
            continue
        kept.append(line)
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

    def log_and_return(stage: str, status: str, reason: str, **kwargs) -> dict:
        _record_auto_scan_log(
            user_id, chat_id, binance_symbol, mode,
            scan_slot=scan_slot, stage=stage, status=status, reason=reason, **kwargs,
        )
        return {"send": False, "reason": reason, "stage": stage, "status": status, **kwargs}

    # The quota guard MUST run before cooldown, Binance, and DeepSeek.
    # This way, once a user hits 5/5, their entire Auto Scan truly stops until 07:00.
    quota_state = get_auto_scan_glm_quota_state(user_id)
    if not quota_state.get("allowed"):
        return log_and_return(
            "quota",
            "skipped",
            f"Đã dùng đủ {AUTO_SCAN_MAX_GLM_CALLS_PER_DAY} lượt gọi AI cuối trong ngày Auto Scan; sẽ tự bật lại lúc 07:00 VN.",
        )

    if _auto_scan_recently_sent(user_id, binance_symbol, mode):
        return log_and_return("cooldown", "skipped", "cooldown")

    timeframe_data = await collect_timeframe_data(binance_symbol, mode)
    if not any(df is not None and not df.empty for df in timeframe_data.values()):
        return log_and_return("binance", "error", "no binance data")

    # GLM Auto Scan uses the exact same context builder as manual analysis.
    ctx = await prepare_analysis_context(
        binance_symbol,
        mode,
        user_id=user_id,
        timeframe_data=timeframe_data,
    )
    system_prompt = ctx["system_prompt"]
    current_price_str = ctx["current_price_str"]
    current_price = ctx["current_price"]
    open_signal_context = ctx["open_signal_context"]
    feature_block = ctx["feature_block"]
    feature_snapshot = ctx["feature_snapshot"]
    decision_snapshot = ctx["decision_snapshot"]
    direction_scorecard = None
    direction_scorecard_payload = None
    market_snapshot = ctx["market_snapshot"]

    prefilter_text = build_deepseek_prefilter_text(
        symbol=binance_symbol,
        mode=mode,
        current_price_str=current_price_str,
        feature_snapshot=feature_snapshot,
        feature_block=feature_block,
        decision_snapshot=decision_snapshot,
        open_signal_context=open_signal_context,
        direction_scorecard=None,
    )

    prefilter = await asyncio.to_thread(request_deepseek_prefilter, prefilter_text)
    gate = _evaluate_deepseek_prefilter_gate(prefilter)
    pre_direction = gate.get("direction")
    pre_conf = gate.get("best_score")
    if gate.get("parse_ok"):
        prefilter_score_kwargs = {
            "pre_long_score": gate.get("long_score"),
            "pre_short_score": gate.get("short_score"),
            "pre_gap": gate.get("gap"),
        }
    else:
        # Do not persist fake LONG 0 / SHORT 0 when the Flash answer was not parseable.
        prefilter_score_kwargs = {"pre_long_score": None, "pre_short_score": None, "pre_gap": None}
    deepseek_direction = pre_direction
    deepseek_conf = pre_conf
    deepseek_reason = gate.get("reason")

    # Rolling confirmation: require 2 qualifying snapshots inside the latest 3.
    # A parseable same-direction snapshot below threshold is neutral and does
    # not wipe the previous qualifying bias. Parse errors remain non-evidence.
    bias_state = None
    if gate.get("parse_ok") and pre_direction in {"LONG", "SHORT"}:
        bias_state = await asyncio.to_thread(
            _record_auto_scan_bias_snapshot,
            user_id, binance_symbol, mode, pre_direction, bool(gate.get("should_call_glm")),
        )

    if not gate.get("should_call_glm"):
        return log_and_return(
            "deepseek",
            "rejected",
            gate.get("reason") or "DeepSeek Flash không thấy ứng viên LONG/SHORT đủ mạnh để gọi AI cuối.",
            pre_direction=pre_direction,
            pre_confidence=pre_conf,
            **prefilter_score_kwargs,
        )

    confirmations = int((bias_state or {}).get("confirmations") or 0)
    confirmed_for_direction = bool((bias_state or {}).get("qualified_for_direction"))
    if not confirmed_for_direction:
        history = (bias_state or {}).get("history") or []
        history_text = " → ".join(history) if history else "N/A"
        return log_and_return(
            "confirmation",
            "waiting",
            f"Bias {pre_direction} mới đạt {confirmations}/{AUTO_SCAN_DIRECTION_CONFIRMATIONS} snapshot đạt chuẩn trong 3 snapshot gần nhất; chưa gọi planner. Cửa sổ: {history_text}.",
            pre_direction=pre_direction,
            pre_confidence=pre_conf,
            **prefilter_score_kwargs,
        )

    quota = reserve_auto_scan_glm_call(user_id)
    if not quota.get("allowed"):
        return log_and_return(
            "quota", "skipped",
            f"Đã dùng đủ {AUTO_SCAN_MAX_GLM_CALLS_PER_DAY} lượt gọi AI cuối trong ngày Auto Scan; sẽ tự bật lại lúc 07:00 VN.",
            pre_direction=pre_direction, pre_confidence=pre_conf, **prefilter_score_kwargs,
        )

    user_prompt = ctx["user_prompt"]
    flash_note = "\n\nLỌC NHANH DEEPSEEK FLASH — CHỈ BÁO RẰNG SNAPSHOT ĐÁNG PHÂN TÍCH SÂU:\n" + (
        "- Lớp lọc nhanh đã đạt điều kiện gọi AI cuối, nhưng điểm LONG/SHORT của Flash không được đưa vào đây để tránh neo hướng.\n"
        "- Bạn phải tự chọn LONG / SHORT / NO TRADE và lập plan từ dữ liệu đầy đủ bên trên. Không tự chấm điểm; Flash reviewer độc lập sẽ chấm sau."
    )
    planner_input = user_prompt + flash_note
    raw_output = await asyncio.to_thread(request_claude_analysis, system_prompt, planner_input)
    planner_clean = _remove_rubric_block(raw_output)
    planner_pred = parse_prediction_from_output(planner_clean)
    if (planner_pred.get("direction") or "").upper() in {"LONG", "SHORT"}:
        review = await asyncio.to_thread(
            review_and_gate_plan, _review_market_packet(user_prompt), planner_clean, mode, AUTO_SCAN_MIN_FINAL_SIGNAL_SCORE
        )
        output = ensure_current_price_line(
            sanitize_user_output(_apply_reviewer_score(planner_clean, review)), current_price
        )
    else:
        review = {"score": None, "verdict": "REJECT", "raw": "", "reason": "Planner chọn NO TRADE."}
        output = ensure_current_price_line(
            sanitize_user_output(_insert_public_signal_score(planner_clean, None)), current_price
        )
    pred = parse_prediction_from_output(output)
    direction = (pred.get("direction") or "").upper()
    await asyncio.to_thread(
        _save_analysis_snapshot,
        user_id=user_id, chat_id=chat_id, symbol=binance_symbol, mode=mode, source="autoscan",
        model=get_ai_model_name(), prefilter_output=json.dumps(prefilter, ensure_ascii=False),
        planner_input=planner_input, planner_output=planner_clean,
        reviewer_output=review.get("raw"), reviewer_score=review.get("score"),
        reviewer_verdict=review.get("verdict"), setup_status=_extract_setup_status(output),
        current_price=current_price, public_output=output,
    )
    final_conf = int(review.get("score") or pred.get("signal_score") or pred.get("confidence") or 0)

    if not review.get("passed") and direction in {"LONG", "SHORT"}:
        return log_and_return(
            "reviewer", "rejected",
            f"Flash reviewer REJECT: {review.get('reason') or 'kế hoạch chưa được dữ liệu hỗ trợ đủ.'}",
            pre_direction=pre_direction, pre_confidence=pre_conf,
            final_direction=direction, final_confidence=(int(review.get("score")) if review.get("score") is not None else None),
            reviewer_verdict=review.get("verdict"),
            **prefilter_score_kwargs,
        )

    setup_status = _extract_setup_status(output)
    # SETUP_WAITING_TRIGGER is still a valid plan that needs to be sent to the user so they can
    # place a pending order / track it in advance. The status only affects execution,
    # it's no longer a gate that blocks sending. Only NO_TRADE or a failed reviewer/score gate
    # gets skipped.

    if direction == "NO_TRADE":
        if AUTO_SCAN_SEND_NO_TRADE:
            return {"send": True, "text": _auto_scan_text_header(binance_symbol, mode) + output, "prediction_id": None}
        return log_and_return("planner", "rejected", "Planner Pro chọn NO TRADE sau phân tích đầy đủ.", pre_direction=pre_direction, pre_confidence=pre_conf, final_direction=direction, final_confidence=final_conf, **prefilter_score_kwargs)

    if direction not in {"LONG", "SHORT"}:
        return log_and_return("planner", "rejected", "Planner Pro không trả quyết định LONG/SHORT hợp lệ.", pre_direction=pre_direction, pre_confidence=pre_conf, final_direction=direction, final_confidence=final_conf, **prefilter_score_kwargs)

    # Not hard-blocked just because the final AI's direction differs from DeepSeek Flash's.
    # Flash is only a cost-saving prefilter; the final AI still decides independently from the full data.
    # Python only sanity-checks the final AI's direction against objective data after the model has decided.

    if _auto_scan_recently_sent(user_id, binance_symbol, mode, direction=direction):
        return log_and_return("cooldown", "skipped", "direction cooldown", pre_direction=pre_direction, pre_confidence=pre_conf, final_direction=direction, final_confidence=final_conf, **prefilter_score_kwargs)

    guard_errors = _validate_actionable_trade_plan(pred, timeframe_data, mode, current_price, output)
    if guard_errors:
        log_hidden_rejection(binance_symbol, mode, pred, guard_errors, output)
        return log_and_return("guard", "rejected", "guard rejected", pre_direction=pre_direction, pre_confidence=pre_conf, final_direction=direction, final_confidence=final_conf, **prefilter_score_kwargs)

    can_track = all(pred.get(k) is not None for k in ("entry_low", "entry_high", "sl", "tp1"))
    if not can_track:
        return log_and_return("planner", "rejected", "Planner Pro thiếu Entry/SL/TP bắt buộc", pre_direction=pre_direction, pre_confidence=pre_conf, final_direction=direction, final_confidence=final_conf, **prefilter_score_kwargs)

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
    )
    try:
        if _price_in_entry_range(current_price, pred.get("entry_low"), pred.get("entry_high")):
            entry_price = _entry_price(direction, pred.get("entry_low"), pred.get("entry_high"), current_price)
            if entry_price is not None:
                await asyncio.to_thread(mark_entry_filled, prediction_id, float(entry_price), utc_now(), mode)
    except Exception:
        pass

    _record_auto_scan_signal(user_id, chat_id, binance_symbol, mode, direction, final_conf, int(prediction_id))
    if setup_status == "SETUP_WAITING_TRIGGER":
        execution_note = (
            "\n\n⏳ Setup đã được duyệt nhưng đang chờ trigger. "
            "Bạn có thể đặt lệnh chờ/theo dõi vùng Entry trước; không vào market ngay nếu điều kiện kích hoạt chưa xuất hiện."
        )
    else:
        execution_note = "\n\n✅ Trigger đã sẵn sàng; có thể thực thi theo kế hoạch trong vùng Entry."
    public_output = _strip_public_evidence_for_user(output)
    text = (
        _auto_scan_text_header(binance_symbol, mode)
        + public_output
        + execution_note
        + "\n\nBot đã tự lưu tín hiệu Auto Scan này để theo dõi. Không cần bấm xác nhận."
    )
    return {
        "send": True,
        "text": text,
        "prediction_id": int(prediction_id),
        "direction": direction,
        "confidence": final_conf,
        "pre_direction": pre_direction,
        "pre_confidence": pre_conf,
        "pre_long_score": gate.get("long_score"),
        "pre_short_score": gate.get("short_score"),
        "pre_gap": gate.get("gap"),
        "final_direction": direction,
        "final_confidence": final_conf,
        "reviewer_verdict": review.get("verdict"),
    }


async def _run_auto_scan_cycle(bot=None, force: bool = False) -> dict:
    """Run exactly one Auto Scan candle slot without overlap/catch-up handling."""
    window = maintain_auto_scan_daily_window()
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

    users = get_auto_scan_enabled_users()
    modes = _normalize_auto_scan_modes()
    payload = {"users": len(users), "symbols": 0, "modes": modes, "sent": 0, "checked": 0, "errors": 0, "skipped": False, "slot": slot_info.get("slot"), "next_scan_at": slot_info.get("next_slot")}
    if not users:
        mark_auto_scan_slot_done(slot_info.get("slot") or iso(utc_now()))
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
                        await bot.send_message(chat_id=user["chat_id"], text=result["text"])
                        # A valid Auto Scan signal has already been saved into predictions (/history) above.
                        # After the Telegram message sends successfully, also save a separate record into
                        # auto_scan_logs so the signal also shows up in /autoscanlog.
                        _record_auto_scan_log(
                            user.get("user_id"),
                            user.get("chat_id"),
                            normalize_auto_scan_symbol(symbol),
                            mode,
                            scan_slot=slot_info.get("slot"),
                            stage="sent",
                            status="sent",
                            reason="Đã gửi tín hiệu Auto Scan và lưu đồng thời vào history cùng Auto Scan log.",
                            pre_direction=result.get("pre_direction"),
                            pre_confidence=result.get("pre_confidence"),
                            pre_long_score=result.get("pre_long_score"),
                            pre_short_score=result.get("pre_short_score"),
                            pre_gap=result.get("pre_gap"),
                            final_direction=result.get("final_direction") or result.get("direction"),
                            final_confidence=result.get("final_confidence") if result.get("final_confidence") is not None else result.get("confidence"),
                            reviewer_verdict=result.get("reviewer_verdict"),
                            prediction_id=result.get("prediction_id"),
                        )
                        payload["sent"] += 1
                except Exception as exc:
                    payload["errors"] += 1
                    _record_auto_scan_log(
                        user.get("user_id"), user.get("chat_id"), symbol, mode,
                        scan_slot=slot_info.get("slot"), stage="error", status="error", reason=str(exc)[:500],
                    )
                    print(f"[AUTO_SCAN] error user={user.get('user_id')} symbol={symbol} mode={mode}: {exc}", flush=True)
    mark_auto_scan_slot_done(slot_info.get("slot") or iso(utc_now()))
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


# ─── Final overrides (must remain after legacy prefilter definitions) ─────────

def build_deepseek_prefilter_text(
    symbol: str,
    mode: str,
    current_price_str: str,
    feature_snapshot: str | None,
    feature_block: str | None,
    decision_snapshot: str | None,
    open_signal_context: str | None,
    direction_scorecard: str | None = None,
) -> str:
    """Data-only user message for Flash prefilter; rules live in system prompt."""
    mode_label = "SCALP" if mode == "short" else "SWING"
    compact_feature = feature_snapshot or feature_block or "Không có feature snapshot."
    return "\n".join([
        f"AUTO SCAN PREFILTER — {symbol} {mode_label}",
        current_price_str,
        "",
        "SNAPSHOT QUYẾT ĐỊNH ĐỒNG BỘ VỚI PLANNER:",
        decision_snapshot or "SYNCHRONIZED_DECISION_SNAPSHOT: không có.",
        "",
        "SNAPSHOT KỸ THUẬT RÚT GỌN:",
        compact_feature,
    ])

