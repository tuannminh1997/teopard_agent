import gzip
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = os.getenv("DB_PATH", "bot.db")
EVALUATION_ENABLED = os.getenv("EVALUATION_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
EVALUATION_FULL_RETENTION_DAYS = max(7, int(os.getenv("EVALUATION_FULL_RETENTION_DAYS", "60")))
EVALUATION_METADATA_RETENTION_DAYS = max(EVALUATION_FULL_RETENTION_DAYS, int(os.getenv("EVALUATION_METADATA_RETENTION_DAYS", "180")))
AUTOSCAN_LOG_RETENTION_DAYS = max(1, int(os.getenv("AUTOSCAN_LOG_RETENTION_DAYS", os.getenv("AUTO_SCAN_LOG_RETENTION_DAYS", "14"))))
BOT_VERSION = os.getenv("BOT_VERSION", "3.0")

# Single source of truth for lifecycle timing by mode (short = SCALP, long = SWING).
# analyze.py imports these two dicts instead of redefining them, to avoid the hour
# mismatch between where predictions are created and where evaluation_cases are tracked.
ENTRY_WAIT_HOURS = {
    "short": 12,      # Scalp: wait up to 12h for Entry to fill
    "long": 24,       # Swing: wait up to 24h for Entry to fill
}
TRADE_MAX_HOLD_HOURS = {
    "short": 72,      # Scalp: track for up to 72h after Entry fills
    "long": 24 * 7,   # Swing: track for up to 7 days after Entry fills
}


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compress(text: str | None) -> bytes | None:
    if not text:
        return None
    return gzip.compress(str(text).encode("utf-8"), compresslevel=6)


def prompt_hash(text: str | None) -> str | None:
    if not text:
        return None
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]


def init_evaluation_db() -> None:
    if not EVALUATION_ENABLED:
        return
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                user_id INTEGER,
                chat_id INTEGER,
                source TEXT NOT NULL,
                symbol TEXT NOT NULL,
                mode TEXT NOT NULL,
                pipeline_phase TEXT NOT NULL,
                final_result TEXT NOT NULL,
                current_price REAL,
                prefilter_long_score INTEGER,
                prefilter_short_score INTEGER,
                prefilter_direction TEXT,
                bias_window TEXT,
                planner_direction TEXT,
                planner_status TEXT,
                reviewer_score REAL,
                reviewer_verdict TEXT,
                entry_low REAL,
                entry_high REAL,
                sl REAL,
                tp1 REAL,
                tp2 REAL,
                market_packet_gzip BLOB,
                planner_output_gzip BLOB,
                reviewer_output_gzip BLOB,
                public_output_gzip BLOB,
                bot_version TEXT,
                planner_prompt_hash TEXT,
                reviewer_prompt_hash TEXT,
                prefilter_prompt_hash TEXT,
                tracking_status TEXT NOT NULL DEFAULT 'OPEN',
                outcome TEXT,
                max_favorable_price REAL,
                max_adverse_price REAL,
                expires_at TEXT,
                updated_at TEXT NOT NULL
            )
        """)
        for col, definition in [
            ("entry_touched_at", "TEXT"),
            ("sl_touched_at", "TEXT"),
            ("tp1_touched_at", "TEXT"),
            ("post_sl_max_favorable_price", "REAL"),
            ("post_sl_max_adverse_price", "REAL"),
            ("post_sl_tp1_reached", "INTEGER NOT NULL DEFAULT 0"),
            ("post_sl_entry_recovered_at", "TEXT"),
            ("post_sl_diagnosis", "TEXT"),
            ("entry_deadline", "TEXT"),
            ("post_tp1_max_favorable_price", "REAL"),
            ("post_tp1_max_adverse_price", "REAL"),
            ("post_tp1_tp2_reached", "INTEGER NOT NULL DEFAULT 0"),
            ("post_tp1_sl_hit", "INTEGER NOT NULL DEFAULT 0"),
            ("post_tp1_diagnosis", "TEXT"),
            ("funding_rate_pct", "REAL"),
            ("open_interest_trend", "TEXT"),
            ("btc_context_text", "TEXT"),
            ("long_short_divergence", "TEXT"),
        ]:
            try:
                conn.execute(f"ALTER TABLE evaluation_cases ADD COLUMN {col} {definition}")
            except sqlite3.OperationalError:
                pass
        conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_created ON evaluation_cases(created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_source_phase ON evaluation_cases(source, pipeline_phase, created_at DESC)")
        conn.commit()


def save_evaluation_case(**kwargs) -> int | None:
    if not EVALUATION_ENABLED:
        return None
    init_evaluation_db()
    now = datetime.now(timezone.utc)
    mode = str(kwargs.get("mode") or "short")
    # BUGFIX: this used to be "12 if mode=='short' else 72", which was wrong on two counts:
    # the hours didn't match the real ENTRY_WAIT_HOURS/TRADE_MAX_HOLD_HOURS (especially bad
    # for swing: 72h instead of the real 24h entry wait + 168h tracking = 192h), and it used
    # a single expires_at timestamp for two different things (entry-wait expiry vs.
    # post-fill tracking expiry). Old result: many cases were closed as "EXPIRED_AFTER_ENTRY"
    # right after the entry filled even though they hadn't been tracked long enough yet,
    # skewing the outcome stats. Now entry_deadline (the entry-wait expiry) and expires_at
    # (the final cutoff = entry_deadline + max_hold, used as a safety ceiling) are stored
    # separately; update_evaluation_tracking() recomputes the real tracking deadline based
    # on the actual entry_touched_at time, rather than relying on the static info from
    # when the case was created).
    entry_wait_hours = ENTRY_WAIT_HOURS.get(mode, 24)
    max_hold_hours = TRADE_MAX_HOLD_HOURS.get(mode, 72)
    entry_deadline = (now + timedelta(hours=entry_wait_hours)).isoformat()
    expires_at = (now + timedelta(hours=entry_wait_hours + max_hold_hours)).isoformat()
    fields = [
        "created_at","user_id","chat_id","source","symbol","mode","pipeline_phase","final_result",
        "current_price","prefilter_long_score","prefilter_short_score","prefilter_direction","bias_window",
        "planner_direction","planner_status","reviewer_score","reviewer_verdict","entry_low","entry_high",
        "sl","tp1","tp2","market_packet_gzip","planner_output_gzip","reviewer_output_gzip",
        "public_output_gzip","bot_version","planner_prompt_hash","reviewer_prompt_hash","prefilter_prompt_hash",
        "tracking_status","outcome","expires_at","entry_deadline","updated_at",
        "funding_rate_pct","open_interest_trend","btc_context_text","long_short_divergence"
    ]
    values = [
        now.isoformat(), kwargs.get("user_id"), kwargs.get("chat_id"), kwargs.get("source"), kwargs.get("symbol"), mode,
        kwargs.get("pipeline_phase") or "UNKNOWN", kwargs.get("final_result") or "UNKNOWN", kwargs.get("current_price"),
        kwargs.get("prefilter_long_score"), kwargs.get("prefilter_short_score"), kwargs.get("prefilter_direction"),
        json.dumps(kwargs.get("bias_window"), ensure_ascii=False) if isinstance(kwargs.get("bias_window"), dict) else kwargs.get("bias_window"),
        kwargs.get("planner_direction"), kwargs.get("planner_status"), kwargs.get("reviewer_score"), kwargs.get("reviewer_verdict"),
        kwargs.get("entry_low"), kwargs.get("entry_high"), kwargs.get("sl"), kwargs.get("tp1"), kwargs.get("tp2"),
        _compress(kwargs.get("market_packet")), _compress(kwargs.get("planner_output")), _compress(kwargs.get("reviewer_output")),
        _compress(kwargs.get("public_output")), BOT_VERSION, kwargs.get("planner_prompt_hash"), kwargs.get("reviewer_prompt_hash"),
        kwargs.get("prefilter_prompt_hash"), "OPEN", None, expires_at, entry_deadline, now.isoformat(),
        kwargs.get("funding_rate_pct"), kwargs.get("open_interest_trend"), kwargs.get("btc_context_text"),
        kwargs.get("long_short_divergence"),
    ]
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            f"INSERT INTO evaluation_cases ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})", values
        )
        conn.commit()
        return int(cur.lastrowid)


def cleanup_evaluation_data() -> None:
    if not EVALUATION_ENABLED:
        return
    init_evaluation_db()
    now = datetime.now(timezone.utc)
    full_cutoff = (now - timedelta(days=EVALUATION_FULL_RETENTION_DAYS)).isoformat()
    metadata_cutoff = (now - timedelta(days=EVALUATION_METADATA_RETENTION_DAYS)).isoformat()
    log_cutoff = (now - timedelta(days=AUTOSCAN_LOG_RETENTION_DAYS)).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            UPDATE evaluation_cases
            SET market_packet_gzip=NULL, planner_output_gzip=NULL, reviewer_output_gzip=NULL, public_output_gzip=NULL
            WHERE created_at < ? AND market_packet_gzip IS NOT NULL
        """, (full_cutoff,))
        conn.execute("DELETE FROM evaluation_cases WHERE created_at < ?", (metadata_cutoff,))
        try:
            conn.execute("DELETE FROM auto_scan_logs WHERE scanned_at < ?", (log_cutoff,))
        except sqlite3.OperationalError:
            pass
        # analysis_snapshots is no longer written to (was the prefilter-call log); this just ages
        # out any old rows left over from before the pipeline dropped the prefilter/reviewer stages.
        try:
            conn.execute("DELETE FROM analysis_snapshots WHERE created_at < ?", (full_cutoff,))
        except sqlite3.OperationalError:
            pass
        conn.commit()


def export_database_snapshot(destination: str) -> str:
    dest = str(Path(destination))
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(DB_PATH)
    try:
        dst = sqlite3.connect(dest)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    return dest

# Outcome tracking must use the same futures market the bot analyzes and the user trades —
# spot candles can diverge from perp price action, especially during volatility, which would
# make WIN/LOSS/SL/TP hits recorded here inconsistent with what actually happened on futures.
BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[list]:
    import requests
    response = requests.get(
        BINANCE_KLINES_URL,
        params={"symbol": symbol, "interval": interval, "startTime": start_ms, "endTime": end_ms, "limit": 1000},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, list) else []


def update_evaluation_tracking() -> dict:
    """Objective tracking using closed OHLC candles; no AI calls and no guessing of intra-candle tick order."""
    if not EVALUATION_ENABLED:
        return {"checked": 0, "updated": 0}
    init_evaluation_db()
    now = datetime.now(timezone.utc)
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        # LIMIT is a generous safety cap, not a real-world ceiling: oldest-first ordering means
        # cases past the cap would starve indefinitely if the open backlog ever exceeded it.
        rows = conn.execute("""
            SELECT * FROM evaluation_cases
            WHERE tracking_status IN ('OPEN','POST_SL','POST_TP1') AND created_at >= ?
            ORDER BY created_at ASC
            LIMIT 2000
        """, ((now - timedelta(days=10)).isoformat(),)).fetchall()

    grouped: dict[tuple[str, str], list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault((row["symbol"], row["mode"]), []).append(row)

    updated = 0
    for (symbol, mode), cases in grouped.items():
        interval = "15m" if mode == "short" else "1h"
        start = min(_parse_dt(row["created_at"]) for row in cases)
        try:
            klines = _fetch_klines(symbol, interval, int(start.timestamp() * 1000), int(now.timestamp() * 1000))
        except Exception as exc:
            print(f"[EVAL_TRACK_ERROR] {symbol} {mode}: {exc}", flush=True)
            continue
        candles = [
            {
                "open_time": datetime.fromtimestamp(float(k[0]) / 1000, timezone.utc),
                "close_time": datetime.fromtimestamp(float(k[6]) / 1000, timezone.utc) if len(k) > 6 else None,
                "open": float(k[1]), "high": float(k[2]), "low": float(k[3]), "close": float(k[4]),
            }
            for k in klines if isinstance(k, list) and len(k) >= 5
        ]
        # Don't use the still-open candle to conclude that a level was touched.
        candles = [c for c in candles if c["close_time"] is None or c["close_time"] <= now]

        with sqlite3.connect(DB_PATH) as conn:
            for row in cases:
                created = _parse_dt(row["created_at"])
                expires = _parse_dt(row["expires_at"])
                # BUGFIX: entry_deadline is kept separate from the "entry-wait expiry" (12h/24h),
                # apart from expires (the final cutoff). Cases created before this migration
                # don't have entry_deadline yet -> fall back to expires so old data doesn't break.
                entry_deadline_raw = row["entry_deadline"] if "entry_deadline" in row.keys() else None
                entry_deadline = _parse_dt(entry_deadline_raw) if entry_deadline_raw else expires
                max_hold_hours = TRADE_MAX_HOLD_HOURS.get(mode, 72)
                # Only candles that STARTED after the signal existed. Filtering by close_time would
                # admit the candle already in progress at creation time, whose high/low contain ticks
                # from before the plan was made — enough to record a fake entry fill or even a fake SL
                # hit. Binance's startTime happens to exclude it for the earliest case in each
                # symbol/mode batch, but not for the later ones sharing that same fetch window.
                relevant = [c for c in candles if c["open_time"] >= created]
                if not relevant:
                    continue

                current = float(row["current_price"] or relevant[0]["close"])
                direction = str(row["planner_direction"] or "NO_TRADE").upper()
                outcome = row["outcome"]
                tracking_status = "OPEN"
                entry_touched_at = row["entry_touched_at"] if "entry_touched_at" in row.keys() else None
                sl_touched_at = row["sl_touched_at"] if "sl_touched_at" in row.keys() else None
                tp1_touched_at = row["tp1_touched_at"] if "tp1_touched_at" in row.keys() else None
                mfe = row["max_favorable_price"]
                mae = row["max_adverse_price"]
                post_sl_mfe = row["post_sl_max_favorable_price"] if "post_sl_max_favorable_price" in row.keys() else None
                post_sl_mae = row["post_sl_max_adverse_price"] if "post_sl_max_adverse_price" in row.keys() else None
                post_sl_tp1_reached = int(row["post_sl_tp1_reached"] or 0) if "post_sl_tp1_reached" in row.keys() else 0
                post_sl_entry_recovered_at = row["post_sl_entry_recovered_at"] if "post_sl_entry_recovered_at" in row.keys() else None
                post_sl_diagnosis = row["post_sl_diagnosis"] if "post_sl_diagnosis" in row.keys() else None
                post_tp1_mfe = row["post_tp1_max_favorable_price"] if "post_tp1_max_favorable_price" in row.keys() else None
                post_tp1_mae = row["post_tp1_max_adverse_price"] if "post_tp1_max_adverse_price" in row.keys() else None
                post_tp1_tp2_reached = int(row["post_tp1_tp2_reached"] or 0) if "post_tp1_tp2_reached" in row.keys() else 0
                post_tp1_sl_hit = int(row["post_tp1_sl_hit"] or 0) if "post_tp1_sl_hit" in row.keys() else 0
                post_tp1_diagnosis = row["post_tp1_diagnosis"] if "post_tp1_diagnosis" in row.keys() else None

                if direction in {"LONG", "SHORT"} and row["entry_low"] is not None and row["entry_high"] is not None:
                    entry_low, entry_high = float(row["entry_low"]), float(row["entry_high"])
                    sl = float(row["sl"]) if row["sl"] is not None else None
                    tp1 = float(row["tp1"]) if row["tp1"] is not None else None
                    entry_index = None
                    entry_price = None

                    for idx, c in enumerate(relevant):
                        if c["low"] <= entry_high and c["high"] >= entry_low:
                            entry_index = idx
                            entry_touched_at = c["open_time"].isoformat()
                            if entry_low <= c["open"] <= entry_high:
                                entry_price = c["open"]
                            elif direction == "LONG":
                                entry_price = entry_high if c["open"] > entry_high else entry_low
                            else:
                                entry_price = entry_low if c["open"] < entry_low else entry_high
                            break

                    if entry_index is None:
                        if now >= entry_deadline:
                            outcome, tracking_status = "EXPIRED_NO_ENTRY", "CLOSED"
                    else:
                        post_entry = relevant[entry_index:]
                        first = post_entry[0]
                        opened_in_entry = entry_low <= first["open"] <= entry_high
                        first_sl = sl is not None and ((direction == "LONG" and first["low"] <= sl) or (direction == "SHORT" and first["high"] >= sl))
                        first_tp = tp1 is not None and ((direction == "LONG" and first["high"] >= tp1) or (direction == "SHORT" and first["low"] <= tp1))

                        if (first_sl or first_tp) and not opened_in_entry:
                            # Entry and exit both appear in the same candle, but the tick order is unknown.
                            outcome, tracking_status = "AMBIGUOUS_ENTRY_EXIT_SAME_CANDLE", "CLOSED"
                            if first_sl:
                                sl_touched_at = first["open_time"].isoformat()
                            if first_tp:
                                tp1_touched_at = first["open_time"].isoformat()
                        elif first_sl and first_tp:
                            outcome, tracking_status = "AMBIGUOUS_SAME_CANDLE", "CLOSED"
                            sl_touched_at = tp1_touched_at = first["open_time"].isoformat()
                        elif first_sl:
                            outcome, tracking_status = "SL_BEFORE_TP1", "POST_SL"
                            sl_touched_at = first["open_time"].isoformat()
                        elif first_tp:
                            outcome, tracking_status = "TP1_BEFORE_SL", "POST_TP1"
                            tp1_touched_at = first["open_time"].isoformat()
                        else:
                            for c in post_entry[1:]:
                                sl_hit = sl is not None and ((direction == "LONG" and c["low"] <= sl) or (direction == "SHORT" and c["high"] >= sl))
                                tp_hit = tp1 is not None and ((direction == "LONG" and c["high"] >= tp1) or (direction == "SHORT" and c["low"] <= tp1))
                                if sl_hit and tp_hit:
                                    outcome, tracking_status = "AMBIGUOUS_SAME_CANDLE", "CLOSED"
                                    sl_touched_at = tp1_touched_at = c["open_time"].isoformat()
                                    break
                                if sl_hit:
                                    outcome, tracking_status = "SL_BEFORE_TP1", "POST_SL"
                                    sl_touched_at = c["open_time"].isoformat()
                                    break
                                if tp_hit:
                                    outcome, tracking_status = "TP1_BEFORE_SL", "POST_TP1"
                                    tp1_touched_at = c["open_time"].isoformat()
                                    break
                            if tracking_status == "OPEN":
                                outcome = "ENTRY_TOUCHED"
                                _et = _parse_dt(entry_touched_at)
                                if _et and now >= _et + timedelta(hours=max_hold_hours):
                                    outcome, tracking_status = "EXPIRED_AFTER_ENTRY", "CLOSED"

                        # MFE/MAE are only calculated from when Entry was touched. For an intrabar
                        # entry candle, its full extremes aren't used since they could have occurred before Entry.
                        measurable = post_entry if opened_in_entry else post_entry[1:]
                        base = float(entry_price if entry_price is not None else current)
                        if measurable:
                            highs = [c["high"] for c in measurable]
                            lows = [c["low"] for c in measurable]
                            if direction == "SHORT":
                                mfe, mae = max(0.0, base - min(lows)), max(0.0, max(highs) - base)
                            else:
                                mfe, mae = max(0.0, max(highs) - base), max(0.0, base - min(lows))
                        else:
                            mfe, mae = 0.0, 0.0

                    # After SL is touched, tracking continues until expiry.
                    # The trade outcome doesn't change (SL_BEFORE_TP1); the post_sl_* columns
                    # are only used to diagnose "direction was right but SL was too tight".
                    if sl_touched_at:
                        sl_time = _parse_dt(sl_touched_at)
                        post_sl = [c for c in relevant if c["open_time"] >= sl_time]
                        if post_sl:
                            sl_base = float(sl if sl is not None else current)
                            highs_after = [c["high"] for c in post_sl]
                            lows_after = [c["low"] for c in post_sl]
                            if direction == "LONG":
                                post_sl_mfe = max(0.0, max(highs_after) - sl_base)
                                post_sl_mae = max(0.0, sl_base - min(lows_after))
                                recovered = next((c for c in post_sl if c["high"] >= entry_low), None)
                                tp_after = next((c for c in post_sl if tp1 is not None and c["high"] >= tp1), None)
                            else:
                                post_sl_mfe = max(0.0, sl_base - min(lows_after))
                                post_sl_mae = max(0.0, max(highs_after) - sl_base)
                                recovered = next((c for c in post_sl if c["low"] <= entry_high), None)
                                tp_after = next((c for c in post_sl if tp1 is not None and c["low"] <= tp1), None)

                            if recovered and not post_sl_entry_recovered_at:
                                post_sl_entry_recovered_at = recovered["open_time"].isoformat()
                            if tp_after:
                                post_sl_tp1_reached = 1
                                post_sl_diagnosis = "SL_THEN_TP1"
                            elif post_sl_entry_recovered_at:
                                post_sl_diagnosis = "SL_THEN_ENTRY_RECOVERED"
                            else:
                                post_sl_diagnosis = "SL_HIT_UNRESOLVED"

                        _et_sl = _parse_dt(entry_touched_at)
                        if _et_sl and now >= _et_sl + timedelta(hours=max_hold_hours):
                            tracking_status = "CLOSED"
                            if post_sl_tp1_reached:
                                post_sl_diagnosis = "SL_THEN_TP1"
                            elif post_sl_entry_recovered_at:
                                post_sl_diagnosis = "SL_THEN_ENTRY_RECOVERED"
                            else:
                                post_sl_diagnosis = "SL_AND_CONTINUED_WRONG_OR_UNRESOLVED"
                        else:
                            tracking_status = "POST_SL"
                # Post-TP1: diagnose what happened after TP1 was hit (not applicable to AMBIGUOUS cases).
                if outcome == "TP1_BEFORE_SL" and tp1_touched_at:
                    tp1_time = _parse_dt(tp1_touched_at)
                    post_tp1_candles = [c for c in relevant if c["open_time"] >= tp1_time]
                    if post_tp1_candles:
                        tp1_base = float(tp1 if tp1 is not None else current)
                        tp2_val = float(row["tp2"]) if row["tp2"] is not None else None
                        highs_after = [c["high"] for c in post_tp1_candles]
                        lows_after = [c["low"] for c in post_tp1_candles]
                        if direction == "LONG":
                            post_tp1_mfe = max(0.0, max(highs_after) - tp1_base)
                            post_tp1_mae = max(0.0, tp1_base - min(lows_after))
                            tp2_after = next((c for c in post_tp1_candles if tp2_val is not None and c["high"] >= tp2_val), None)
                            sl_after = next((c for c in post_tp1_candles if sl is not None and c["low"] <= sl), None)
                        else:
                            post_tp1_mfe = max(0.0, tp1_base - min(lows_after))
                            post_tp1_mae = max(0.0, max(highs_after) - tp1_base)
                            tp2_after = next((c for c in post_tp1_candles if tp2_val is not None and c["low"] <= tp2_val), None)
                            sl_after = next((c for c in post_tp1_candles if sl is not None and c["high"] >= sl), None)
                        if tp2_after:
                            post_tp1_tp2_reached = 1
                        if sl_after:
                            post_tp1_sl_hit = 1
                    _et_tp1 = _parse_dt(entry_touched_at)
                    if _et_tp1 and now >= _et_tp1 + timedelta(hours=max_hold_hours):
                        tracking_status = "CLOSED"
                        if post_tp1_tp2_reached:
                            post_tp1_diagnosis = "TP1_THEN_TP2"
                        elif post_tp1_sl_hit:
                            post_tp1_diagnosis = "TP1_THEN_REVERSED_TO_SL"
                        else:
                            post_tp1_diagnosis = "TP1_HELD"
                    else:
                        tracking_status = "POST_TP1"

                elif direction not in {"LONG", "SHORT"}:
                    # NO_TRADE only observes the objective price range from the decision point onward.
                    highs = [c["high"] for c in relevant]
                    lows = [c["low"] for c in relevant]
                    # Clamped like the LONG/SHORT branch above: these two columns are shared, so a
                    # negative value here (possible when the recorded current_price sits outside the
                    # observed range) would silently skew any aggregate over max_favorable/adverse.
                    mfe, mae = max(0.0, max(highs) - current), max(0.0, current - min(lows))
                    outcome = "NO_TRADE_OBSERVED"
                    if now >= expires:
                        tracking_status = "CLOSED"

                conn.execute("""
                    UPDATE evaluation_cases
                    SET tracking_status=?, outcome=?, max_favorable_price=?, max_adverse_price=?,
                        entry_touched_at=?, sl_touched_at=?, tp1_touched_at=?,
                        post_sl_max_favorable_price=?, post_sl_max_adverse_price=?,
                        post_sl_tp1_reached=?, post_sl_entry_recovered_at=?, post_sl_diagnosis=?,
                        post_tp1_max_favorable_price=?, post_tp1_max_adverse_price=?,
                        post_tp1_tp2_reached=?, post_tp1_sl_hit=?, post_tp1_diagnosis=?,
                        updated_at=?
                    WHERE id=?
                """, (tracking_status, outcome, mfe, mae, entry_touched_at, sl_touched_at, tp1_touched_at,
                      post_sl_mfe, post_sl_mae, post_sl_tp1_reached, post_sl_entry_recovered_at, post_sl_diagnosis,
                      post_tp1_mfe, post_tp1_mae, post_tp1_tp2_reached, post_tp1_sl_hit, post_tp1_diagnosis,
                      now.isoformat(), row["id"]))
                updated += 1
            conn.commit()
    return {"checked": len(rows), "updated": updated}

