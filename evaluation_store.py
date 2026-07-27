import gzip
import hashlib
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = os.getenv("DB_PATH", "bot.db")
EVALUATION_ENABLED = os.getenv("EVALUATION_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
EVALUATION_FULL_RETENTION_DAYS = max(7, int(os.getenv("EVALUATION_FULL_RETENTION_DAYS", "60")))
EVALUATION_METADATA_RETENTION_DAYS = max(EVALUATION_FULL_RETENTION_DAYS, int(os.getenv("EVALUATION_METADATA_RETENTION_DAYS", "180")))
AUTO_SCAN_LOG_RETENTION_DAYS = max(1, int(os.getenv("AUTO_SCAN_LOG_RETENTION_DAYS", "14")))
BOT_VERSION = os.getenv("BOT_VERSION", "1.0")


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
    hours = 12 if mode == "short" else 72
    expires_at = (now + timedelta(hours=hours)).isoformat()
    fields = [
        "created_at","user_id","chat_id","source","symbol","mode","pipeline_phase","final_result",
        "current_price","prefilter_long_score","prefilter_short_score","prefilter_direction","bias_window",
        "planner_direction","planner_status","reviewer_score","reviewer_verdict","entry_low","entry_high",
        "sl","tp1","tp2","market_packet_gzip","planner_output_gzip","reviewer_output_gzip",
        "public_output_gzip","bot_version","planner_prompt_hash","reviewer_prompt_hash","prefilter_prompt_hash",
        "tracking_status","outcome","expires_at","updated_at"
    ]
    values = [
        now.isoformat(), kwargs.get("user_id"), kwargs.get("chat_id"), kwargs.get("source"), kwargs.get("symbol"), mode,
        kwargs.get("pipeline_phase") or "UNKNOWN", kwargs.get("final_result") or "UNKNOWN", kwargs.get("current_price"),
        kwargs.get("prefilter_long_score"), kwargs.get("prefilter_short_score"), kwargs.get("prefilter_direction"), kwargs.get("bias_window"),
        kwargs.get("planner_direction"), kwargs.get("planner_status"), kwargs.get("reviewer_score"), kwargs.get("reviewer_verdict"),
        kwargs.get("entry_low"), kwargs.get("entry_high"), kwargs.get("sl"), kwargs.get("tp1"), kwargs.get("tp2"),
        _compress(kwargs.get("market_packet")), _compress(kwargs.get("planner_output")), _compress(kwargs.get("reviewer_output")),
        _compress(kwargs.get("public_output")), BOT_VERSION, kwargs.get("planner_prompt_hash"), kwargs.get("reviewer_prompt_hash"),
        kwargs.get("prefilter_prompt_hash"), "OPEN", None, expires_at, now.isoformat()
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
    log_cutoff = (now - timedelta(days=AUTO_SCAN_LOG_RETENTION_DAYS)).isoformat()
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
        # Bảng legacy từng lưu trùng full packet; bản mới không ghi thêm và dọn theo retention.
        try:
            conn.execute("DELETE FROM analysis_snapshots WHERE created_at < ?", (full_cutoff,))
        except sqlite3.OperationalError:
            pass
        conn.commit()


def export_database_snapshot(destination: str) -> str:
    dest = str(Path(destination))
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(DB_PATH)
    dst = sqlite3.connect(dest)
    try:
        with dst:
            src.backup(dst)
    finally:
        dst.close()
        src.close()
    return dest

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


def _parse_dt(value: str) -> datetime:
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
    """Theo dõi khách quan bằng OHLC đã đóng; không gọi AI và không suy diễn thứ tự nội nến."""
    if not EVALUATION_ENABLED:
        return {"checked": 0, "updated": 0}
    init_evaluation_db()
    now = datetime.now(timezone.utc)
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT * FROM evaluation_cases
            WHERE tracking_status IN ('OPEN','POST_SL') AND created_at >= ?
            ORDER BY created_at ASC
            LIMIT 200
        """, ((now - timedelta(days=7)).isoformat(),)).fetchall()

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
        # Không dùng nến đang mở để kết luận chạm mức.
        candles = [c for c in candles if c["close_time"] is None or c["close_time"] <= now]

        with sqlite3.connect(DB_PATH) as conn:
            for row in cases:
                created = _parse_dt(row["created_at"])
                expires = _parse_dt(row["expires_at"])
                relevant = [c for c in candles if (c["close_time"] or c["open_time"]) >= created]
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
                        if now >= expires:
                            outcome, tracking_status = "EXPIRED_NO_ENTRY", "CLOSED"
                    else:
                        post_entry = relevant[entry_index:]
                        first = post_entry[0]
                        opened_in_entry = entry_low <= first["open"] <= entry_high
                        first_sl = sl is not None and ((direction == "LONG" and first["low"] <= sl) or (direction == "SHORT" and first["high"] >= sl))
                        first_tp = tp1 is not None and ((direction == "LONG" and first["high"] >= tp1) or (direction == "SHORT" and first["low"] <= tp1))

                        if (first_sl or first_tp) and not opened_in_entry:
                            # Entry và exit cùng xuất hiện trong một nến nhưng không biết thứ tự tick.
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
                            outcome, tracking_status = "TP1_BEFORE_SL", "CLOSED"
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
                                    outcome, tracking_status = "TP1_BEFORE_SL", "CLOSED"
                                    tp1_touched_at = c["open_time"].isoformat()
                                    break
                            if tracking_status == "OPEN":
                                outcome = "ENTRY_TOUCHED"
                                if now >= expires:
                                    outcome, tracking_status = "EXPIRED_AFTER_ENTRY", "CLOSED"

                        # MFE/MAE chỉ tính từ lúc Entry được chạm. Với nến entry intrabar,
                        # không dùng toàn bộ cực trị của nến đó vì có thể xảy ra trước Entry.
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

                    # Sau khi SL đã chạm, vẫn tiếp tục quan sát đến expiry.
                    # Kết quả giao dịch không đổi (SL_BEFORE_TP1); các cột post_sl_*
                    # chỉ dùng để chẩn đoán hướng đúng nhưng SL có thể quá sát.
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

                        if now >= expires:
                            tracking_status = "CLOSED"
                            if post_sl_tp1_reached:
                                post_sl_diagnosis = "SL_THEN_TP1"
                            elif post_sl_entry_recovered_at:
                                post_sl_diagnosis = "SL_THEN_ENTRY_RECOVERED"
                            else:
                                post_sl_diagnosis = "SL_AND_CONTINUED_WRONG_OR_UNRESOLVED"
                        else:
                            tracking_status = "POST_SL"
                else:
                    # NO_TRADE chỉ quan sát biên độ khách quan từ thời điểm quyết định.
                    highs = [c["high"] for c in relevant]
                    lows = [c["low"] for c in relevant]
                    mfe, mae = max(highs) - current, current - min(lows)
                    outcome = "NO_TRADE_OBSERVED"
                    if now >= expires:
                        tracking_status = "CLOSED"

                conn.execute("""
                    UPDATE evaluation_cases
                    SET tracking_status=?, outcome=?, max_favorable_price=?, max_adverse_price=?,
                        entry_touched_at=?, sl_touched_at=?, tp1_touched_at=?,
                        post_sl_max_favorable_price=?, post_sl_max_adverse_price=?,
                        post_sl_tp1_reached=?, post_sl_entry_recovered_at=?, post_sl_diagnosis=?,
                        updated_at=?
                    WHERE id=?
                """, (tracking_status, outcome, mfe, mae, entry_touched_at, sl_touched_at, tp1_touched_at,
                      post_sl_mfe, post_sl_mae, post_sl_tp1_reached, post_sl_entry_recovered_at, post_sl_diagnosis,
                      now.isoformat(), row["id"]))
                updated += 1
            conn.commit()
    return {"checked": len(rows), "updated": updated}

