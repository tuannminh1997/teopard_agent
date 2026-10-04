"""Quy tắc đợt 2: KHÔNG drop cột/bảng nào — DB cũ phải khởi động giữ nguyên."""
import os
import sqlite3
import tempfile

_TMP = tempfile.mkdtemp()
_OLD_DB = os.path.join(_TMP, "old_schema.db")
_FRESH_DB = os.path.join(_TMP, "fresh.db")

import analyze  # noqa: E402

OLD_PREDICTIONS_COLS = {
    "check_after_hours", "hold_hours", "market_snapshot", "feature_snapshot",
    "reasoning_summary", "full_response", "result_checked_at",
    "setup_status", "lifecycle_status", "mae", "mfe",
}
LEGACY_TABLES = {"predictions", "analysis_snapshots", "auto_scan_signals"}

OLD_PREDICTIONS_SCHEMA = """
    CREATE TABLE predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, chat_id INTEGER, symbol TEXT NOT NULL, mode TEXT NOT NULL,
        created_at TEXT NOT NULL, check_after_hours INTEGER NOT NULL DEFAULT 12,
        entry_wait_hours INTEGER NOT NULL DEFAULT 12, max_hold_hours INTEGER NOT NULL DEFAULT 72,
        next_check_at TEXT, direction TEXT NOT NULL, entry_low REAL, entry_high REAL,
        sl REAL, tp1 REAL, tp2 REAL, entry_status TEXT NOT NULL DEFAULT 'PENDING_ENTRY',
        entry_filled_at TEXT, entry_price REAL, trade_closed_at TEXT, rr_result REAL,
        hold_hours REAL, market_snapshot TEXT, feature_snapshot TEXT, reasoning_summary TEXT,
        full_response TEXT, result TEXT NOT NULL DEFAULT 'PENDING_ENTRY',
        result_price REAL, result_reason TEXT, result_checked_at TEXT,
        setup_status TEXT, lifecycle_status TEXT, mae REAL, mfe REAL
    )
"""


def _build_old_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute(OLD_PREDICTIONS_SCHEMA)
    conn.execute(
        "INSERT INTO predictions (symbol, mode, created_at, direction, setup_status, full_response) "
        "VALUES ('BTCUSDT','short','2026-01-01','LONG','READY_TO_ENTER','legacy text')"
    )
    conn.execute(
        "CREATE TABLE analysis_snapshots (id INTEGER PRIMARY KEY, created_at TEXT, symbol TEXT, setup_status TEXT)"
    )
    conn.execute(
        "INSERT INTO analysis_snapshots (created_at, symbol, setup_status) "
        "VALUES ('2026-01-01','BTC','SETUP_WAITING_TRIGGER')"
    )
    conn.execute(
        "CREATE TABLE auto_scan_signals (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, "
        "chat_id INTEGER, symbol TEXT, mode TEXT, direction TEXT, confidence INTEGER, "
        "sent_at TEXT, prediction_id INTEGER)"
    )
    conn.execute(
        "INSERT INTO auto_scan_signals (symbol, mode, direction, sent_at) "
        "VALUES ('BTCUSDT','short','LONG','2026-01-01')"
    )
    conn.commit()
    conn.close()


def _run_inits(monkeypatch, path: str) -> None:
    monkeypatch.setattr(analyze, "DB_PATH", path)
    monkeypatch.setattr(analyze, "_prediction_db_initialized", False)
    monkeypatch.setattr(analyze, "_auto_scan_db_initialized", False)
    analyze.init_prediction_db()
    analyze.init_auto_scan_db()
    analyze._ensure_trend_state_table()


def test_old_db_boot_preserves_all_columns_and_tables(monkeypatch):
    _build_old_db(_OLD_DB)
    _run_inits(monkeypatch, _OLD_DB)
    conn = sqlite3.connect(_OLD_DB)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(predictions)")}
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    pred_row = conn.execute("SELECT setup_status, full_response FROM predictions").fetchone()
    snap_row = conn.execute("SELECT setup_status FROM analysis_snapshots").fetchone()
    sig_count = conn.execute("SELECT COUNT(*) FROM auto_scan_signals").fetchone()[0]
    conn.close()
    assert OLD_PREDICTIONS_COLS <= cols, f"mất cột: {OLD_PREDICTIONS_COLS - cols}"
    assert LEGACY_TABLES <= tables, f"mất bảng: {LEGACY_TABLES - tables}"
    assert pred_row == ("READY_TO_ENTER", "legacy text")
    assert snap_row == ("SETUP_WAITING_TRIGGER",)
    assert sig_count == 1


def test_fresh_db_gets_original_schema(monkeypatch):
    if os.path.exists(_FRESH_DB):
        os.remove(_FRESH_DB)
    _run_inits(monkeypatch, _FRESH_DB)
    conn = sqlite3.connect(_FRESH_DB)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(predictions)")}
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert OLD_PREDICTIONS_COLS <= cols
    assert LEGACY_TABLES <= tables


def test_source_has_no_destructive_schema_statements():
    root = os.path.join(os.path.dirname(__file__), "..")
    # auth.py có lệnh DROP TABLE whitelist_old là MIGRATION GỐC (đời đầukhông do đợt thêm) → không quét.
    for fname in ("analyze.py", "evaluation_store.py", "symbol_control.py"):
        src = open(os.path.join(root, fname), encoding="utf-8").read()
        assert "DROP COLUMN" not in src, fname
        assert "DROP TABLE" not in src, fname
