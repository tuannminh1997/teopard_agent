"""Giai đoạn 1.5: /clearhistory và rollback phải thực sự xóa dữ liệu trên DB mẫu."""
import os
import sqlite3
import tempfile
import uuid

_TMP = tempfile.mkdtemp()
_DB = None  # mỗi test một file riêng (style sqlite3 `with connect` của codebase không đóng ngay)

import analyze  # noqa: E402
import auth  # noqa: E402
import evaluation_store  # noqa: E402
import symbol_control  # noqa: E402


def _fresh(monkeypatch) -> str:
    global _DB
    _DB = os.path.join(_TMP, f"clear_{uuid.uuid4().hex}.db")
    monkeypatch.setattr(analyze, "DB_PATH", _DB)
    monkeypatch.setattr(evaluation_store, "DB_PATH", _DB)
    monkeypatch.setattr(auth, "DB_PATH", _DB)
    monkeypatch.setattr(symbol_control, "DB_PATH", _DB)
    monkeypatch.setattr(analyze, "_prediction_db_initialized", False)
    monkeypatch.setattr(analyze, "_auto_scan_db_initialized", False)
    analyze.init_prediction_db()
    analyze.init_auto_scan_db()
    analyze._ensure_trend_state_table()
    evaluation_store.init_evaluation_db()
    auth.init_auth_db()
    symbol_control.init_symbol_db()


def _count(table: str) -> int:
    with sqlite3.connect(_DB) as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _seed() -> None:
    with sqlite3.connect(_DB) as conn:
        conn.execute(
            "INSERT INTO predictions (symbol, mode, created_at, direction, result) "
            "VALUES ('BTCUSDT','short','2026-01-01','LONG','WIN')"
        )
        conn.execute(
            "INSERT INTO auto_scan_signals (user_id, chat_id, symbol, mode, direction, sent_at, prediction_id) "
            "VALUES (1, 1, 'BTCUSDT', 'short', 'LONG', '2026-01-01', 77)"
        )
        conn.execute(
            "INSERT INTO auto_scan_signals (user_id, chat_id, symbol, mode, direction, sent_at, prediction_id) "
            "VALUES (1, 1, 'ETHUSDT', 'short', 'SHORT', '2026-01-01', 88)"
        )
        conn.execute(
            "INSERT INTO auto_scan_logs (user_id, chat_id, symbol, mode, scanned_at, stage, status) "
            "VALUES (1, 1, 'BTCUSDT', 'short', '2026-01-01', 'sent', 'sent')"
        )
        conn.execute(
            "INSERT INTO analysis_snapshots (created_at, symbol, mode, source) "
            "VALUES ('2026-01-01', 'BTCUSDT', 'short', 'manual')"
        )
        conn.execute(
            "INSERT INTO auto_scan_trend_state (user_id, symbol, mode, last_direction, updated_at) "
            "VALUES (1, 'BTCUSDT', 'short', 'LONG', '2026-01-01')"
        )
        conn.execute(
            "INSERT INTO evaluation_cases (created_at, source, symbol, mode, pipeline_phase, final_result, updated_at) "
            "VALUES ('2026-01-01', 'manual', 'BTCUSDT', 'short', 'PLANNER_APPROVED', 'LONG', '2026-01-01')"
        )
        # Cấu hình phải được GIỮ sau /clearhistory.
        conn.execute("INSERT INTO whitelist (user_id, daily_limit, used_today, last_reset_date) VALUES (9, 10, 0, '')")
        conn.execute("INSERT INTO allowed_symbols (symbol) VALUES ('BTC')")
        conn.execute(
            "INSERT INTO auto_scan_settings (user_id, chat_id, enabled, symbols, updated_at) "
            "VALUES (9, 9, 1, 'BTCUSDT', '2026-01-01')"
        )
        conn.commit()


def test_clearhistory_really_deletes_history_keeps_config(monkeypatch):
    _fresh(monkeypatch)
    _seed()
    for table in ("predictions", "auto_scan_signals", "auto_scan_logs",
                  "analysis_snapshots", "auto_scan_trend_state", "evaluation_cases"):
        assert _count(table) >= 1, table

    payload = analyze.clear_prediction_history()
    assert payload["total_prediction_count"] == 1
    assert payload["evaluation_count"] == 1

    for table in ("predictions", "auto_scan_signals", "auto_scan_logs",
                  "analysis_snapshots", "auto_scan_trend_state", "evaluation_cases"):
        assert _count(table) == 0, f"clearhistory không xóa {table}"
    # Cấu hình không bị đụng.
    assert _count("whitelist") == 1
    assert _count("allowed_symbols") == 1
    assert _count("auto_scan_settings") == 1


def test_rollback_really_deletes_signal_row(monkeypatch):
    _fresh(monkeypatch)
    _seed()
    assert _count("auto_scan_signals") == 2

    analyze._rollback_auto_scan_signal(77)
    with sqlite3.connect(_DB) as conn:
        left = {r[0] for r in conn.execute("SELECT prediction_id FROM auto_scan_signals")}
    assert left == {88}, f"rollback phải xóa đúng dòng prediction_id=77, còn {left}"

    # prediction_id=None là no-op, không xóa gì và không crash.
    analyze._rollback_auto_scan_signal(None)
    assert _count("auto_scan_signals") == 1
