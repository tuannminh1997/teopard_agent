"""Regression: các fix an toàn tiền/ledger trong đợt nâng version 4.0.

Mỗi test tương ứng một bug đã sửa — nếu ai đó gỡ lại phần fix thì test này đỏ.
"""
import os
import sqlite3
import tempfile

import pytest

import analyze
import binance_executor
import key_store
import symbol_control

_TMP = tempfile.mkdtemp()
_DB = os.path.join(_TMP, "v4_test.db")


def _analyze(monkeypatch):
    monkeypatch.setattr(analyze, "DB_PATH", _DB)
    monkeypatch.setattr(analyze, "_prediction_db_initialized", False)
    monkeypatch.setattr(analyze, "_auto_scan_db_initialized", False)
    return analyze


# ─── Lệnh lỗi mạng không được bỏ qua cleanup ─────────────────────────────────

def test_network_error_becomes_executor_error(monkeypatch):
    """Timeout/reset từng thoát thô → các `except ExecutorError` bỏ qua cleanup → position trần."""
    import requests

    def boom(*a, **k):
        raise requests.Timeout("timed out")

    monkeypatch.setattr(binance_executor.requests, "get", boom)
    monkeypatch.setattr(binance_executor.requests, "post", boom)

    for method in ("GET", "POST"):
        try:
            binance_executor.signed_request("https://x", "k", "s", method, "/p", {})
            assert False, "phải ném ExecutorError"
        except binance_executor.ExecutorError as exc:
            assert exc.code == -1000


def test_network_error_during_entry_cancels_soft(monkeypatch):
    """Mất kết nối đúng lúc gửi entry → lệnh có thể đã lên sàn → phải cancel best-effort."""
    calls = []

    def fake(base, api_key, secret, method, path, params):
        calls.append((method, path, dict(params)))
        if method == "POST" and path.endswith("/fapi/v1/order"):
            raise binance_executor.ExecutorError(-1000, "lỗi mạng")
        if path.endswith("positionSide/dual"):
            return {"dualSidePosition": False}
        if path.endswith("/fapi/v1/leverage"):
            return {}
        return {}

    monkeypatch.setattr(binance_executor, "signed_request", fake)
    monkeypatch.setattr(
        binance_executor, "_market_filters",
        lambda *a, **k: {"tick": 0.01, "step": 0.001, "min_notional": 5},
    )

    with pytest.raises(binance_executor.ExecutorError):
        binance_executor.place_futures_plan(
            "ETHUSDT", "LONG", 2700.0, 2750.0, 2680.0, 0.01, 20, ("k", "s"), "futu-eth-1")

    cancels = [p for m, path, p in calls if m == "DELETE"]
    assert cancels, "phải thử hủy lệnh entry theo clientOrderId"
    assert cancels[0]["origClientOrderId"] == "futu-eth-1-e"


def test_partial_fill_cancels_remainder_before_close(monkeypatch):
    """Khớp nửa chừng: phải hủy phần entry còn lại TRƯỚC khi đóng, không thì khớp thêm sau."""
    calls = []

    def fake(base, api_key, secret, method, path, params):
        calls.append((method, path, dict(params)))
        if path.endswith("/fapi/v1/algoOrder"):
            raise binance_executor.ExecutorError(-2021, "would immediately trigger")
        if path.endswith("positionSide/dual"):
            return {"dualSidePosition": True}
        if method == "GET" and path.endswith("/fapi/v1/order"):
            return {"status": "PARTIALLY_FILLED", "executedQty": "0.004"}
        if method == "POST" and path.endswith("/fapi/v1/order"):
            return {"orderId": 555}
        return {}

    monkeypatch.setattr(binance_executor, "signed_request", fake)
    monkeypatch.setattr(
        binance_executor, "_market_filters",
        lambda *a, **k: {"tick": 0.01, "step": 0.001, "min_notional": 5},
    )

    with pytest.raises(binance_executor.ExecutorError) as exc_info:
        binance_executor.place_futures_plan(
            "ETHUSDT", "LONG", 2700.0, 2750.0, 2680.0, 0.01, 20, ("k", "s"), "futu-eth-1")
    assert "ĐÓNG NGAY" in exc_info.value.msg

    idx_delete = next(i for i, (m, p, _) in enumerate(calls)
                      if m == "DELETE" and p.endswith("/fapi/v1/order"))
    idx_close = next(i for i, (m, p, prm) in enumerate(calls)
                     if m == "POST" and p.endswith("/fapi/v1/order") and prm.get("type") == "MARKET")
    assert idx_delete < idx_close, "phải hủy phần entry còn lại trước khi đóng position"
    assert float(calls[idx_close][2]["quantity"]) == 0.004


def test_orphan_tp_algo_cancelled_when_sl_fails(monkeypatch):
    """TP đặt xong mà SL fail → TP algo phải bị gỡ, nếu không nó vẫn ARMED và tự đóng lệnh sau."""
    calls = []

    def fake(base, api_key, secret, method, path, params):
        calls.append((method, path, dict(params)))
        if path.endswith("/fapi/v1/algoOrder"):
            if params.get("type") == "STOP_MARKET":
                raise binance_executor.ExecutorError(-2021, "reject")
            return {"algoId": 111}
        if path.endswith("positionSide/dual"):
            return {"dualSidePosition": False}
        if method == "GET" and path.endswith("/fapi/v1/order"):
            return {"status": "NEW", "executedQty": "0"}
        if method == "POST" and path.endswith("/fapi/v1/order"):
            return {"orderId": 555}
        return {}

    monkeypatch.setattr(binance_executor, "signed_request", fake)
    monkeypatch.setattr(
        binance_executor, "_market_filters",
        lambda *a, **k: {"tick": 0.01, "step": 0.001, "min_notional": 5},
    )

    with pytest.raises(binance_executor.ExecutorError):
        binance_executor.place_futures_plan(
            "ETHUSDT", "LONG", 2700.0, 2750.0, 2680.0, 0.01, 20, ("k", "s"), "futu-eth-1")

    removed = [p["algoid"] for m, path, p in calls
               if m == "DELETE" and path.endswith("/fapi/v1/algoOrder")]
    assert removed == [111], "phải gỡ TP algo đã tạo được"


def test_spot_entry_resp_none_guards(monkeypatch):
    """6 lần trùng clientOrderId hết mà không ra response → trước đây ném AttributeError."""
    def fake(base, api_key, secret, method, path, params):
        if path.endswith("/api/v3/order") and method == "POST":
            raise binance_executor.ExecutorError(-2010, "Duplicate order sent.")
        if path.endswith("positionSide/dual"):
            return {"dualSidePosition": False}
        return {}

    monkeypatch.setattr(binance_executor, "signed_request", fake)
    monkeypatch.setattr(
        binance_executor, "_market_filters",
        lambda *a, **k: {"tick": 0.01, "step": 0.001, "min_notional": 5},
    )

    with pytest.raises(binance_executor.ExecutorError) as exc_info:
        binance_executor.place_spot_plan("ETHUSDT", 2700.0, 2750.0, 2680.0, 0.01,
                                         ("k", "s"), "spot-eth-1")
    assert exc_info.value.code == -2010


def test_spot_unfilled_entry_is_cancelled(monkeypatch):
    """Entry không khớp sau timeout vẫn treo GTC → sau này khớp lúc bot không theo dõi = position trần."""
    calls = []

    def fake(base, api_key, secret, method, path, params):
        calls.append((method, path, dict(params)))
        if path.endswith("/api/v3/order") and method == "GET":
            return {"status": "NEW", "executedQty": "0"}
        if path.endswith("/api/v3/order") and method == "POST":
            return {"orderId": 7}
        if path.endswith("/api/v3/order") and method == "DELETE":
            return {"status": "CANCELED"}
        return {}

    monkeypatch.setattr(binance_executor, "signed_request", fake)
    monkeypatch.setattr(
        binance_executor, "_market_filters",
        lambda *a, **k: {"tick": 0.01, "step": 0.001, "min_notional": 5},
    )

    result = binance_executor.place_spot_plan(
        "ETHUSDT", 2700.0, 2750.0, 2680.0, 0.01, ("k", "s"), "spot-eth-1",
        fill_timeout=0, poll_seconds=0)

    assert result["filled"] is False
    assert result["remainder_cancelled"] is True
    assert any(m == "DELETE" for m, path, _ in calls), "phải hủy entry chưa khớp"


def test_spot_notional_checked(monkeypatch):
    """Spot trước đây không đọc MIN_NOTIONAL → lệnh dưới mức tối thiểu bị sàn từ chối."""
    monkeypatch.setattr(
        binance_executor, "_market_filters",
        lambda *a, **k: {"tick": 0.01, "step": 0.001, "min_notional": 100},
    )
    with pytest.raises(binance_executor.ExecutorError) as exc_info:
        binance_executor.place_spot_plan("ETHUSDT", 10.0, 12.0, 9.0, 0.01,
                                         ("k", "s"), "spot-eth-1")
    assert exc_info.value.code == -4164


# ─── Ledger không bị xóa nhầm khi lệnh còn sống trên sàn ─────────────────────

def test_rollback_keeps_row_with_placed_orders(monkeypatch):
    """Gửi Telegram fail → không được DELETE dòng đã đặt lệnh: đó là bản ghi duy nhất về orderId."""
    a = _analyze(monkeypatch)
    a.init_auto_scan_db()
    conn = sqlite3.connect(a.DB_PATH)
    conn.execute("DELETE FROM auto_scan_signals")
    conn.execute(
        "INSERT INTO auto_scan_signals (user_id, chat_id, symbol, mode, direction, sent_at,"
        " prediction_id, plan_id, order_status) VALUES (7,1,'ETHUSDT','futures','LONG',"
        " '2026-10-06T00:00:00+00:00', 9001, 'futu-eth-1', 'placed')"
    )
    conn.execute(
        "INSERT INTO auto_scan_signals (user_id, chat_id, symbol, mode, direction, sent_at,"
        " prediction_id, plan_id, order_status) VALUES (7,1,'BTCUSDT','futures','SHORT',"
        " '2026-10-06T00:00:00+00:00', 9002, 'futu-btc-1', 'pending')"
    )
    conn.commit()
    conn.close()

    a._rollback_auto_scan_signal(9001)   # đã đặt lệnh → GIỮ
    a._rollback_auto_scan_signal(9002)   # chưa đặt → xóa được

    rows = {r["prediction_id"]: r["order_status"] for r in a.list_session_signals(7, "futures")}
    assert rows.get(9001) == "send_failed", "dòng có lệnh trên sàn phải được giữ lại"
    assert 9002 not in rows, "dòng chưa đặt lệnh mới được rollback"


def test_update_signal_orders_scoped_by_prediction_id(monkeypatch):
    """plan_id là 'futu-eth-1' ở mọi user → update không scope sẽ ghi đè orderId sang user khác."""
    a = _analyze(monkeypatch)
    a.init_auto_scan_db()
    conn = sqlite3.connect(a.DB_PATH)
    conn.execute("DELETE FROM auto_scan_signals")
    for uid, pid in ((7, 9101), (8, 9102)):
        conn.execute(
            "INSERT INTO auto_scan_signals (user_id, chat_id, symbol, mode, direction, sent_at,"
            " prediction_id, plan_id, order_status) VALUES (?,1,'ETHUSDT','futures','LONG',"
            " '2026-10-06T00:00:00+00:00', ?, 'futu-eth-1', 'pending')",
            (uid, pid),
        )
    conn.commit()
    conn.close()

    updated = a.update_signal_orders("futu-eth-1", prediction_id=9101, entry_order_id=4242)
    assert updated == 1, "chỉ được update đúng 1 dòng"

    conn = sqlite3.connect(a.DB_PATH)
    conn.row_factory = sqlite3.Row
    got = {r["prediction_id"]: r["entry_order_id"]
           for r in conn.execute("SELECT prediction_id, entry_order_id FROM auto_scan_signals")}
    conn.close()
    assert got[9101] == "4242"
    assert got[9102] is None, "không được ghi đè orderId sang dòng của user khác"


def test_delete_session_signals_can_keep_placed(monkeypatch):
    """Thiếu key → /off* không hủy được lệnh; nếu xóa luôn dòng placed thì mất mỗi đường về orderId."""
    a = _analyze(monkeypatch)
    a.init_auto_scan_db()
    conn = sqlite3.connect(a.DB_PATH)
    conn.execute("DELETE FROM auto_scan_signals")
    conn.execute(
        "INSERT INTO auto_scan_signals (user_id, chat_id, symbol, mode, direction, sent_at,"
        " prediction_id, plan_id, order_status) VALUES (7,1,'ETHUSDT','futures','LONG',"
        " '2026-10-06T00:00:00+00:00', 9201, 'futu-eth-1', 'placed')"
    )
    conn.execute(
        "INSERT INTO auto_scan_signals (user_id, chat_id, symbol, mode, direction, sent_at,"
        " prediction_id, plan_id, order_status) VALUES (7,1,'ETHUSDT','futures','LONG',"
        " '2026-10-06T00:00:00+00:00', 9202, 'futu-eth-2', 'pending')"
    )
    conn.commit()
    conn.close()

    deleted = a.delete_session_signals(7, "futures", keep_placed=True)
    assert deleted == 1
    remaining = a.list_session_signals(7, "futures")
    assert [r["prediction_id"] for r in remaining] == [9201]


# ─── Prune không xóa lệnh còn đang mở ────────────────────────────────────────

def test_prune_never_deletes_open_predictions(monkeypatch):
    """Prune từng xóa cả PENDING_ENTRY/ENTRY_FILLED → job auto-check hết thấy vị thế."""
    a = _analyze(monkeypatch)
    a.init_prediction_db()
    conn = sqlite3.connect(a.DB_PATH)
    conn.execute("DELETE FROM predictions")
    # 10 dòng terminal (đủ để lấp limit) + 1 lệnh đang mở CŨ NHẤT (id nhỏ nhất).
    for i in range(10):
        conn.execute(
            "INSERT INTO predictions (user_id, chat_id, symbol, mode, created_at, direction, result)"
            " VALUES (5,5,'BTCUSDT','futures',?,'LONG','WIN')",
            (f"2026-10-{i + 1:02d}T00:00:00+00:00",),
        )
    conn.execute(
        "INSERT INTO predictions (user_id, chat_id, symbol, mode, created_at, direction, result)"
        " VALUES (5,5,'ETHUSDT','futures','2026-01-01T00:00:00+00:00','LONG','ENTRY_FILLED')"
    )
    conn.commit()
    conn.close()

    a.prune_prediction_history(5)

    conn = sqlite3.connect(a.DB_PATH)
    open_rows = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE result IN ('PENDING_ENTRY','ENTRY_FILLED')"
    ).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    conn.close()
    assert open_rows == 1, "lệnh đang mở phải sống sót qua prune"
    assert total == 11, "prune chỉ cắt 10 dòng terminal cũ nhất"


# ─── Nhập liệu key/qty ───────────────────────────────────────────────────────

def test_parse_qty_treats_comma_as_decimal(monkeypatch):
    """"0,97" từng bị replace thành 97 (~100 lần khối lượng dự kiến)."""
    assert symbol_control._parse_qty("0,97") == pytest.approx(0.97)
    assert symbol_control._parse_qty("0.008") == pytest.approx(0.008)
    assert symbol_control._parse_qty("1,000") == pytest.approx(1.0)
    assert symbol_control._parse_qty("1.000,50") == pytest.approx(1000.50)
    assert symbol_control._parse_qty("-1") is None
    assert symbol_control._parse_qty("abc") is None
    assert symbol_control._parse_qty("inf") is None


def test_fmt_qty_roundtrip(monkeypatch):
    """qty lưu xuống DB phải là số thuần để lần đặt lệnh sau đọc đúng."""
    assert float(symbol_control._fmt_qty(0.97)) == pytest.approx(0.97)
    assert symbol_control._fmt_qty(0.97).replace(".", "").isdigit()


def test_pending_state_expires(monkeypatch):
    """State nhập dở dang không hết hạn → tin nhắn giờ này bị coi là API key và lưu đè key tốt."""
    symbol_control._AUTO_PENDING.clear()
    try:
        symbol_control._new_pending(123, stage="api_key", market="futures", symbol="ETHUSDT")
        state = symbol_control._AUTO_PENDING[123]
        state["ts"] -= symbol_control._PENDING_TTL_SECONDS + 1

        import asyncio
        from types import SimpleNamespace

        class _Msg:
            text = "một tin nhắn thường"
            async def delete(self):
                return True
            async def reply_text(self, *a, **k):
                self.replied = True

        msg = _Msg()

        async def run():
            await symbol_control.autoscan_pending_message(
                SimpleNamespace(effective_user=SimpleNamespace(id=123), effective_message=msg),
                None,
            )

        asyncio.run(run())
        assert 123 not in symbol_control._AUTO_PENDING, "state hết hạn phải bị xóa"
        assert getattr(msg, "replied", False), "phải báo cho user biết phiên đã hết hạn"
    finally:
        symbol_control._AUTO_PENDING.clear()


# ─── Cửa sổ ngủ / API key status ─────────────────────────────────────────────

def test_sleep_equals_wake_is_not_permanent_sleep(monkeypatch):
    """sleep == wake từng làm biểu thức luôn đúng → autoscan tắt vĩnh viễn, không cảnh báo."""
    a = _analyze(monkeypatch)
    a.init_auto_scan_db()
    monkeypatch.setattr(a, "AUTOSCAN_SLEEP_HOUR_VN", 0)
    monkeypatch.setattr(a, "AUTOSCAN_WAKE_HOUR_VN", 0)

    from datetime import datetime, timedelta, timezone
    w = a.maintain_auto_scan_daily_window(
        datetime(2026, 10, 6, 3, 0, tzinfo=timezone(timedelta(hours=7))), allow_wipe=False)
    assert w["in_sleep_window"] is False, "không cấu hình nghĩa là KHÔNG có cửa sổ ngủ"


def test_api_key_status_detects_undecryptable_key(monkeypatch):
    """Key dòng tồn tại nhưng không giải mã được → UI từng vẫn hiện 'Đã Thêm'."""
    db = os.path.join(_TMP, "keystore_v4.db")
    monkeypatch.setattr(key_store, "DB_PATH", db)
    from cryptography.fernet import Fernet
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    key_store.save_api_keys(31, "futures", "APIKEY" + "x" * 30, "SECRET" + "y" * 30)
    assert key_store.api_key_status(31, "futures") == "ok"
    assert key_store.api_key_status(31, "spot") == "missing"

    # Đổi khóa mã hóa → dòng vẫn nằm trong DB nhưng không đọc được.
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    assert key_store.has_api_keys(31, "futures") is True
    assert key_store.api_key_status(31, "futures") == "broken"


# ─── Cửa sổ ngủ / status không wipe ──────────────────────────────────────────

def test_status_command_never_wipes_ledger(monkeypatch):
    """/autoscanstatus là lệnh CHỈ ĐỌC — không được hủy lệnh + xóa ledger của mọi user."""
    a = _analyze(monkeypatch)
    a.init_auto_scan_db()
    conn = sqlite3.connect(a.DB_PATH)
    conn.execute("DELETE FROM auto_scan_signals")
    conn.execute("DELETE FROM auto_scan_state")
    conn.execute(
        "INSERT INTO auto_scan_signals (user_id, chat_id, symbol, mode, direction, sent_at,"
        " prediction_id, plan_id, order_status) VALUES (7,1,'ETHUSDT','futures','LONG',"
        " '2026-10-06T00:00:00+00:00', 9301, 'futu-eth-1', 'pending')"
    )
    conn.commit()
    conn.close()

    a.get_auto_scan_runtime_status(7)

    assert len(a.list_session_signals(7, "futures")) == 1, "lệnh status không được wipe ledger"
