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
        lambda *a, **k: {"tick": 0.01, "step": 0.001, "min_notional": 5,
                         "min_qty": 0.00001, "base_asset": "BTC"},
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
        lambda *a, **k: {"tick": 0.01, "step": 0.001, "min_notional": 5,
                         "min_qty": 0.00001, "base_asset": "BTC"},
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
        lambda *a, **k: {"tick": 0.01, "step": 0.001, "min_notional": 5,
                         "min_qty": 0.00001, "base_asset": "BTC"},
    )

    with pytest.raises(binance_executor.ExecutorError):
        binance_executor.place_futures_plan(
            "ETHUSDT", "LONG", 2700.0, 2750.0, 2680.0, 0.01, 20, ("k", "s"), "futu-eth-1")

    removed = [p["algoId"] for m, path, p in calls
               if m == "DELETE" and path.endswith("/fapi/v1/algoOrder")]
    assert removed == [111], "phải gỡ TP algo đã tạo được"


def test_spot_entry_resp_none_guards(monkeypatch):
    """6 lần trùng clientOrderId hết mà không ra response → trước đây ném AttributeError."""
    def fake(base, api_key, secret, method, path, params):
        if method == "POST" and path.endswith(("order", "otoco")):
            raise binance_executor.ExecutorError(-2010, "Duplicate order sent.")
        if path.endswith("positionSide/dual"):
            return {"dualSidePosition": False}
        return {}

    monkeypatch.setattr(binance_executor, "signed_request", fake)
    monkeypatch.setattr(binance_executor, "_market_filters", lambda *a, **k: dict(_FILTERS))

    for kwargs in ({}, {"tp2": 2800.0}):
        with pytest.raises(binance_executor.ExecutorError) as exc_info:
            binance_executor.place_spot_plan("ETHUSDT", 2700.0, 2750.0, 2680.0, 0.01,
                                             ("k", "s"), "spot-eth-1", **kwargs)
        assert exc_info.value.code == -2010, f"nhanh {kwargs} khong bao -2010"


def test_spot_market_entry_without_fill_is_aborted(monkeypatch):
    """Chưa khớp KHÔNG được hủy — futures cũng treo entry GTC chờ giá về vùng.

    Đây là nhánh CÓ tp2 (entry đơn, OCO gắn sau khi khớp). Truoc day spot huy sau 30s
    → plan co zone chua cham thi khong bao gio vao lenh (khac han futu).
    """
    calls = []
    monkeypatch.setattr(binance_executor, "signed_request",
                        _spot_fake(calls, entry_status="NEW"))
    monkeypatch.setattr(binance_executor, "_market_filters", lambda *a, **k: dict(_FILTERS))

    with pytest.raises(binance_executor.ExecutorError):
        binance_executor.place_spot_plan(
            "ETHUSDT", 2700.0, 2750.0, 2680.0, 0.01, ("k", "s"), "spot-eth-1",
            fill_timeout=0, poll_seconds=0, tp2=2800.0)
    deletes = [c for c in calls if c[0] == "DELETE"]
    assert deletes, "market entry khong khop phai best-effort cancel"
    assert not any(p.endswith("/orderList/otoco") for _, p, _ in calls)


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
    key_store.save_api_keys(31, "APIKEY" + "x" * 30, "SECRET" + "y" * 30)
    # 1 key chung futures+spot → cả 2 market cùng thấy key của user 31.
    assert key_store.api_key_status(31) == "ok"
    assert key_store.api_key_status(999) == "missing"

    # Đổi khóa mã hóa → dòng vẫn nằm trong DB nhưng không đọc được.
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    assert key_store.has_api_keys(31) is True
    assert key_store.api_key_status(31) == "broken"


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


# ─── Endpoint OCO spot + claim slot nguyên tử ────────────────────────────────

_FILTERS = {"tick": 0.01, "step": 0.001, "min_notional": 5,
            "min_qty": 0.00001, "base_asset": "BTC"}


def _spot_fake(calls, *, oco_result=None, oco_error=None, legacy_result=None,
               legacy_error=None, btc_free="1.00000000", otoco_error=None,
               entry_status="FILLED", pending_dead=False):
    """Fake signed_request cho luồng spot (entry + OCO)."""

    def fake(base, api_key, secret, method, path, params):
        calls.append((method, path, dict(params)))
        if path.endswith("/api/v3/account"):
            # _free_base_qty / spot_fee_rate doc day — phai tra ve du lieu.
            return {"balances": [{"asset": "BTC", "free": btc_free, "locked": "0"}],
                    "commissionRates": {"maker": "0.001", "taker": "0.001"}}
        if path.endswith("/orderList/otoco"):
            if otoco_error is not None:
                raise otoco_error
            return {"orderListId": 4242,
                    "orderReports": [
                        {"type": "LIMIT", "side": "BUY", "orderId": 7,
                         "clientOrderId": params.get("workingClientOrderId")},
                        {"type": "LIMIT_MAKER", "side": "SELL", "orderId": 8,
                         "clientOrderId": params.get("pendingAboveClientOrderId")},
                        {"type": "STOP_LOSS", "side": "SELL", "orderId": 9,
                         "clientOrderId": params.get("pendingBelowClientOrderId")},
                    ]}
        if path.endswith("/api/v3/orderList/oco"):
            if oco_error is not None:
                raise oco_error
            if oco_result is not None:
                return oco_result
            # Mac dinh nhu phan hoi that cua exchange (orderListId + 2 leg).
            return {"orderListId": 4242,
                    "orders": [
                        {"clientOrderId": params.get("aboveClientOrderId"), "orderId": 9001},
                        {"clientOrderId": params.get("belowClientOrderId"), "orderId": 9002},
                    ]}
        if path.endswith("/api/v3/order/oco"):
            if legacy_error is not None:
                raise legacy_error
            return legacy_result or {}
        if path.endswith("/api/v3/order") and method == "POST":
            if params.get("type") == "MARKET":
                return {"orderId": 7, "clientOrderId": "spot-eth-1-e",
                        "status": entry_status,
                        "executedQty": "0.01" if entry_status == "FILLED" else "0"}
            return {"orderId": 7, "clientOrderId": "spot-eth-1-e"}
        if path.endswith("/api/v3/order") and method == "GET":
            if pending_dead and str(params.get("origClientOrderId") or "").endswith("-tp"):
                raise binance_executor.ExecutorError(-2011, "Order does not exist.")
            return {"status": entry_status,
                    "executedQty": "0.01" if entry_status == "FILLED" else "0"}
        return {}

    return fake


def _futu_fake(calls):
    """Fake signed_request cho place_futures_plan (khong loi)."""

    def fake(base, api_key, secret, method, path, params):
        calls.append((method, path, dict(params)))
        if path.endswith("positionSide/dual"):
            return {"dualSidePosition": False}
        if path.endswith("/fapi/v1/leverage"):
            return {}
        if path.endswith("/fapi/v1/algoOrder"):
            return {"algoId": 1000 + len(calls)}
        if path.endswith("/fapi/v1/order") and method == "POST":
            return {"orderId": 555}
        if path.endswith("/fapi/v1/order") and method == "GET":
            return {"status": "NEW", "executedQty": "0"}
        return {}

    return fake


def test_spot_market_entry_then_oco(monkeypatch):
    """KHÔNG có tp2 → OTOCO: entry + TP + SL gửi trong CÙNG 1 request (task 2).

    Truoc day phai dat entry roi task nen gan OCO; OTOCO de exchange tu gan khi khop
    → song song qua restart, khong con van de position tran khi bot chet giua chung.
    """
    calls = []
    monkeypatch.setattr(binance_executor, "signed_request", _spot_fake(calls))
    monkeypatch.setattr(binance_executor, "_market_filters", lambda *a, **k: dict(_FILTERS))

    result = binance_executor.place_spot_plan(
        "ETHUSDT", 2700.0, 2750.0, 2680.0, 0.01, ("k", "s"), "spot-eth-1",
        fill_timeout=0, poll_seconds=0)

    market = [prm for m, p, prm in calls
              if m == "POST" and p.endswith("/api/v3/order")]
    oco = [prm for m, p, prm in calls if p.endswith("/orderList/oco")]
    assert len(market) == 1 and market[0]["type"] == "MARKET"
    assert len(oco) == 1
    assert oco[0]["abovePrice"] == "2750"
    assert oco[0]["belowStopPrice"] == "2680"
    assert result["market_entry"] is True
    assert result["protected"] is True


def test_spot_market_oco_uses_executed_qty(monkeypatch):
    """PHI spot tinh vao coin: pendingQuantity = qty DAY bi exchange bo qua IM LANG.

    Da gap that tren demo: mua 0.01 khi chi con bui 9.2e-6 → khop xong `lenh treo: 0`,
    position tran ma khong co loi nao. Phai kep bang free_before + qty*(1-phi).
    """
    calls = []
    monkeypatch.setattr(
        binance_executor, "signed_request",
        _spot_fake(calls, btc_free="0.00900000"),
    )
    monkeypatch.setattr(binance_executor, "_market_filters", lambda *a, **k: dict(_FILTERS))

    result = binance_executor.place_spot_plan(
        "ETHUSDT", 2700.0, 2750.0, 2680.0, 0.01, ("k", "s"), "spot-eth-1",
        fill_timeout=0, poll_seconds=0)

    prm = next(prm for m, p, prm in calls if p.endswith("/orderList/oco"))
    assert prm["quantity"] == "0.009"
    assert result["protected"] is True


def test_spot_market_entry_is_protected_by_oco(monkeypatch):
    """Da khop ma exchange bo qua pending → PHAI bao protected=False de safety net chay."""
    calls = []
    monkeypatch.setattr(
        binance_executor, "signed_request",
        _spot_fake(calls, entry_status="FILLED"),
    )
    monkeypatch.setattr(binance_executor, "_market_filters", lambda *a, **k: dict(_FILTERS))

    result = binance_executor.place_spot_plan(
        "ETHUSDT", 2700.0, 2750.0, 2680.0, 0.01, ("k", "s"), "spot-eth-1",
        fill_timeout=1, poll_seconds=0)

    assert result["filled"] is True
    assert result["protected"] is True
    assert any(p.endswith("/orderList/oco") for _, p, _ in calls)


def test_post_oco_uses_new_endpoint(monkeypatch):
    """POST /api/v3/order/oco bi deprecated 2024-04-02; con dung la TP/SL spot treo chet."""
    calls = []
    monkeypatch.setattr(
        binance_executor, "signed_request",
        _spot_fake(calls, oco_result={
            "orderListId": 555,
            "orders": [
                {"clientOrderId": "p-tp", "orderId": 9001},
                {"clientOrderId": "p-sl", "orderId": 9002},
            ],
        }),
    )

    out = binance_executor._post_oco(
        "https://x", "k", "s", "ETHUSDT",
        tp_price=2750.0, sl_price=2680.0, tp_coid="p-tp", sl_coid="p-sl",
        qty_sell=0.01, flt=dict(_FILTERS))

    oco = [(m, p, prm) for m, p, prm in calls if p.endswith("/api/v3/orderList/oco")]
    assert len(oco) == 1, "phai goi dung 1 lan OCO"
    prm = oco[0][2]
    assert prm["aboveType"] == "LIMIT_MAKER"
    assert prm["belowType"] == "STOP_LOSS"
    assert prm["abovePrice"] == "2750" and prm["belowStopPrice"] == "2680"
    # Khong gui tham so khong ton tai (truoc do gui timeInForce + newClientOrderId).
    assert "timeInForce" not in prm and "newClientOrderId" not in prm
    assert out["oco_list_id"] == 555
    assert out["tp_leg_order_id"] == 9001
    assert out["sl_leg_order_id"] == 9002


def test_post_oco_falls_back_to_legacy_only_on_404(monkeypatch):
    """Host cu chua co endpoint moi → thu lai endpoint deprecated 1 lan."""
    calls = []
    monkeypatch.setattr(
        binance_executor, "signed_request",
        _spot_fake(calls,
                   oco_error=binance_executor.ExecutorError(404, "Not Found"),
                   legacy_result={"orderListId": 777, "orders": []}),
    )

    out = binance_executor._post_oco(
        "https://x", "k", "s", "ETHUSDT",
        tp_price=2750.0, sl_price=2680.0, tp_coid="p-tp", sl_coid="p-sl",
        qty_sell=0.01, flt=dict(_FILTERS))

    oco_paths = [p for _, p, _ in calls if p.endswith("/oco")]
    assert oco_paths == ["/api/v3/orderList/oco", "/api/v3/order/oco"]
    legacy = next(prm for _, p, prm in calls if p == "/api/v3/order/oco")
    assert legacy["limitClientOrderId"] == "p-tp"
    assert legacy["stopClientOrderId"] == "p-sl"
    assert "timeInForce" not in legacy
    assert out["oco_list_id"] == 777


def test_post_oco_no_fallback_on_business_error(monkeypatch):
    """Loi nghiep vu (-2011...) KHONG duoc fallback: luc do co the sinh OCO trung."""
    calls = []
    monkeypatch.setattr(
        binance_executor, "signed_request",
        _spot_fake(calls, oco_error=binance_executor.ExecutorError(-2011, "Order does not exist.")),
    )

    with pytest.raises(binance_executor.ExecutorError):
        binance_executor._post_oco(
            "https://x", "k", "s", "ETHUSDT",
            tp_price=2750.0, sl_price=2680.0, tp_coid="p-tp", sl_coid="p-sl",
            qty_sell=0.01, flt=dict(_FILTERS))

    oco_paths = [p for _, p, _ in calls if p.endswith("/oco")]
    assert oco_paths == ["/api/v3/orderList/oco"], "khong duoc thu endpoint cu khi loi nghiep vu"


def test_attach_spot_oco_sells_when_oco_fails(monkeypatch):
    """Gan OCO that bai → PHAI ban ngay thi truong, khong de position tran."""
    calls = []
    monkeypatch.setattr(
        binance_executor, "signed_request",
        _spot_fake(calls, oco_error=binance_executor.ExecutorError(-2011, "Order does not exist.")),
    )
    monkeypatch.setattr(binance_executor, "_market_filters", lambda *a, **k: dict(_FILTERS))

    with pytest.raises(binance_executor.ExecutorError) as exc_info:
        binance_executor.attach_spot_oco("https://x", ("k", "s"), "ETHUSDT",
                                         2750.0, 2680.0, "spot-eth-1", 0.01)
    assert "ĐÃ BÁN NGAY" in exc_info.value.msg

    market_sells = [prm for m, p, prm in calls
                    if m == "POST" and p.endswith("/api/v3/order") and prm.get("type") == "MARKET"]
    assert len(market_sells) == 1, "phai ban ngay 1 lenh market"
    # Khong duoc gui lenh endpoint cu.
    assert not any(p == "/api/v3/order/oco" for _, p, _ in calls)


def test_attach_spot_oco_two_legs_when_tp2(monkeypatch):
    """Co tp2 → 2 OCO rieng: 70% o tp1, 30% o tp2, ca 2 deu kem SL (task 1)."""
    calls = []
    monkeypatch.setattr(binance_executor, "signed_request", _spot_fake(calls))
    monkeypatch.setattr(binance_executor, "_market_filters", lambda *a, **k: dict(_FILTERS))

    out = binance_executor.attach_spot_oco("https://x", ("k", "s"), "ETHUSDT",
                                           2750.0, 2680.0, "spot-eth-1", 0.01, tp2=2800.0)

    oco = [prm for m, p, prm in calls if p.endswith("/api/v3/orderList/oco")]
    assert len(oco) == 2, "phai co 2 OCO khi co tp2"
    assert oco[0]["quantity"] == "0.007" and oco[0]["abovePrice"] == "2750"
    assert oco[1]["quantity"] == "0.003" and oco[1]["abovePrice"] == "2800"
    # Ca 2 deu co SL cung gia; coid khac nhau de khong chan nhau.
    assert oco[0]["belowStopPrice"] == "2680" and oco[1]["belowStopPrice"] == "2680"
    assert oco[0]["aboveClientOrderId"] == "spot-eth-1-tp"
    assert oco[1]["aboveClientOrderId"] == "spot-eth-1-tp2"
    assert oco[0]["belowClientOrderId"] == "spot-eth-1-sl"
    assert oco[1]["belowClientOrderId"] == "spot-eth-1-sl2"
    # Tong 2 OCO = toan bo so ban duoc (khong ro ra).
    assert abs(float(oco[0]["quantity"]) + float(oco[1]["quantity"]) - 0.01) < 1e-12
    assert out["tp2_oco_list_id"] == 4242
    assert out["qty_tp1"] == 0.007 and out["qty_tp2"] == 0.003


def test_split_tp_qty_respects_step_and_sum(monkeypatch):
    """Tach 70/30 phai la boi stepSize (khong thi exchange tra -1111) va tong dung qty."""
    # step 0.001: 0.023 -> 0.016 / 0.007 (khong phai 0.0161/0.0069)
    assert binance_executor.split_tp_qty(0.023, 0.001) == (0.016, 0.007)
    # qty khong phai boi step -> tu floor
    assert binance_executor.split_tp_qty(0.0075, 0.001) == (0.004, 0.003)
    # qua nho de chia -> don het vao TP1
    assert binance_executor.split_tp_qty(0.001, 0.001) == (0.001, 0.0)
    # tong luon bang qty da floor
    for qty, step in ((1.0, 0.01), (5.0, 0.0001), (0.02, 0.001), (0.00002, 1e-05)):
        q1, q2 = binance_executor.split_tp_qty(qty, step)
        assert abs(q1 + q2 - binance_executor._floor_to(qty, step)) < 1e-12
        assert binance_executor._floor_to(q1, step) == q1
        assert binance_executor._floor_to(q2, step) == q2


def test_futures_tp_split_70_30_when_tp2_present(monkeypatch):
    """Plan co tp2 → TP1 = 70%, TP2 = 30%, SL = 100% (task 1, futures)."""
    calls = []
    monkeypatch.setattr(binance_executor, "signed_request", _futu_fake(calls))
    monkeypatch.setattr(binance_executor, "_market_filters",
                        lambda *a, **k: {"tick": 0.01, "step": 0.001,
                                         "min_notional": 5, "min_qty": 0.001,
                                         "base_asset": "ETH"})

    result = binance_executor.place_futures_plan(
        "ETHUSDT", "LONG", 2700.0, 2750.0, 2600.0, 0.01, 20, ("k", "s"),
        "futu-eth-1", tp2=2800.0)

    algos = [prm for m, p, prm in calls if p.endswith("/fapi/v1/algoOrder")]
    tp_algos = [x for x in algos if x["type"] == "TAKE_PROFIT_MARKET"]
    sl_algos = [x for x in algos if x["type"] == "STOP_MARKET"]
    assert len(tp_algos) == 2, "phai co 2 lenh TP khi co tp2"
    assert len(sl_algos) == 1, "SL van la 1 lenh"
    assert float(tp_algos[0]["quantity"]) == 0.007
    assert float(tp_algos[1]["quantity"]) == 0.003
    assert abs(float(tp_algos[0]["quantity"]) + float(tp_algos[1]["quantity"]) - 0.01) < 1e-12
    # SL giu 100%: TP1 khop truoc thi con 30% vi the, algo STOP van dong duoc (da test demo).
    assert float(sl_algos[0]["quantity"]) == 0.01
    assert tp_algos[0]["triggerPrice"] == "2750"
    assert tp_algos[1]["triggerPrice"] == "2800"
    assert sl_algos[0]["triggerPrice"] == "2600"
    assert tp_algos[0]["clientAlgoId"] == "futu-eth-1-tp"
    assert tp_algos[1]["clientAlgoId"] == "futu-eth-1-tp2"
    assert result["tp2_algo_id"] is not None
    assert result["qty_tp1"] == 0.007 and result["qty_tp2"] == 0.003


def test_futures_tp1_full_qty_when_no_tp2(monkeypatch):
    """KHONG co tp2 → TP1 phai la TOAN BO qty (khong duoc chia 70% roi bo 30% lai troi)."""
    calls = []
    monkeypatch.setattr(binance_executor, "signed_request", _futu_fake(calls))
    monkeypatch.setattr(binance_executor, "_market_filters",
                        lambda *a, **k: {"tick": 0.01, "step": 0.001,
                                         "min_notional": 5, "min_qty": 0.001,
                                         "base_asset": "ETH"})

    result = binance_executor.place_futures_plan(
        "ETHUSDT", "LONG", 2700.0, 2750.0, 2600.0, 0.01, 20, ("k", "s"), "futu-eth-1")

    algos = [prm for m, p, prm in calls if p.endswith("/fapi/v1/algoOrder")]
    tp_algos = [x for x in algos if x["type"] == "TAKE_PROFIT_MARKET"]
    sl_algos = [x for x in algos if x["type"] == "STOP_MARKET"]
    assert len(tp_algos) == 1, "khong co tp2 thi chi 1 lenh TP"
    assert float(tp_algos[0]["quantity"]) == 0.01, "TP1 phai la toan bo qty"
    assert float(sl_algos[0]["quantity"]) == 0.01
    assert result["tp2_algo_id"] is None
    assert result["qty_tp2"] == 0.0


def test_claim_auto_scan_slot_is_atomic(monkeypatch):
    """2 process chung bot.db từng qua cửa check-then-set → cùng quét, cùng đặt lệnh trùng."""
    a = _analyze(monkeypatch)
    a.init_auto_scan_db()
    conn = sqlite3.connect(a.DB_PATH)
    conn.execute("DELETE FROM auto_scan_state")
    conn.commit()
    conn.close()

    assert a.claim_auto_scan_slot("2026-10-06T00:00:00+00:00") is True
    # Bên thứ 2 claim cùng slot → phải bị từ chối.
    assert a.claim_auto_scan_slot("2026-10-06T00:00:00+00:00") is False
    # Slot khác thì claim được.
    assert a.claim_auto_scan_slot("2026-10-06T01:00:00+00:00") is True

    # Claim phải ghi luôn last_scan_at để /autoscanstatus còn hiện "lần quét gần nhất".
    assert a._auto_scan_state_get("last_scan_at") is not None


def _seed_spot_signal(a, plan_id, tp_algo_id=None):
    conn = sqlite3.connect(a.DB_PATH)
    conn.execute("DELETE FROM auto_scan_signals")
    conn.execute(
        "INSERT INTO auto_scan_signals (user_id, chat_id, symbol, mode, direction, sent_at,"
        " prediction_id, plan_id, order_status, sl, tp1, qty, tp_algo_id)"
        " VALUES (7,1,'ETHUSDT','spot','BUY','2026-10-06T00:00:00+00:00', 9501, ?, 'placed',"
        " 2680.0, 2750.0, 0.01, ?)",
        (plan_id, tp_algo_id),
    )
    conn.commit()
    conn.close()


def test_ensure_spot_protection_protects_filled_entry(monkeypatch):
    """Entry spot da khop ma TREN SAN chua co OCO → phai gan OCO (khong de position tran)."""
    a = _analyze(monkeypatch)
    a.init_auto_scan_db()
    _seed_spot_signal(a, "spot-eth-1")

    import binance_executor
    monkeypatch.setattr(key_store, "get_api_keys", lambda uid: ("k", "s"))
    monkeypatch.setattr(binance_executor, "spot_order_state",
                        lambda *a_, **k: {"status": "FILLED", "executed": 0.01})
    # Hoi TREN SAN: khong co TP/SL → phai gan
    monkeypatch.setattr(binance_executor, "spot_pending_state",
                        lambda *a_, **k: {"alive": False, "list_id": None})
    monkeypatch.setattr(binance_executor, "attach_spot_oco",
                        lambda *a_, **k: {"oco_list_id": 777, "tp_leg_order_id": 1, "sl_leg_order_id": 2})

    payload = a.ensure_spot_protection()

    assert payload["candidates"] == 1 and payload["protected"] == 1, payload
    rows = a.list_session_signals(7, "spot")
    assert rows[0]["tp_algo_id"] == "777", "phai ghi oco_list_id vao ledger"


def test_ensure_spot_protection_waits_when_not_filled(monkeypatch):
    """Lenh chua khop → KHONG gan OCO, giu nguyen cho lan job sau."""
    a = _analyze(monkeypatch)
    a.init_auto_scan_db()
    _seed_spot_signal(a, "spot-eth-2")

    import binance_executor
    monkeypatch.setattr(key_store, "get_api_keys", lambda uid: ("k", "s"))
    monkeypatch.setattr(binance_executor, "spot_order_state",
                        lambda *a_, **k: {"status": "NEW", "executed": 0})

    def boom(*a_, **k):
        raise AssertionError("khong duoc goi attach_spot_oco khi chua khop")

    monkeypatch.setattr(binance_executor, "attach_spot_oco", boom)

    payload = a.ensure_spot_protection()
    assert payload["protected"] == 0 and payload["waiting"] == 1, payload
    rows = a.list_session_signals(7, "spot")
    assert rows[0]["tp_algo_id"] in (None, ""), "van chua co OCO"


def test_ensure_spot_protection_trusts_exchange_not_db(monkeypatch):
    """DB ghi tp_algo_id nhung TREN SAN khong co → van phai gan lai (khong tin DB).

    Day la ca quyet dinh: OCO bi exchange bo qua khi khop (do phi) hoac watcher chet khi
    restart → DB noi "da bao ve" trong khi san khong co gi = position tran.
    """
    a = _analyze(monkeypatch)
    a.init_auto_scan_db()
    _seed_spot_signal(a, "spot-eth-4", tp_algo_id="555")

    import binance_executor
    monkeypatch.setattr(key_store, "get_api_keys", lambda uid: ("k", "s"))
    monkeypatch.setattr(binance_executor, "spot_order_state",
                        lambda *a_, **k: {"status": "FILLED", "executed": 0.01})
    monkeypatch.setattr(binance_executor, "spot_pending_state",
                        lambda *a_, **k: {"alive": False, "list_id": None})
    monkeypatch.setattr(binance_executor, "attach_spot_oco",
                        lambda *a_, **k: {"oco_list_id": 888, "tp_leg_order_id": 1})

    payload = a.ensure_spot_protection()
    assert payload["candidates"] == 1 and payload["protected"] == 1, payload
    rows = a.list_session_signals(7, "spot")
    assert rows[0]["tp_algo_id"] == "888", "phai ghi lai list id moi"


def test_ensure_spot_protection_skips_when_exchange_has_protection(monkeypatch):
    """Tren san DA co TP/SL → khong gan them (khong tao OCO trung)."""
    a = _analyze(monkeypatch)
    a.init_auto_scan_db()
    _seed_spot_signal(a, "spot-eth-3", tp_algo_id="555")

    import binance_executor
    monkeypatch.setattr(key_store, "get_api_keys", lambda uid: ("k", "s"))
    monkeypatch.setattr(binance_executor, "spot_order_state",
                        lambda *a_, **k: {"status": "FILLED", "executed": 0.01})
    monkeypatch.setattr(binance_executor, "spot_pending_state",
                        lambda *a_, **k: {"alive": True, "list_id": 555})

    def boom(*a_, **k):
        raise AssertionError("khong duoc gan OCO khi tren san da co TP/SL")

    monkeypatch.setattr(binance_executor, "attach_spot_oco", boom)

    payload = a.ensure_spot_protection()
    assert payload["candidates"] == 1 and payload["protected"] == 1, payload


def test_claim_slot_second_caller_gets_skipped_payload(monkeypatch):
    import asyncio

    a = _analyze(monkeypatch)
    a.init_auto_scan_db()
    conn = sqlite3.connect(a.DB_PATH)
    conn.execute("DELETE FROM auto_scan_state")
    conn.commit()
    conn.close()

    first = asyncio.run(a._run_auto_scan_cycle(bot=None, force=True))
    assert first.get("skipped") is False, "lần chạy đầu phải đi tiếp (kể cả khi không có user)"

    second = asyncio.run(a._run_auto_scan_cycle(bot=None, force=True))
    assert second.get("skipped") is True
    assert "claim" in str(second.get("reason") or "")


# ─── Đợt 3d: 5 bug cuoi + SL mồ côi khi TP khớp ─────────────────────────────

def test_startup_prune_uses_hidden_limit_for_hidden_group(tmp_path, monkeypatch):
    """Nhom an (NO_TRADE/REJECTED) phai giu theo HIDDEN limit (5), khong phai VISIBLE (10)."""
    db = str(tmp_path / "prune.db")
    monkeypatch.setattr(analyze, "DB_PATH", db)
    monkeypatch.setattr(analyze, "_prediction_db_initialized", False)
    analyze.init_prediction_db()  # tao bang

    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM predictions")
    for i in range(8):  # nhom an → phai bi cat xuong 5
        conn.execute(
            "INSERT INTO predictions (user_id, chat_id, symbol, mode, created_at, direction, result)"
            " VALUES (5,5,'ETHUSDT','futures',?,'LONG','NO_TRADE')",
            (f"2026-01-{i + 1:02d}T00:00:00+00:00",))
    for i in range(12):  # nhom hien thi → giu 10
        conn.execute(
            "INSERT INTO predictions (user_id, chat_id, symbol, mode, created_at, direction, result)"
            " VALUES (5,5,'BTCUSDT','futures',?,'LONG','WIN')",
            (f"2026-02-{i + 1:02d}T00:00:00+00:00",))
    conn.commit()
    conn.close()

    monkeypatch.setattr(analyze, "_prediction_db_initialized", False)
    analyze.init_prediction_db()  # chay lai startup prune

    conn = sqlite3.connect(db)
    hidden = conn.execute("SELECT COUNT(*) FROM predictions WHERE result='NO_TRADE'").fetchone()[0]
    visible = conn.execute("SELECT COUNT(*) FROM predictions WHERE result NOT IN "
                           "('NO_TRADE','REJECTED_PLAN')").fetchone()[0]
    conn.close()
    assert analyze.HIDDEN_LEARNING_RETENTION_LIMIT == 5
    assert hidden == analyze.HIDDEN_LEARNING_RETENTION_LIMIT, \
        f"nhom an phai giu {analyze.HIDDEN_LEARNING_RETENTION_LIMIT}, got {hidden}"
    assert visible == analyze.VISIBLE_PREDICTION_RETENTION_LIMIT


def test_wipe_runs_when_bot_missed_the_night_window(monkeypatch):
    """Bot down qua dem → phai wipe vao lan dau chay len, khong bi bo qua mai mai."""
    from datetime import datetime, timedelta, timezone

    a = _analyze(monkeypatch)
    a.init_auto_scan_db()
    monkeypatch.setattr(a, "AUTOSCAN_SLEEP_HOUR_VN", 0)
    monkeypatch.setattr(a, "AUTOSCAN_WAKE_HOUR_VN", 7)

    conn = sqlite3.connect(a.DB_PATH)
    conn.execute("DELETE FROM auto_scan_state")
    conn.execute("DELETE FROM auto_scan_signals")
    for pid, sym in ((9601, "ETHUSDT"), (9602, "BTCUSDT")):
        conn.execute(
            "INSERT INTO auto_scan_signals (user_id, chat_id, symbol, mode, direction, sent_at,"
            " prediction_id, plan_id, order_status) VALUES (7,1,?, 'futures','LONG',"
            " '2026-10-05T01:00:00+00:00', ?, 'futu-x-1', 'pending')",
            (sym, pid))
    conn.commit()
    conn.close()

    # Lan wipe cu nay = hom qua → da bo qua ca dem, moi bot day len luc 10:00 (qua gio wake)
    a._auto_scan_state_set("signals_wiped_day", "2026-10-05")
    tz7 = timezone(timedelta(hours=7))
    result = a.maintain_auto_scan_daily_window(datetime(2026, 10, 6, 10, 0, tzinfo=tz7))
    assert result["in_sleep_window"] is False, "10:00 khong con trong cua so ngu"
    assert result["signals_wiped"] == 2, f"phai wipe ngay khi bot day len, got {result}"

    # Van chi wipe 1 lan/ngay
    result2 = a.maintain_auto_scan_daily_window(datetime(2026, 10, 6, 11, 0, tzinfo=tz7))
    assert result2["signals_wiped"] == 0, "khong duoc wipe lan 2 trong cung ngay"


def test_new_pending_warns_when_switching_market():
    """Bat dau /onfutu do dang roi gop /onspot → PHAI bao, khong de nhap so lieu vao nham phien."""
    symbol_control._AUTO_PENDING.clear()
    try:
        note1 = symbol_control._new_pending(1, stage="qty", market="futures", symbol="ETHUSDT")
        assert note1 == "", "lan dau tien khong can canh bao"

        note2 = symbol_control._new_pending(1, stage="qty", market="spot", symbol="ETHUSDT")
        assert note2, "phai canh bao khi doi market"
        assert "FUTURES" in note2 and "SPOT" in note2

        # Cung luong thi tiep tinh binh thuong
        note3 = symbol_control._new_pending(1, stage="qty", market="spot", symbol="ETHUSDT")
        assert note3 == ""
    finally:
        symbol_control._AUTO_PENDING.clear()


def _can_release(monkeypatch, *, entry_status, position):
    import binance_executor as ex
    monkeypatch.setattr(ex, "futures_entry_state",
                        lambda *a, **k: {"status": entry_status, "executed": 0.0})
    monkeypatch.setattr(ex, "has_open_futures_position", lambda *a, **k: position)
    return analyze._can_release_futures_protection(("k", "s"), "ETHU", 111)


def test_can_release_futures_protection_gates(monkeypatch):
    """Chi goc khi entry khong con mo VA khong con vi the — thieu 1 trong 2 la KHONG duoc goc."""
    assert _can_release(monkeypatch, entry_status="NEW", position=False)[0] is False
    assert _can_release(monkeypatch, entry_status="PARTIALLY_FILLED", position=False)[0] is False
    assert _can_release(monkeypatch, entry_status="FILLED", position=True)[0] is False
    assert _can_release(monkeypatch, entry_status="FILLED", position=False)[0] is True
    # Entry khong con (da bi huy/ khong tim thay) nhung van khong con vi the → goc duoc
    assert _can_release(monkeypatch, entry_status="GONE", position=False)[0] is True

    # Khong doc duoc vi the → KHONG phan doan, giu nguyen
    import binance_executor as ex
    monkeypatch.setattr(ex, "futures_entry_state",
                        lambda *a, **k: {"status": "FILLED", "executed": 0.0})
    monkeypatch.setattr(ex, "has_open_futures_position", lambda *a, **k: None)
    assert analyze._can_release_futures_protection(("k", "s"), "ETHU", 111)[0] is False


def _seed_futures_signal(a, plan_id, *, order_status="placed"):
    conn = sqlite3.connect(a.DB_PATH)
    try:
        conn.execute("DELETE FROM auto_scan_signals")
        conn.execute(
            "INSERT INTO auto_scan_signals (user_id, chat_id, symbol, mode, direction, sent_at,"
            " prediction_id, plan_id, order_status, entry_order_id, tp_algo_id, tp2_algo_id,"
            " sl_algo_id)"
            " VALUES (7,1,'ETHU','futures','LONG','2026-10-06T00:00:00+00:00', 9701, ?, ?,"
            " 111, 112, 113, 114)",
            (plan_id, order_status),
        )
        conn.commit()
    finally:
        conn.close()


def test_ensure_no_futures_orphan_protection_releases(monkeypatch):
    """TP da khop, vi the da dong ma SL con treo → PHAI goc (Binance khong tu huy)."""
    import binance_executor as ex

    a = _analyze(monkeypatch)
    a.init_auto_scan_db()
    _seed_futures_signal(a, "futu-eth-77-9")

    monkeypatch.setattr(key_store, "get_api_keys", lambda uid: ("k", "s"))
    monkeypatch.setattr(ex, "futures_entry_state",
                        lambda *x, **k: {"status": "FILLED", "executed": 0.0})
    monkeypatch.setattr(ex, "open_futures_algo_orders", lambda *x, **k: [])
    monkeypatch.setattr(ex, "has_open_futures_position", lambda *x, **k: False)
    calls = []
    monkeypatch.setattr(
        ex, "cancel_futures_algo_orders",
        lambda keys, symbol, ids: (calls.append([i for i in ids if i]),
                                   [{"kind": "algo", "symbol": symbol, "algo_id": i}
                                    for i in ids if i])[1],
    )

    payload = a.ensure_no_futures_orphan_protection()

    assert payload["candidates"] == 1 and payload["released"] == 1, payload
    # entry=111, tp=112, tp2=113, sl=114 → goc 3 lenh TP/SL
    assert calls and calls[0] == ["112", "113", "114"], f"phai goc ca TP1+TP2+SL, got {calls}"
    rows = a.list_session_signals(7, "futures")
    assert rows[0]["order_status"] == "closed", "phai doi trang thai de khong goc lai"


def test_ensure_no_futures_orphan_protection_keeps_live_position(monkeypatch):
    """Van con vi the → KHONG duoc goc TP/SL (dang bao ve vi the)."""
    import binance_executor as ex

    a = _analyze(monkeypatch)
    a.init_auto_scan_db()
    _seed_futures_signal(a, "futu-eth-77-8")

    monkeypatch.setattr(key_store, "get_api_keys", lambda uid: ("k", "s"))
    monkeypatch.setattr(ex, "futures_entry_state",
                        lambda *x, **k: {"status": "FILLED", "executed": 0.0})
    monkeypatch.setattr(ex, "open_futures_algo_orders", lambda *x, **k: [])
    monkeypatch.setattr(ex, "has_open_futures_position", lambda *x, **k: True)

    def boom(*a, **k):
        raise AssertionError("khong duoc goc TP/SL khi van con vi the")

    monkeypatch.setattr(ex, "cancel_futures_algo_orders", boom)

    payload = a.ensure_no_futures_orphan_protection()
    assert payload["released"] == 0 and payload["waiting"] == 1, payload
    rows = a.list_session_signals(7, "futures")
    assert rows[0]["order_status"] == "placed"


def _insert_futures_protection_row(a, *, plan_id, symbol="ETHUSDT", exchange_symbol="ETHU",
                                   tp=112, tp2=None, sl=114):
    conn = sqlite3.connect(a.DB_PATH)
    try:
        conn.execute(
            "INSERT INTO auto_scan_signals "
            "(user_id, chat_id, symbol, exchange_symbol, mode, direction, sent_at, "
            "prediction_id, plan_id, order_status, entry_order_id, tp_algo_id, "
            "tp2_algo_id, sl_algo_id) "
            "VALUES (7,1,?,?, 'futures','LONG','2026-10-06T00:00:00+00:00', ?, ?, "
            "'placed', 111, ?, ?, ?)",
            (symbol, exchange_symbol, abs(hash(plan_id)) % 100000, plan_id, tp, tp2, sl),
        )
        conn.commit()
    finally:
        conn.close()


def _algo_event(symbol, algo_id, status="FINISHED", event_type="ALGO_UPDATE"):
    return {
        "e": event_type,
        "o": {"s": symbol, "aid": algo_id, "caid": f"futu-eth-77-{algo_id}", "X": status},
    }


def test_futures_stream_matches_demo_exchange_symbol_and_cleans_sl_siblings(monkeypatch):
    """Demo event dùng ETHU nhưng ledger giữ symbol phân tích ETHUSDT; SL phải gỡ TP1+TP2."""
    import binance_executor as ex

    a = _analyze(monkeypatch)
    a.init_auto_scan_db()
    with sqlite3.connect(a.DB_PATH) as conn:
        conn.execute("DELETE FROM auto_scan_signals")
    _insert_futures_protection_row(a, plan_id="futu-eth-77-112", tp=112, tp2=113, sl=114)
    monkeypatch.setattr(ex, "cancel_futures_algo_orders", lambda keys, symbol, ids: [
        {"kind": "algo", "symbol": symbol, "algo_id": i} for i in ids if i
    ])

    result = a.handle_futures_algo_update(7, ("k", "s"), _algo_event("ETHU", 114))

    assert result["handled"] is True
    assert [x["algo_id"] for x in result["canceled"]] == ["112", "113"]
    row = a.list_session_signals(7, "futures")[0]
    assert row["order_status"] == "closed"


def test_futures_tp1_keeps_sl_and_tp2_then_sl_cleans_tp2(monkeypatch):
    """TP1 partial không được gỡ SL/TP2; sau đó SL phải gỡ TP2 còn mồ côi."""
    import binance_executor as ex

    a = _analyze(monkeypatch)
    a.init_auto_scan_db()
    with sqlite3.connect(a.DB_PATH) as conn:
        conn.execute("DELETE FROM auto_scan_signals")
    _insert_futures_protection_row(a, plan_id="futu-eth-77-113", tp=112, tp2=113, sl=114)
    calls = []
    monkeypatch.setattr(ex, "cancel_futures_algo_orders", lambda keys, symbol, ids: (
        calls.append(list(ids)),
        [{"kind": "algo", "symbol": symbol, "algo_id": i} for i in ids if i],
    )[1])

    first = a.handle_futures_algo_update(7, ("k", "s"), _algo_event("ETHU", 112))
    assert first["deferred"] == "tp1_partial_exit"
    assert calls == []

    second = a.handle_futures_algo_update(7, ("k", "s"), _algo_event("ETHU", 114))
    assert [x["algo_id"] for x in second["canceled"]] == ["112", "113"]
    assert calls == [["112", "113"]]


def test_futures_tp2_event_cleans_tp1_and_sl_including_conditional_event(monkeypatch):
    """TP2 khớp là kết thúc plan; hỗ trợ cả event type conditional của Binance."""
    import binance_executor as ex

    a = _analyze(monkeypatch)
    a.init_auto_scan_db()
    with sqlite3.connect(a.DB_PATH) as conn:
        conn.execute("DELETE FROM auto_scan_signals")
    _insert_futures_protection_row(a, plan_id="futu-eth-77-115", tp=112, tp2=113, sl=114)
    calls = []
    monkeypatch.setattr(ex, "cancel_futures_algo_orders", lambda keys, symbol, ids: (
        calls.append(list(ids)),
        [{"kind": "algo", "symbol": symbol, "algo_id": i} for i in ids if i],
    )[1])

    result = a.handle_futures_algo_update(
        7, ("k", "s"), _algo_event("ETHU", 113, event_type="CONDITIONAL_ORDER_TRADE_UPDATE"))

    assert result["handled"] is True
    assert [x["algo_id"] for x in result["canceled"]] == ["114"]
    assert calls == [["114"]]


def test_futures_plan_without_tp2_tp1_event_cancels_sl(monkeypatch):
    """Plan chỉ có TP1: TP1 khớp thì SL phải được gỡ vì toàn bộ position đã chốt."""
    import binance_executor as ex

    a = _analyze(monkeypatch)
    a.init_auto_scan_db()
    with sqlite3.connect(a.DB_PATH) as conn:
        conn.execute("DELETE FROM auto_scan_signals")
    _insert_futures_protection_row(a, plan_id="futu-eth-77-116", tp=112, tp2=None, sl=114)
    calls = []
    monkeypatch.setattr(ex, "cancel_futures_algo_orders", lambda keys, symbol, ids: (
        calls.append(list(ids)),
        [{"kind": "algo", "symbol": symbol, "algo_id": i} for i in ids if i],
    )[1])

    result = a.handle_futures_algo_update(7, ("k", "s"), _algo_event("ETHU", 112))

    assert [x["algo_id"] for x in result["canceled"]] == ["114"]
    assert calls == [["114"]]


def test_reconcile_cleans_tp2_when_tp1_was_already_partial_and_sl_triggered(monkeypatch):
    """Nếu stream lỡ mất event SL, openAlgoOrders còn TP2 nhưng SL đã biến mất → phải gỡ TP2."""
    import binance_executor as ex

    a = _analyze(monkeypatch)
    a.init_auto_scan_db()
    with sqlite3.connect(a.DB_PATH) as conn:
        conn.execute("DELETE FROM auto_scan_signals")
    _insert_futures_protection_row(a, plan_id="futu-eth-77-114", tp=112, tp2=113, sl=114)
    monkeypatch.setattr(key_store, "get_api_keys", lambda uid: ("k", "s"))
    monkeypatch.setattr(ex, "futures_entry_state",
                        lambda *x, **k: {"status": "FILLED", "executed": 1.0})
    monkeypatch.setattr(ex, "open_futures_algo_orders",
                        lambda *x, **k: [{"algoId": 113}])
    monkeypatch.setattr(ex, "has_open_futures_position", lambda *x, **k: False)
    calls = []
    monkeypatch.setattr(ex, "cancel_futures_algo_orders", lambda keys, symbol, ids: (
        calls.append(list(ids)),
        [{"kind": "algo", "symbol": symbol, "algo_id": i} for i in ids if i],
    )[1])

    result = a.ensure_no_futures_orphan_protection()

    assert result["released"] == 1
    assert calls == [["113"]]


def test_reconcile_mixed_plans_only_cleans_their_own_tp_sl(monkeypatch):
    """Plan A/C không có TP2 không được lẫn với TP2 của plan B cùng symbol."""
    import binance_executor as ex

    a = _analyze(monkeypatch)
    a.init_auto_scan_db()
    _insert_futures_protection_row(a, plan_id="plan-a", tp=101, tp2=None, sl=102)
    _insert_futures_protection_row(a, plan_id="plan-b", tp=201, tp2=202, sl=203)
    _insert_futures_protection_row(a, plan_id="plan-c", tp=301, tp2=None, sl=302)
    monkeypatch.setattr(key_store, "get_api_keys", lambda uid: ("k", "s"))
    monkeypatch.setattr(ex, "futures_entry_state",
                        lambda *x, **k: {"status": "FILLED", "executed": 1.0})
    # A đã hết cả legs; B còn nguyên; C đã mất TP1 nhưng SL còn.
    monkeypatch.setattr(ex, "open_futures_algo_orders",
                        lambda *x, **k: [{"algoId": 201}, {"algoId": 202},
                                         {"algoId": 203}, {"algoId": 302}])
    monkeypatch.setattr(ex, "has_open_futures_position", lambda *x, **k: True)
    calls = []
    monkeypatch.setattr(ex, "cancel_futures_algo_orders", lambda keys, symbol, ids: (
        calls.append(list(ids)),
        [{"kind": "algo", "symbol": symbol, "algo_id": i} for i in ids if i],
    )[1])

    result = a.ensure_no_futures_orphan_protection()

    assert result["released"] == 2
    assert calls == [["302"]]
    rows = {r["plan_id"]: r for r in a.list_session_signals(7, "futures")}
    assert rows["plan-a"]["order_status"] == "closed"
    assert rows["plan-b"]["order_status"] == "placed"
    assert rows["plan-c"]["order_status"] == "closed"


def test_symbol_handler_ignores_commands(monkeypatch):
    """Slash commands must not be validated as Binance symbols."""
    import asyncio
    from types import SimpleNamespace

    class _Message:
        text = "/start"

        async def reply_text(self, *args, **kwargs):
            raise AssertionError("/start must not reach coin validation")

        async def delete(self):
            raise AssertionError("/start must not be deleted by symbol handler")

    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=123),
        effective_message=_Message(),
    )

    assert asyncio.run(symbol_control.handle_symbol(update, None)) is False
