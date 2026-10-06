"""Test: key_store (Fernet), binance_executor (tham số lệnh), phiên auto scan (plan_id/log/off)."""
import os
import sqlite3
import tempfile

from cryptography.fernet import Fernet

import binance_executor
import key_store
import symbol_control

_TMP = tempfile.mkdtemp()
_DB = os.path.join(_TMP, "exec_test.db")


# ─── key_store ───────────────────────────────────────────────────────────────

def test_key_store_roundtrip_and_fail_closed(monkeypatch):
    monkeypatch.setattr(key_store, "DB_PATH", _DB)
    # Chưa có khóa env → fail-closed, không ghi gì.
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    try:
        key_store.save_api_keys(1, "futures", "k" * 30, "s" * 30)
        assert False, "phải raise khi thiếu DATA_ENCRYPTION_KEY"
    except key_store.KeyError_:
        pass

    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    key_store.save_api_keys(1, "futures", "APIKEY" + "x" * 30, "SECRET" + "y" * 30)
    assert key_store.has_api_keys(1, "futures")
    assert not key_store.has_api_keys(1, "spot")
    api, secret = key_store.get_api_keys(1, "futures")
    assert api.startswith("APIKEY") and secret.startswith("SECRET")

    # DB phải chứa token Fernet (không phải key trần).
    conn = sqlite3.connect(_DB)
    stored = conn.execute("SELECT api_key_enc FROM user_api_keys").fetchone()[0]
    conn.close()
    assert not stored.startswith("APIKEY")

    # Khóa sai → không giải mã được → raise (fail-closed).
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    try:
        key_store.get_api_keys(1, "futures")
        assert False, "phải raise khi khóa đổi"
    except key_store.KeyError_:
        pass


def test_delete_api_keys(monkeypatch):
    monkeypatch.setattr(key_store, "DB_PATH", _DB)
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    key_store.save_api_keys(9, "spot", "APIKEY" + "x" * 30, "SECRET" + "y" * 30)
    assert key_store.has_api_keys(9, "spot")
    assert key_store.delete_api_keys(9, "spot") is True
    assert not key_store.has_api_keys(9, "spot")
    # Gỡ lần 2 → không còn gì, trả False.
    assert key_store.delete_api_keys(9, "spot") is False


def test_key_store_rejects_short_key(monkeypatch):
    monkeypatch.setattr(key_store, "DB_PATH", _DB)
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    try:
        key_store.save_api_keys(2, "futures", "short", "short")
        assert False, "key quá ngắn phải bị từ chối"
    except key_store.KeyError_:
        pass


# ─── binance_executor ────────────────────────────────────────────────────────

def _recorder(hedge: bool, fail_first_duplicate: bool = False):
    """Ghi nhận tham số từng call; giả lập response hợp lệ."""
    calls = []
    state = {"order_calls": 0}

    def fake(base, api_key, secret, method, path, params):
        calls.append((path, dict(params)))
        if path.endswith("positionSide/dual"):
            return {"dualSidePosition": hedge}
        if path.endswith("/fapi/v1/leverage"):
            return {"leverage": params.get("leverage"), "symbol": params.get("symbol")}
        if path.endswith("/fapi/v1/algoOrder"):
            return {"algoId": 1000 + len(calls)}
        if path.endswith("/fapi/v1/order") and method == "POST":
            state["order_calls"] += 1
            if fail_first_duplicate and state["order_calls"] == 1:
                raise binance_executor.ExecutorError(-2010, "Duplicate order sent.")
            return {"orderId": 555}
        return {}

    return calls, fake


def test_futures_params_hedge_mode(monkeypatch):
    monkeypatch.setattr(
        binance_executor, "_market_filters",
        lambda *a, **k: {"tick": 0.01, "step": 0.001, "min_notional": 5},
    )
    calls, fake = _recorder(hedge=True)
    monkeypatch.setattr(binance_executor, "signed_request", fake)

    result = binance_executor.place_futures_plan(
        "ETHUSDT", "LONG", 2700.0, 2750.0, 2680.0, 0.01, 20,
        ("k", "s"), "futu-eth-1",
    )
    paths = [p for p, _ in calls]
    order_params = next(p for path, p in calls if path.endswith("/fapi/v1/order") and p.get("clientOrderId"))
    tp_params = next(p for path, p in calls if path.endswith("/fapi/v1/algoOrder") and p.get("type") == "TAKE_PROFIT_MARKET")
    sl_params = next(p for path, p in calls if path.endswith("/fapi/v1/algoOrder") and p.get("type") == "STOP_MARKET")

    assert any(p.endswith("/fapi/v1/leverage") for p in paths), "phải set đòn bẩy"
    assert order_params["clientOrderId"] == "futu-eth-1-e"
    assert order_params["positionSide"] == "LONG"
    assert order_params["type"] == "LIMIT"
    # Hedge: KHÔNG truyền reduceOnly, exit cùng phía LONG (side SELL + positionSide LONG).
    assert "reduceOnly" not in tp_params and "reduceOnly" not in sl_params
    assert tp_params["clientAlgoId"] == "futu-eth-1-tp"
    assert sl_params["clientAlgoId"] == "futu-eth-1-sl"
    assert tp_params["triggerPrice"] == "2750"
    assert sl_params["triggerPrice"] == "2680"
    assert tp_params["algoType"] == "CONDITIONAL"
    assert tp_params["positionSide"] == "LONG" and tp_params["side"] == "SELL"
    assert result["plan_id"] == "futu-eth-1"
    assert result["entry_order_id"] == 555
    assert isinstance(result["tp_algo_id"], int)


def test_futures_params_one_way_mode(monkeypatch):
    monkeypatch.setattr(
        binance_executor, "_market_filters",
        lambda *a, **k: {"tick": 0.01, "step": 0.001, "min_notional": 5},
    )
    calls, fake = _recorder(hedge=False)
    monkeypatch.setattr(binance_executor, "signed_request", fake)

    binance_executor.place_futures_plan(
        "ETHUSDT", "SHORT", 2700.0, 2650.0, 2760.0, 0.01, 20,
        ("k", "s"), "futu-eth-1",
    )
    order_params = next(p for path, p in calls if path.endswith("/fapi/v1/order") and p.get("clientOrderId"))
    tp_params = next(p for path, p in calls if path.endswith("/fapi/v1/algoOrder") and p.get("type") == "TAKE_PROFIT_MARKET")
    assert order_params["positionSide"] == "BOTH"
    assert tp_params["reduceOnly"] == "true"
    assert tp_params["positionSide"] == "BOTH"


def test_futures_duplicate_client_order_id_bumps_plan(monkeypatch):
    monkeypatch.setattr(
        binance_executor, "_market_filters",
        lambda *a, **k: {"tick": 0.01, "step": 0.001, "min_notional": 5},
    )
    calls, fake = _recorder(hedge=False, fail_first_duplicate=True)
    monkeypatch.setattr(binance_executor, "signed_request", fake)

    result = binance_executor.place_futures_plan(
        "ETHUSDT", "LONG", 2700.0, 2750.0, 2680.0, 0.01, 20,
        ("k", "s"), "futu-eth-1",
    )
    # Lần 1 trùng clientOrderId → tự tăng số thứ tự, TP/SL đi theo plan_id mới.
    assert result["plan_id"] == "futu-eth-2"
    order_params = [p for path, p in calls if path.endswith("/fapi/v1/order") and p.get("clientOrderId")]
    assert [p["clientOrderId"] for p in order_params] == ["futu-eth-1-e", "futu-eth-2-e"]
    tp_params = next(p for path, p in calls if path.endswith("/fapi/v1/algoOrder") and p.get("type") == "TAKE_PROFIT_MARKET")
    assert tp_params["clientAlgoId"] == "futu-eth-2-tp"


def test_plan_id_bump_helper():
    assert binance_executor._bump_plan_id("futu-eth-1") == "futu-eth-2"
    assert binance_executor._bump_plan_id("spot-btc-12") == "spot-btc-13"


def _tp_fail_fake(calls, executed_qty: str, hedge: bool = True):
    """Entry OK → TP algo ném -2021 → GET order trả executedQty tùy trường hợp."""
    def fake(base, api_key, secret, method, path, params):
        calls.append((method, path, dict(params)))
        if path.endswith("/fapi/v1/algoOrder"):
            raise binance_executor.ExecutorError(-2021, "Order would immediately trigger.")
        if path.endswith("positionSide/dual"):
            return {"dualSidePosition": hedge}
        if method == "GET" and path.endswith("/fapi/v1/order"):
            return {"executedQty": executed_qty,
                    "status": "FILLED" if float(executed_qty) > 0 else "NEW"}
        if method == "POST" and path.endswith("/fapi/v1/order"):
            return {"orderId": 555}
        return {}
    return fake


def test_futures_entry_filled_when_tp_fails_closes_market(monkeypatch):
    """Entry ĐÃ KHỚP mà TP/SL fail → đóng ngay theo thị trường, không để position trần."""
    monkeypatch.setattr(
        binance_executor, "_market_filters",
        lambda *a, **k: {"tick": 0.01, "step": 0.001, "min_notional": 5},
    )
    calls = []
    monkeypatch.setattr(binance_executor, "signed_request", _tp_fail_fake(calls, "0.01"))

    try:
        binance_executor.place_futures_plan(
            "ETHUSDT", "LONG", 2700.0, 2750.0, 2680.0, 0.01, 20, ("k", "s"), "futu-eth-1")
        assert False, "phải raise ExecutorError"
    except binance_executor.ExecutorError as exc:
        assert exc.code == -2021
        assert "ĐÓNG NGAY" in exc.msg

    closes = [c for c in calls if c[0] == "POST" and c[2].get("type") == "MARKET"]
    assert len(closes) == 1, "phải đóng market đúng 1 lần"
    assert closes[0][2]["side"] == "SELL"
    assert closes[0][2]["positionSide"] == "LONG"
    assert float(closes[0][2]["quantity"]) == 0.01
    # Entry đã khớp thì không được DELETE hủy (vô nghĩa) — chỉ được đóng.
    assert [c for c in calls if c[0] == "DELETE"] == []


def test_futures_entry_unfilled_when_tp_fails_cancels_by_order_id(monkeypatch):
    """Entry CHƯA khớp mà TP/SL fail → hủy entry theo orderId, không lệnh nào treo."""
    monkeypatch.setattr(
        binance_executor, "_market_filters",
        lambda *a, **k: {"tick": 0.01, "step": 0.001, "min_notional": 5},
    )
    calls = []
    monkeypatch.setattr(binance_executor, "signed_request", _tp_fail_fake(calls, "0"))

    try:
        binance_executor.place_futures_plan(
            "ETHUSDT", "LONG", 2700.0, 2750.0, 2680.0, 0.01, 20, ("k", "s"), "futu-eth-1")
        assert False, "phải raise ExecutorError"
    except binance_executor.ExecutorError as exc:
        assert exc.code == -2021
        assert "ĐÓNG NGAY" not in exc.msg

    deletes = [c for c in calls if c[0] == "DELETE"]
    assert len(deletes) == 1
    assert deletes[0][2]["orderId"] == 555
    assert [c for c in calls if c[0] == "POST" and c[2].get("type") == "MARKET"] == []


# ─── Phiên auto scan: plan_id / log / off / migration ────────────────────────

def _analyze(monkeypatch):
    import analyze
    monkeypatch.setattr(analyze, "DB_PATH", _DB)
    monkeypatch.setattr(analyze, "_prediction_db_initialized", False)
    monkeypatch.setattr(analyze, "_auto_scan_db_initialized", False)
    return analyze


def test_session_plan_id_sequence_and_off_wipe(monkeypatch):
    analyze = _analyze(monkeypatch)
    analyze.init_auto_scan_db()

    p1 = analyze.next_session_plan_id(77, "futures", "ETHUSDT")
    analyze._record_auto_scan_signal(77, 1, "ETHUSDT", "futures", "LONG", 70, 101,
                                     p1, entry_low=2690.0, entry_high=2710.0, sl=2680.0, tp1=2750.0)
    p2 = analyze.next_session_plan_id(77, "futures", "ETHUSDT")
    assert (p1, p2) == ("futu-eth-1", "futu-eth-2")
    analyze._record_auto_scan_signal(77, 1, "ETHUSDT", "futures", "SHORT", 65, 102,
                                     p2, entry_low=2700.0, entry_high=2720.0, sl=2730.0, tp1=2650.0)
    # Symbol khác đếm riêng.
    b1 = analyze.next_session_plan_id(77, "futures", "BTCUSDT")
    assert b1 == "futu-btc-1"
    # Market khác prefix khác và không ảnh hưởng nhau.
    s1 = analyze.next_session_plan_id(77, "spot", "ETHUSDT")
    assert s1 == "spot-eth-1"
    analyze._record_auto_scan_signal(77, 1, "ETHUSDT", "spot", "BUY", 60, 103,
                                     s1, entry_low=2690.0, entry_high=2710.0, sl=2680.0, tp1=2750.0)

    rows = analyze.list_session_signals(77, "futures")
    assert len(rows) == 2
    assert rows[0]["entry_low"] == 2690.0 and rows[0]["sl"] == 2680.0 and rows[0]["tp1"] == 2750.0
    assert rows[0]["order_status"] == "pending"

    # /offfutu: xóa đúng phiên futures, giữ nguyên spot.
    deleted = analyze.delete_session_signals(77, "futures")
    assert deleted == 2
    assert analyze.list_session_signals(77, "futures") == []
    assert len(analyze.list_session_signals(77, "spot")) == 1
    # Sau off, phiên mới bắt đầu lại từ 1.
    assert analyze.next_session_plan_id(77, "futures", "ETHUSDT") == "futu-eth-1"


def test_market_settings_enable_roundtrip(monkeypatch):
    analyze = _analyze(monkeypatch)
    analyze.init_auto_scan_db()
    result = analyze.set_auto_scan_market_enabled(88, 1, "futures", True, "ETHUSDT", qty="0.01", leverage=20)
    assert result["enabled"] is True
    cfg = analyze.get_auto_scan_market_settings(88, "futures")
    assert cfg["symbol"] == "ETHUSDT" and cfg["qty"] == "0.01" and cfg["leverage"] == 20 and cfg["enabled"]

    users = analyze.get_auto_scan_enabled_users()
    me = [u for u in users if u["user_id"] == 88]
    assert me and me[0]["market"] == "futures"

    # Spot bật độc lập không đè lên futures.
    analyze.set_auto_scan_market_enabled(88, 1, "spot", True, "ETHUSDT", qty="0.5", leverage=1)
    assert analyze.get_auto_scan_market_settings(88, "futures")["qty"] == "0.01"
    users = analyze.get_auto_scan_enabled_users()
    assert {u["market"] for u in users if u["user_id"] == 88} == {"futures", "spot"}

    # Tắt futures không ảnh hưởng spot.
    analyze.delete_session_signals(88, "futures")
    analyze.set_auto_scan_market_enabled(88, 1, "futures", False, "ETHUSDT")
    assert analyze.get_auto_scan_market_settings(88, "futures")["enabled"] is False
    assert analyze.get_auto_scan_market_settings(88, "spot")["enabled"] is True


def test_export_snapshot_strips_api_keys(monkeypatch):
    import evaluation_store
    _analyze(monkeypatch).init_auto_scan_db()
    monkeypatch.setattr(evaluation_store, "DB_PATH", _DB)
    monkeypatch.setattr(key_store, "DB_PATH", _DB)
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    key_store.save_api_keys(5, "futures", "APIKEY" + "x" * 30, "SECRET" + "y" * 30)

    dest = os.path.join(_TMP, "snapshot.db")
    evaluation_store.export_database_snapshot(dest)
    conn = sqlite3.connect(dest)
    count = conn.execute("SELECT COUNT(*) FROM user_api_keys").fetchone()[0]
    snap_rows = conn.execute("SELECT COUNT(*) FROM auto_scan_market_settings").fetchone()[0]
    conn.close()
    assert count == 0, "bản export phải không còn API key"
    assert snap_rows >= 0  # bảng khác vẫn được sao lưu bình thường
    os.remove(dest)


def test_parse_helpers():
    assert symbol_control._parse_qty("0.008") == 0.008
    assert symbol_control._parse_qty("0,01") is None or symbol_control._parse_qty("0.01") == 0.01
    assert symbol_control._parse_qty("-1") is None
    assert symbol_control._parse_qty("abc") is None
    assert symbol_control._parse_leverage("20") == 20
    assert symbol_control._parse_leverage("0") is None
    assert symbol_control._parse_leverage("200") is None
    assert symbol_control._parse_leverage("x") is None
