"""IT: Auto Scan → ĐẶT LỆNH (futures + spot) → ledger → HỦY lệnh.

Chỉ mock ở ranh giới HTTP (signed_request / public_get) và LLM; toàn bộ code
executor + analyze chạy THẬT. Không cần mạng — chạy được trong CI.

Mục đích: bắt các bug chỉ lộ khi ghép 2 tầng với nhau (vd plan_id không scope,
tp2 bị mất khi truyền qua 4 lớp, OCO sai coid...).
"""
import asyncio
import json
import os
import sqlite3

import pytest

import analyze
import binance_executor
import key_store

_TMP = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".pytest_it")
_DB = os.path.join(_TMP, "it_autoscan.db")


def _async(value):
    async def _inner(*a, **k):
        return value
    return _inner()


def _s(v: float) -> str:
    return binance_executor._fmt(v)


# ─── Fake ở ranh giới HTTP (executor vẫn chạy logic thật) ────────────────────

def _exchange_info(symbol: str, step: float, tick: float, min_notional: float, base: str):
    return {"symbols": [{
        "symbol": symbol, "status": "TRADING", "baseAsset": base,
        "filters": [
            {"filterType": "PRICE_FILTER", "minPrice": "0.01",
             "maxPrice": "1000000", "tickSize": _s(tick)},
            {"filterType": "LOT_SIZE", "minQty": _s(step),
             "maxQty": "9000", "stepSize": _s(step)},
            {"filterType": "NOTIONAL", "minNotional": str(min_notional),
             "applyMinToMarket": True},
        ],
    }]}


class HttpRecorder:
    """Ghi lại mọi lệnh HTTP gửi lên sàn; trả response hợp lệ để code đi tiếp.

    entry_filled=False → entry se van NEW (dung cho truong hop can huy lenh treo);
    entry_filled=True → entry FILLED ngay (dung cho truong hop can gan OCO).
    """

    def __init__(self, *, entry_filled=False):
        self.calls = []
        self.entry_filled = entry_filled
        self._n = 0
        # Số coin "thật" đã khớp của MARKET BUY gần nhất. Entry qty giờ được tính từ số
        # USDT, nên response giả phải echo đúng số coin đã yêu cầu chứ không phải số cứng.
        self.market_qty = None

    def _next_id(self):
        self._n += 1
        return 1000 + self._n

    def signed_request(self, base, api_key, secret, method, path, params):
        self.calls.append((method, path, dict(params)))
        if path.endswith("/fapi/v1/positionSide/dual"):
            return {"dualSidePosition": False}
        if path.endswith("/fapi/v1/leverage") or path.endswith("/multiAssetsMargin"):
            return {}
        if path.endswith("/fapi/v1/algoOrder"):
            return {"algoId": self._next_id()}
        if path.endswith("/fapi/v1/time") or path.endswith("/api/v3/time"):
            import time as _t
            return {"serverTime": int(_t.time() * 1000)}
        if path.endswith("/api/v3/account"):
            return {"balances": [{"asset": "BTC", "free": "1.0", "locked": "0"}],
                    "commissionRates": {"maker": "0.001", "taker": "0.001"}}
        if path.endswith("/orderList/otoco"):
            return {"orderListId": self._next_id(),
                    "orderReports": [
                        {"type": "LIMIT", "side": "BUY", "orderId": self._next_id(),
                         "clientOrderId": params.get("workingClientOrderId")},
                        {"type": "LIMIT_MAKER", "side": "SELL", "orderId": self._next_id(),
                         "clientOrderId": params.get("pendingAboveClientOrderId")},
                        {"type": "STOP_LOSS", "side": "SELL", "orderId": self._next_id(),
                         "clientOrderId": params.get("pendingBelowClientOrderId")},
                    ]}
        if path.endswith("/api/v3/orderList/oco") or path.endswith("/api/v3/order/oco"):
            coid = params.get("aboveClientOrderId") or params.get("limitClientOrderId")
            return {"orderListId": self._next_id(),
                    "orders": [{"clientOrderId": coid, "orderId": self._next_id()},
                               {"clientOrderId": params.get("belowClientOrderId")
                                or params.get("stopClientOrderId"),
                                "orderId": self._next_id()}]}
        if path.endswith("/fapi/v1/order") or path.endswith("/api/v3/order"):
            if method == "POST":
                response = {"orderId": self._next_id(),
                            "clientOrderId": params.get("clientOrderId")
                            or params.get("newClientOrderId")}
                if path.endswith("/api/v3/order") and params.get("type") == "MARKET":
                    # Echo đúng số coin đã yêu cầu (qty nay duoc tinh tu USDT).
                    self.market_qty = params.get("quantity") or self.market_qty
                    response.update({"status": "FILLED",
                                     "executedQty": self.market_qty,
                                     "cummulativeQuoteQty": "27000.0"})
                return response
            if self.entry_filled:
                return {"status": "FILLED",
                        "executedQty": self.market_qty or "0.1"}
            return {"status": "NEW", "executedQty": "0"}
        if method == "DELETE":
            return {"status": "CANCELED"}
        return {}

    def public_get(self, base, path, params=None):
        params = params or {}
        if path.endswith("/exchangeInfo"):
            return _exchange_info(params.get("symbol", "BTCUSDT"),
                                  0.001, 0.01, 5.0, "BTC")
        if path.endswith("/ticker/price"):
            return {"price": "2700.00"}
        if path.endswith("/time"):
            import time as _t
            return {"serverTime": int(_t.time() * 1000)}
        return {}

    def paths(self, suffix: str, method=None):
        return [(m, p, prm) for m, p, prm in self.calls
                if p.endswith(suffix) and (method is None or m == method)]


@pytest.fixture
def http(monkeypatch):
    rec = HttpRecorder()
    monkeypatch.setattr(binance_executor, "signed_request", rec.signed_request)
    monkeypatch.setattr(binance_executor, "public_get", rec.public_get)
    # Cache filter theo (base, symbol, spot) là state TOÀN CỤC. Nếu test khác đã cache
    # entry cho cùng key, fake ở file này sẽ không chạy và `/api/v3/account` trả base
    # coin khác → free=0. Xoá sạch để test không phụ thuộc thứ tự chạy của file khác.
    binance_executor._FILTER_CACHE.clear()
    # Không gọi mạng thật để map symbol demo.
    monkeypatch.setattr(analyze, "_resolve_demo_symbol", lambda base, sym: sym)
    monkeypatch.setattr(binance_executor, "SPOT_FILL_TIMEOUT_SECONDS", 1)
    return rec


@pytest.fixture
def http_filled(monkeypatch):
    """Cung nhu `nhung entry da khop` — cho cac truong hop can gan OCO."""
    rec = HttpRecorder(entry_filled=True)
    monkeypatch.setattr(binance_executor, "signed_request", rec.signed_request)
    monkeypatch.setattr(binance_executor, "public_get", rec.public_get)
    binance_executor._FILTER_CACHE.clear()
    monkeypatch.setattr(analyze, "_resolve_demo_symbol", lambda base, sym: sym)
    return rec


@pytest.fixture
def env(monkeypatch, tmp_path):
    db = str(tmp_path / "it.db")
    monkeypatch.setattr(analyze, "DB_PATH", db)
    monkeypatch.setattr(analyze, "_prediction_db_initialized", False)
    monkeypatch.setattr(analyze, "_auto_scan_db_initialized", False)
    monkeypatch.setattr(key_store, "DB_PATH", db)
    # Key mã hóa chỉ tồn tại trong lúc test — sinh random, không bao giờ dùng key thật.
    from cryptography.fernet import Fernet
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    key_store.save_api_keys(77, "APIKEY" + "x" * 30, "SECRET" + "y" * 30)
    analyze.init_prediction_db()
    analyze.init_auto_scan_db()
    return db


PLAN_TP2 = {
    "quyet_dinh": "LONG",
    "entry_thap": 2690.0, "entry_cao": 2710.0,
    "sl": 2600.0, "tp1": 2750.0, "tp2": 2800.0,
    "do_tin_cay": 80,
}

PLAN_NO_TP2 = {
    "quyet_dinh": "LONG",
    "entry_thap": 2690.0, "entry_cao": 2710.0,
    "sl": 2600.0, "tp1": 2750.0,
    "do_tin_cay": 80,
}

# Spot chi chap nhan BUY/NO_TRADE (khac futures LONG/SHORT)
SPOT_PLAN = {"quyet_dinh": "BUY", "entry_thap": 2690.0, "entry_cao": 2710.0,
             "sl": 2600.0, "tp1": 2750.0, "do_tin_cay": 80}


# ─── _auto_execute_plan: tầng đặt lệnh thật ──────────────────────────────────

def test_it_futures_places_entry_tp1_70_tp2_30_sl_100(env, http, monkeypatch):
    """Futures: entry theo giá PLAN, TP1 70%, TP2 30%, SL 100% (task 1).

    Cấu hình phiên nay là số USDT: 100 USDT x đòn bẩy 10 = vị thế 1000 USDT, chia cho giá
    entry 2710 → 0.369 ETH.
    """
    analyze.set_auto_scan_market_enabled(77, 1, "futures", True, "ETHUSDT",
                                         qty="100", leverage=10)
    monkeypatch.setattr(analyze, "get_current_price_raw", lambda sym, mkt: 2700.0)

    block, info = asyncio.run(analyze._auto_execute_plan(
        user_id=77, mode="futures", symbol="ETHUSDT", direction="LONG",
        plan=dict(PLAN_TP2), plan_id="it-futu-77-1", current_price=2700.0))

    assert info["status"] == "placed", info
    order = info["order"]
    assert order["tp_algo_id"] is not None and order["tp2_algo_id"] is not None
    assert order["sl_algo_id"] is not None
    # TP1 70%, TP2 30%, SL 100%
    assert order["qty"] == 0.369
    assert order["qty_tp1"] == 0.258 and order["qty_tp2"] == 0.111
    assert "Chốt lời" in block, "phải báo cho user biết lệnh đã chia 70/30"

    # Entry lay GIA CUA PLAN (khong phai gia packet)
    entry = next(prm for m, p, prm in http.paths("/fapi/v1/order", "POST")
                 if prm.get("type") == "LIMIT")
    assert entry["price"] == "2710", "entry phai la entry_cao cua LONG"
    assert entry["quantity"] == "0.369", "quantity phai tinh tu USDT / entry_price"

    algos = http.paths("/fapi/v1/algoOrder", "POST")
    tp_algos = [prm for _, _, prm in algos if prm["type"] == "TAKE_PROFIT_MARKET"]
    sl_algos = [prm for _, _, prm in algos if prm["type"] == "STOP_MARKET"]
    assert len(tp_algos) == 2 and len(sl_algos) == 1
    assert float(tp_algos[0]["quantity"]) == 0.258
    assert float(tp_algos[1]["quantity"]) == 0.111
    assert float(sl_algos[0]["quantity"]) == 0.369, "SL phai la 100% vi the"
    assert tp_algos[0]["triggerPrice"] == "2750"
    assert tp_algos[1]["triggerPrice"] == "2800"


def test_it_futures_without_tp2_uses_full_qty(env, http, monkeypatch):
    """KHONG co tp2 → TP1 phai nhan TOAN BO qty, khong duoc chia 70% roi bo 30% lai troi."""
    analyze.set_auto_scan_market_enabled(77, 1, "futures", True, "ETHUSDT",
                                         qty="100", leverage=10)
    monkeypatch.setattr(analyze, "get_current_price_raw", lambda sym, mkt: 2700.0)

    block, info = asyncio.run(analyze._auto_execute_plan(
        user_id=77, mode="futures", symbol="ETHUSDT", direction="LONG",
        plan=dict(PLAN_NO_TP2), plan_id="it-futu-77-2", current_price=2700.0))

    assert info["status"] == "placed", info
    tp_algos = [prm for _, _, prm in http.paths("/fapi/v1/algoOrder", "POST")
                if prm["type"] == "TAKE_PROFIT_MARKET"]
    assert len(tp_algos) == 1
    assert float(tp_algos[0]["quantity"]) == 0.369, "TP1 phai la toan bo qty"
    assert info["order"]["tp2_algo_id"] is None


def test_it_spot_market_entry_then_oco(env, http, monkeypatch):
    """Spot: market entry rồi gắn OCO, không dùng OTOCO.

    Spot không có đòn bẩy: 270 USDT / 2700 giá = 0.1 ETH.
    """
    analyze.set_auto_scan_market_enabled(77, 1, "spot", True, "ETHUSDT",
                                         qty="270", leverage=1)
    monkeypatch.setattr(analyze, "get_current_price_raw", lambda sym, mkt: 2700.0)

    block, info = asyncio.run(analyze._auto_execute_plan(
        user_id=77, mode="spot", symbol="ETHUSDT", direction="LONG",
        plan=dict(PLAN_NO_TP2), plan_id="it-spot-77-1", current_price=2700.0))

    assert info["status"] == "placed", info
    assert info["order"]["market_entry"] is True
    assert info["order"]["protected"] is True
    market = [p for _, _, p in http.paths("/api/v3/order", "POST")]
    assert len(market) == 1 and market[0]["type"] == "MARKET"
    assert float(market[0]["quantity"]) == 0.1, "270 USDT / 2700 = 0.1 ETH"
    oco = http.paths("/api/v3/orderList/oco", "POST")
    assert len(oco) == 1
    assert oco[0][2]["abovePrice"] == "2750"
    assert oco[0][2]["belowStopPrice"] == "2600"
    assert float(oco[0][2]["quantity"]) == 0.1


def test_it_spot_with_tp2_uses_two_ocos_after_fill(http_filled, env, monkeypatch):
    """Spot + tp2: OCO khong chia duoc 70/30 → entry don, sau khop gan 2 OCO (task 1)."""
    analyze.set_auto_scan_market_enabled(77, 1, "spot", True, "ETHUSDT",
                                         qty="270", leverage=1)
    monkeypatch.setattr(analyze, "get_current_price_raw", lambda sym, mkt: 2700.0)

    block, info = asyncio.run(analyze._auto_execute_plan(
        user_id=77, mode="spot", symbol="ETHUSDT", direction="LONG",
        plan=dict(PLAN_TP2), plan_id="it-spot-77-2", current_price=2700.0))

    assert info["status"] == "placed", info
    # Phai co entry don truoc
    assert http_filled.paths("/api/v3/order", "POST"), "phai dat entry rieng khi co tp2"
    assert not http_filled.paths("/api/v3/orderList/otoco"), "khong duoc dung OTOCO"

    oco = [prm for _, _, prm in http_filled.paths("/api/v3/orderList/oco", "POST")]
    assert len(oco) == 2, f"phai co 2 OCO khi co tp2, got {len(oco)}"
    assert float(oco[0]["quantity"]) + float(oco[1]["quantity"]) == pytest.approx(0.1)
    assert oco[0]["abovePrice"] == "2750" and oco[1]["abovePrice"] == "2800"
    assert oco[0]["aboveClientOrderId"] != oco[1]["aboveClientOrderId"]
    assert info["order"]["protected"] is True


# ─── Full pipeline: auto_scan_symbol_for_user → ledger ───────────────────────

def _run_autoscan(monkeypatch, mode: str, plan: dict, symbol="ETHUSDT", user=77):
    """Chạy full Auto Scan (LLM mocked) → trả (result, dong ledger trong DB)."""
    import pandas as pd

    fake_dfs = {"4H": pd.DataFrame({"close": [1.0]})} if mode == "futures" else \
               {"4H": pd.DataFrame({"close": [1.0]})}
    monkeypatch.setattr(analyze, "collect_timeframe_data", lambda *a, **k: _async(fake_dfs))
    monkeypatch.setattr(analyze, "_missing_critical_timeframes", lambda *a, **k: [])
    monkeypatch.setattr(analyze, "prepare_analysis_context",
                        lambda *a, **k: _async(_canned_ctx()))
    monkeypatch.setattr(analyze, "request_json_analysis", lambda s, u: json.dumps(plan))
    monkeypatch.setattr(analyze, "_auto_scan_consume_trend_skip", lambda *a: None)
    monkeypatch.setattr(analyze, "get_current_price_raw", lambda sym, mkt: 2700.0)

    import plan_validator
    monkeypatch.setattr(plan_validator, "validate_plan", lambda plan, facts: [])

    # Cấu hình phiên là số USDT: futures 100 USDT x đòn bẩy 10 = vị thế 1000 USDT;
    # spot 270 USDT (không có đòn bẩy) = vị thế 270 USDT.
    analyze.set_auto_scan_market_enabled(user, 1, mode, True, symbol,
                                         qty="100" if mode == "futures" else "270",
                                         leverage=10 if mode == "futures" else 1)
    key_store.save_api_keys(user, "APIKEY" + "x" * 30, "SECRET" + "y" * 30)
    conn = sqlite3.connect(analyze.DB_PATH)
    conn.execute("DELETE FROM auto_scan_signals")
    conn.commit()
    conn.close()

    result = asyncio.run(analyze.auto_scan_symbol_for_user(symbol, mode, user, 1,
                                                            scan_slot="slot-1"))
    conn = sqlite3.connect(analyze.DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM auto_scan_signals WHERE user_id=? AND mode=? ORDER BY id DESC LIMIT 1",
        (user, mode)).fetchone()
    conn.close()
    return result, (dict(row) if row else None)


def _canned_ctx():
    return {"timeframe_data": {}, "system_prompt": "SP", "current_price": 2700.0,
            "user_prompt": "UP", "feature_snapshot": "FS", "market_snapshot": "MS",
            "funding_context": None, "facts": {"price": 2700.0, "current_price": 2700.0}}


def test_it_autoscan_futures_full_pipeline_records_tp_ids(env, http, monkeypatch):
    """Full pipeline futures: LLM → dat lenh → ghi ledger du 3 id (tp, tp2, sl)."""
    result, row = _run_autoscan(monkeypatch, "futures", PLAN_TP2)
    assert result.get("send") is True, result
    assert row is not None, "phai co dong auto_scan_signals"
    assert row["order_status"] == "placed"
    assert row["entry_order_id"], "phai ghi entry orderId"
    assert row["tp_algo_id"], "phai ghi TP1 algo id"
    assert row["tp2_algo_id"], "phai ghi TP2 algo id"
    assert row["sl_algo_id"], "phai ghi SL algo id"
    # Ledger lưu giá trị VỊ THẾ (USDT) = qty cấu hình × đòn bẩy: 100 × 10 = 1000.
    assert row["qty"] == "1000.0", row["qty"]
    # plan_id mang user_id de khong trung giua 2 user
    assert str(row["user_id"]) in row["plan_id"], row["plan_id"]
    assert row["tp2"] == pytest.approx(2800.0), "phai luu tp2 de safety net dung lai duoc"


def test_it_autoscan_spot_full_pipeline_records_oco(env, http, monkeypatch):
    """Full pipeline spot: LLM → market entry + OCO → ledger ghi oco_list_id."""
    result, row = _run_autoscan(monkeypatch, "spot", SPOT_PLAN)
    assert result.get("send") is True, result
    assert row is not None
    assert row["order_status"] == "placed"
    assert row["entry_order_id"], "phai ghi entry orderId"
    assert row["tp_algo_id"], "phai ghi oco_list_id vao tp_algo_id"
    assert row["tp2"] is None, "khong co tp2 thi tp2 la NULL"


# ─── Huy lenh treo ────────────────────────────────────────────────────────────

def test_it_cancel_pending_futures_cancels_entry_and_algos(env, http, monkeypatch):
    """Tat autoscan → huy entry + TP1 + TP2 + SL (khong de lenh mo coi)."""
    _, row = _run_autoscan(monkeypatch, "futures", PLAN_TP2)
    assert row is not None

    cancelled = binance_executor.cancel_pending_plan_orders(
        ("k", "s"), "futures",
        [{"user_id": 77, "mode": "futures", "symbol": "ETHUSDT",
          "entry_order_id": row["entry_order_id"],
          "tp_algo_id": row["tp_algo_id"], "tp2_algo_id": row["tp2_algo_id"],
          "sl_algo_id": row["sl_algo_id"]}])

    kinds = {c["kind"] for c in cancelled}
    assert "entry" in kinds, f"phai huy entry, got {cancelled}"
    algo_ids = [c["algo_id"] for c in cancelled if c["kind"] == "algo"]
    assert len(algo_ids) == 3, f"phai huy ca 3 algo (TP1+TP2+SL), got {algo_ids}"
    assert row["tp_algo_id"] in algo_ids and row["tp2_algo_id"] in algo_ids
    assert row["sl_algo_id"] in algo_ids


def test_it_cancel_pending_spot_keeps_filled_market_position(env, http_filled, monkeypatch):
    """Market entry đã khớp thì off không được hủy position/protection."""
    _, row = _run_autoscan(monkeypatch, "spot", SPOT_PLAN)
    assert row is not None

    # Entry đã FILLED → giữ nguyên position và OCO.
    cancelled = binance_executor.cancel_pending_plan_orders(
        ("k", "s"), "spot",
        [{"user_id": 77, "mode": "spot", "symbol": "ETHUSDT",
          "entry_order_id": row["entry_order_id"],
          "tp_algo_id": row["tp_algo_id"], "tp2_algo_id": row["tp2_algo_id"],
          "sl_algo_id": row["sl_algo_id"]}])
    assert not any(c["kind"] == "entry" for c in cancelled), cancelled
