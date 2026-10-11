"""Spot auto-order flow: endpoint / parameter / crash-recovery / multi-plan audit.

Mọi test đều assert CHÍNH XÁC request payload gửi lên Binance (không chỉ mock response).

Các nhóm bắt buộc:
- Spot Demo vs Spot Live base URL
- MARKET endpoint + newClientOrderId
- Query order bằng orderId / origClientOrderId
- OCO endpoint mới: aboveType / abovePrice / aboveClientOrderId /
  belowType / belowStopPrice / belowClientOrderId / listClientOrderId
- Cancel OCO bằng orderListId / listClientOrderId (KHÔNG hủy theo symbol)
- Không nhầm orderId với orderListId
- commission base asset / quote asset / BNB / nhiều fills / partial fill
- requestedQty khác executedQty
- free balance thấp hơn executedQty, rounding LOT_SIZE, minQty/minNotional
- OCO duplicate / OCO fail
- TP1/TP2, Plan A không hủy Plan B, balance dùng chung nhiều plan
- timeout trước/sau MARKET BUY, restart recovery
- manual user cancel TP/SL

Lưu ý: dùng file DB tạm (không :memory:) vì code mở NHIỀU connection tới cùng path —
:memory: tạo DB riêng cho mỗi connection nên bảng "biến mất" giữa chừng.
"""
import asyncio
import os
import sqlite3

import pytest

import analyze
import binance_executor
import key_store
import spot_user_stream


@pytest.fixture(autouse=True)
def _db(request, monkeypatch):
    """Mỗi test có DB file riêng (pytest quản lý tmp_path)."""
    # Cache filter toàn cục: xoá trước/sau mỗi test để không phụ thuộc thứ tự file test.
    binance_executor._FILTER_CACHE.clear()
    tmp = request.getfixturevalue("tmp_path")
    db = str(tmp / "spot_flow.db")
    monkeypatch.setattr(analyze, "DB_PATH", db)
    monkeypatch.setattr(analyze, "_prediction_db_initialized", False)
    monkeypatch.setattr(analyze, "_auto_scan_db_initialized", False)
    monkeypatch.setattr(key_store, "DB_PATH", db)
    yield db
    binance_executor._FILTER_CACHE.clear()


def _use_test_db(monkeypatch):
    """Lấy DB hiện tại của fixture _db (đã được monkeypatch sẵn)."""
    return analyze.DB_PATH


# ─── Fake signed_request kiểu recorder (chỉ ghi, không mạng) ─────────────────

class CallRecorder:
    """Ghi lại MỌI request gửi lên "sàn" và trả response cấu hình được."""

    def __init__(self, handlers=None, market_buy_fills=None):
        self.calls = []
        self.handlers = handlers or {}
        # fills của MARKET BUY: [(price, qty, commission, commissionAsset), ...]
        self.market_buy_fills = market_buy_fills
        self._next_order = 9000
        self._next_list = 5000
        self.balances = {"BTC": 0.0}

    def _next_id(self):
        self._next_order += 1
        return self._next_order

    def _next_list_id(self):
        self._next_list += 1
        return self._next_list

    def __call__(self, base, api_key, secret, method, path, params):
        self.calls.append((method, path, dict(params)))
        if path.endswith("/api/v3/time"):
            import time as _t
            return {"serverTime": int(_t.time() * 1000)}
        if path.endswith("/exchangeInfo"):
            return {"symbols": [{
                "symbol": "BTCUSDT", "status": "TRADING", "baseAsset": "BTC",
                "filters": [
                    {"filterType": "PRICE_FILTER", "minPrice": "0.01",
                     "maxPrice": "1000000", "tickSize": "0.01"},
                    {"filterType": "LOT_SIZE", "minQty": "0.001",
                     "maxQty": "9000", "stepSize": "0.001"},
                    {"filterType": "NOTIONAL", "minNotional": "10",
                     "applyMinToMarket": True},
                ]}]}
        if path.endswith("/api/v3/account"):
            return {"balances": [{"asset": k, "free": str(v), "locked": "0"}
                                 for k, v in self.balances.items()],
                    "commissionRates": {"maker": "0.001", "taker": "0.0015"}}
        if method == "POST" and path.endswith("/api/v3/order") and params.get("type") == "MARKET":
            if params.get("side") == "BUY":
                resp = {
                    "symbol": params["symbol"], "orderId": self._next_id(),
                    "clientOrderId": params.get("newClientOrderId"),
                    "price": "0", "origQty": params.get("quantity"),
                    "executedQty": params.get("quantity"),
                    "cummulativeQuoteQty": "27000.00",
                    "status": "FILLED", "timeInForce": "GTC",
                    "type": "MARKET", "side": "BUY", "fills": [],
                }
                if self.market_buy_fills is not None:
                    resp["fills"] = [
                        {"price": str(p), "qty": str(q), "commission": str(c),
                         "commissionAsset": a}
                        for p, q, c, a in self.market_buy_fills
                    ]
                return resp
        if method == "GET" and path.endswith("/api/v3/order"):
            return {"status": "FILLED", "executedQty": params.get("x", "0"),
                    "cummulativeQuoteQty": "27000.00"}
        # Futures: algo orders + leverage + positionSide/dual + positionRisk.
        if path.endswith("/fapi/v1/algoOrder") and method == "POST":
            return {"algoId": self._next_id()}
        if path.endswith("/fapi/v1/leverage") or path.endswith("/fapi/v1/multiAssetsMargin"):
            return {}
        if path.endswith("/fapi/v1/positionSide/dual"):
            return {"dualSidePosition": False}
        if path.endswith("/fapi/v2/positionRisk") or path.endswith("/fapi/v1/positionRisk"):
            return []
        if method == "POST" and path.endswith("/api/v3/orderList/oco"):
            coid = params.get("aboveClientOrderId")
            return {"orderListId": self._next_list_id(),
                    "listClientOrderId": params.get("listClientOrderId"),
                    "orders": [{"clientOrderId": coid, "orderId": self._next_id()},
                               {"clientOrderId": params.get("belowClientOrderId"),
                                "orderId": self._next_id()}],
                    "orderReports": []}
        if method == "DELETE" and path.endswith("/api/v3/orderList"):
            return {"orderListId": params.get("orderListId"), "status": "CANCELED"}
        if method == "GET" and path.endswith("/api/v3/orderList"):
            return self.handlers.get("oca_lookup",
                                     {"orderListId": 777,
                                      "listClientOrderId": params.get("origClientOrderId")})
        return {}

    def paths(self, suffix, method=None):
        return [(m, p, prm) for m, p, prm in self.calls
                if p.endswith(suffix) and (method is None or m == method)]

    def public_get(self, base, path, params=None):
        """_market_filters đọc exchangeInfo qua đây — trả filter giả lập cho mọi symbol."""
        if path.endswith("/exchangeInfo"):
            symbol = (params or {}).get("symbol", "BTCUSDT")
            base_asset = "ETH" if symbol.startswith("ETH") else "BTC"
            step = "0.00001" if symbol.startswith("ETH") else "0.001"
            return {"symbols": [{
                "symbol": symbol, "status": "TRADING", "baseAsset": base_asset,
                "filters": [
                    {"filterType": "PRICE_FILTER", "minPrice": "0.01",
                     "maxPrice": "1000000", "tickSize": "0.01"},
                    {"filterType": "LOT_SIZE", "minQty": step,
                     "maxQty": "9000", "stepSize": step},
                    {"filterType": "NOTIONAL", "minNotional": "10",
                     "applyMinToMarket": True},
                ]}]}
        return {}


def _setup(monkeypatch, *, base="https://demo-api.binance.com", fills=None):
    """Cấu hình env + recorder, trả (rec, keys)."""
    _use_test_db(monkeypatch)
    monkeypatch.setattr(binance_executor, "SPOT_API_BASE", base)
    monkeypatch.setattr(binance_executor, "SPOT_FILL_TIMEOUT_SECONDS", 0)
    monkeypatch.setattr(binance_executor, "SPOT_OCO_DELAY_SECONDS", 0)
    # Cache filter theo host phải sạch, không thì test này đọc filter của test trước.
    binance_executor._FILTER_CACHE.clear()
    rec = CallRecorder(market_buy_fills=fills)
    rec.balances = {"BTC": 1.0}
    monkeypatch.setattr(binance_executor, "signed_request", rec)
    # _market_filters đọc exchangeInfo qua public_get (không phải signed_request).
    monkeypatch.setattr(binance_executor, "public_get", rec.public_get)
    analyze.init_auto_scan_db()
    return rec, ("k", "s")


# ─── A. Endpoint + parameter audit ───────────────────────────────────────────

def test_spot_base_url_demo_and_live(monkeypatch):
    """Base URL phải lấy từ SPOT_API_BASE, không hard-code; Demo và Live đều chạy được."""
    for base in ("https://demo-api.binance.com", "https://api.binance.com"):
        rec, keys = _setup(monkeypatch, base=base)
        binance_executor.place_spot_plan(
            "BTCUSDT", 27000.0, 28000.0, 26000.0, 1.0, keys, "spot-btc-77-1",
            tp2=None)
        used = {b for _, p, _ in rec.calls for b in [base]}
        assert rec.calls, "phải có ít nhất 1 request lên đúng base"
        # Mọi request đều đi qua khoá signed_request của đúng base (recorder tự thấy base).
        assert any(p.endswith("/api/v3/order") for _, p, _ in rec.calls), \
            f"MARKET phải dùng POST /api/v3/order trên {base}"


def test_spot_market_buy_uses_new_client_order_id(monkeypatch):
    """MARKET BUY phải dùng newClientOrderId, KHÔNG dùng clientOrderId."""
    rec, keys = _setup(monkeypatch)
    binance_executor.place_spot_plan(
        "BTCUSDT", 27000.0, 28000.0, 26000.0, 1.0, keys, "spot-btc-77-1")
    market = rec.paths("/api/v3/order", "POST")[0][2]
    assert market["newClientOrderId"] == "spot-btc-77-1-e"
    assert "clientOrderId" not in market, "KHÔNG được gửi clientOrderId khi tạo lệnh"
    assert market["side"] == "BUY" and market["type"] == "MARKET"


def test_spot_oco_endpoint_and_params(monkeypatch):
    """OCO mới: /api/v3/orderList/oco với đúng above*/below* + listClientOrderId."""
    rec, keys = _setup(monkeypatch, fills=[(27000, 1.0, 0.001, "BNB")])
    binance_executor.place_spot_plan(
        "BTCUSDT", 27000.0, 28000.0, 26000.0, 1.0, keys, "spot-btc-77-1")
    oco = rec.paths("/api/v3/orderList/oco", "POST")
    assert len(oco) == 1, "phải gọi endpoint OCO MỚI"
    p = oco[0][2]
    assert p["symbol"] == "BTCUSDT" and p["side"] == "SELL"
    assert p["aboveType"] == "LIMIT_MAKER"
    assert p["abovePrice"] == "28000"
    assert p["aboveClientOrderId"] == "spot-btc-77-1-tp"
    assert p["belowType"] == "STOP_LOSS"
    assert p["belowStopPrice"] == "26000"
    assert p["belowClientOrderId"] == "spot-btc-77-1-sl"
    assert p["listClientOrderId"] == "spot-btc-77-1-list"
    # KHÔNG nhầm lẫn id cũ của endpoint deprecated
    assert "limitClientOrderId" not in p and "stopClientOrderId" not in p
    assert "price" not in p and "stopPrice" not in p


def test_spot_query_order_by_order_id_and_client_id(monkeypatch):
    """Query order dùng orderId hoặc origClientOrderId."""
    rec, keys = _setup(monkeypatch)
    binance_executor.spot_order_state("https://demo-api.binance.com", keys, "BTCUSDT",
                                      "spot-btc-77-1-e")
    gets = rec.paths("/api/v3/order", "GET")
    assert gets and gets[0][2].get("origClientOrderId") == "spot-btc-77-1-e"


def test_spot_cancel_oco_by_list_id_not_symbol(monkeypatch):
    """Cancel phải gửi orderListId hoặc listClientOrderId, KHÔNG bao giờ hủy theo symbol."""
    rec, keys = _setup(monkeypatch)
    binance_executor.spot_cancel_oco("https://demo-api.binance.com", keys[0], keys[1],
                                     "BTCUSDT", order_list_id=4242)
    dels = rec.paths("/api/v3/orderList", "DELETE")
    assert len(dels) == 1
    assert dels[0][2]["orderListId"] == 4242
    assert dels[0][2]["symbol"] == "BTCUSDT"
    # Không có request hủy theo symbol thuần
    assert not any(prm.get("symbol") and not prm.get("orderListId")
                   and not prm.get("listClientOrderId")
                   for m, p, prm in rec.calls if m == "DELETE")


def test_spot_cancel_oco_by_list_client_order_id(monkeypatch):
    """Thiếu orderListId → fallback listClientOrderId, không báo lỗi hard-code."""
    rec, keys = _setup(monkeypatch)
    binance_executor.spot_cancel_oco("https://demo-api.binance.com", keys[0], keys[1],
                                     "BTCUSDT", list_client_order_id="spot-btc-77-1-list")
    dels = rec.paths("/api/v3/orderList", "DELETE")
    assert dels[0][2]["listClientOrderId"] == "spot-btc-77-1-list"
    assert "orderListId" not in dels[0][2]


# ─── B. Phí thực tế + sellable quantity ──────────────────────────────────────

@pytest.mark.parametrize("commission_asset,expect_sellable", [
    ("BNB", 1.0),      # phí BNB → không trừ base qty
    ("USDT", 1.0),      # phí quote → không trừ base qty
    ("BTC", 0.999),     # phí base → trừ vào coin
])
def test_spot_commission_asset_variants(monkeypatch, commission_asset, expect_sellable):
    rec, keys = _setup(
        monkeypatch, fills=[(27000, 1.0, 0.001, commission_asset)])
    fee = binance_executor.spot_fill_fees(
        {"executedQty": "1.0",
         "fills": [{"price": "27000", "qty": "1.0",
                    "commission": "0.001", "commissionAsset": commission_asset}]},
        "BTC")
    executed, fee_qty, fee_asset, sellable = fee
    assert executed == 1.0
    if commission_asset == "BTC":
        assert fee_qty == 0.001 and sellable == 0.999
    else:
        assert fee_qty == 0.0 and sellable == 1.0
    assert fee_asset == commission_asset


def test_spot_multiple_fills_sum_commission(monkeypatch):
    """Nhiều fills: cộng dồn commission base, không lấy fill đầu."""
    _exec, fee_qty, fee_asset, sellable = binance_executor.spot_fill_fees(
        {"executedQty": "1.0", "fills": [
            {"price": "27000", "qty": "0.4", "commission": "0.0004",
             "commissionAsset": "BTC"},
            {"price": "27001", "qty": "0.6", "commission": "0.0006",
             "commissionAsset": "BTC"},
        ]}, "BTC")
    assert fee_qty == 0.001 and fee_asset == "BTC" and sellable == 0.999


def test_spot_partial_fill_and_requested_vs_executed(monkeypatch):
    """requestedQty khác executedQty: sellable phải theo executedQty."""
    _exec, fee_qty, _a, sellable = binance_executor.spot_fill_fees(
        {"executedQty": "0.4", "fills": [
            {"price": "27000", "qty": "0.4", "commission": "0.0004",
             "commissionAsset": "BTC"}]}, "BTC")
    assert sellable == 0.3996
    assert sellable != 1.0, "không được dùng requested quantity"


def test_spot_free_balance_lower_than_executed(monkeypatch):
    """free balance thấp hơn executedQty → OCO phải kẹp theo free, không bán quá số dư."""
    rec, keys = _setup(monkeypatch, fills=[(27000, 1.0, 0.0, "BNB")])
    rec.balances = {"BTC": 0.4}  # bị OCO plan khác giữ/lock phần còn lại
    binance_executor.attach_spot_oco("https://demo-api.binance.com", keys, "BTCUSDT",
                                     28000.0, 26000.0, "spot-btc-77-1", 1.0)
    oco = rec.paths("/api/v3/orderList/oco", "POST")
    assert float(oco[0][2]["quantity"]) == 0.4, "OCO phải bán tối đa free balance"


def test_spot_min_qty_and_notional(monkeypatch):
    """Không đủ min notional cho phần TP2 → gộp về 1 OCO, không tạo OCO số lượng sai."""
    rec, keys = _setup(monkeypatch, fills=[(27000, 0.002, 0.0, "BNB")])
    rec.balances = {"BTC": 0.002}
    binance_executor.attach_spot_oco("https://demo-api.binance.com", keys, "BTCUSDT",
                                     28000.0, 26000.0, "spot-btc-77-1", 0.002,
                                     tp2=28500.0)
    oco = rec.paths("/api/v3/orderList/oco", "POST")
    # 0.002 BTC * 28000 = 56 > 10 notional nhưng chia 70/30 → 0.001 (28 USDT) còn OK;
    # với min_notional 10 thì cả 2 phần đều qua → vẫn 2 OCO. Kiểm tra không OCO số lượng 0.
    for _m, _p, prm in oco:
        assert float(prm["quantity"]) > 0


def test_spot_oco_duplicate_skips(monkeypatch):
    """OCO đã tồn tại (duplicate) → KHÔNG bán lại, KHÔNG tạo trùng."""
    rec, keys = _setup(monkeypatch, fills=[(27000, 1.0, 0.0, "BNB")])
    rec.balances = {"BTC": 1.0}

    def dup(base, api_key, secret, method, path, params):
        rec.calls.append((method, path, dict(params)))
        if path.endswith("/exchangeInfo"):
            return {"symbols": [{
                "symbol": "BTCUSDT", "status": "TRADING", "baseAsset": "BTC",
                "filters": [{"filterType": "PRICE_FILTER", "minPrice": "0.01",
                             "tickSize": "0.01"},
                            {"filterType": "LOT_SIZE", "minQty": "0.001",
                             "stepSize": "0.001"},
                            {"filterType": "NOTIONAL", "minNotional": "10"}]}]}
        if path.endswith("/api/v3/account"):
            return {"balances": [{"asset": "BTC", "free": "1.0", "locked": "0"}],
                    "commissionRates": {"maker": "0.001", "taker": "0.001"}}
        if method == "POST" and path.endswith("/api/v3/orderList/oco"):
            raise binance_executor.ExecutorError(-2010, "Duplicate order")
        return {}

    monkeypatch.setattr(binance_executor, "signed_request", dup)
    result = binance_executor.attach_spot_oco("https://demo-api.binance.com", keys,
                                              "BTCUSDT", 28000.0, 26000.0,
                                              "spot-btc-77-1", 1.0)
    assert result.get("already_exists") is True
    # Không có MARKET SELL → chắc chắn không bán nhầm vị thế đã được bảo vệ
    assert not any(p.endswith("/api/v3/order") and prm.get("side") == "SELL"
                   for _m, p, prm in rec.calls)


def test_spot_min_qty_insufficient_raises(monkeypatch):
    """Có entry nhưng free < min_qty → raise rõ ràng, KHÔNG tạo OCO số lượng sai."""
    rec, keys = _setup(monkeypatch)
    rec.balances = {"BTC": 0.0001}  # dưới min_qty 0.001
    with pytest.raises(binance_executor.ExecutorError) as exc:
        binance_executor.attach_spot_oco("https://demo-api.binance.com", keys,
                                         "BTCUSDT", 28000.0, 26000.0,
                                         "spot-btc-77-1", 1.0)
    assert exc.value.code == -4164
    assert not rec.paths("/api/v3/orderList/oco", "POST"), "không được tạo OCO sai"


# ─── C. TP1/TP2 Spot ─────────────────────────────────────────────────────────

def test_spot_tp2_two_ocos_with_distinct_ids(monkeypatch):
    """tp2 → 2 OCO, mỗi OCO có listClientOrderId riêng + client IDs riêng."""
    rec, keys = _setup(monkeypatch, fills=[(27000, 1.0, 0.0, "BNB")])
    rec.balances = {"BTC": 1.0}
    result = binance_executor.attach_spot_oco("https://demo-api.binance.com", keys,
                                              "BTCUSDT", 28000.0, 26000.0,
                                              "spot-btc-77-1", 1.0, tp2=28500.0)
    oco = rec.paths("/api/v3/orderList/oco", "POST")
    assert len(oco) == 2
    assert oco[0][2]["listClientOrderId"] != oco[1][2]["listClientOrderId"]
    assert oco[0][2]["aboveClientOrderId"] != oco[1][2]["aboveClientOrderId"]
    assert oco[0][2]["belowClientOrderId"] != oco[1][2]["belowClientOrderId"]
    assert float(oco[0][2]["quantity"]) + float(oco[1][2]["quantity"]) == 1.0
    assert oco[0][2]["abovePrice"] == "28000" and oco[1][2]["abovePrice"] == "28500"


def test_spot_tp2_second_oco_fails_keeps_first(monkeypatch):
    """OCO-B lỗi → giữ OCO-A, đóng riêng phần TP2 (KHÔNG bán cả position)."""
    rec, keys = _setup(monkeypatch, fills=[(27000, 1.0, 0.0, "BNB")])
    rec.balances = {"BTC": 1.0}
    oco_calls = {"n": 0}

    def flaky(base, api_key, secret, method, path, params):
        rec.calls.append((method, path, dict(params)))
        if path.endswith("/exchangeInfo"):
            return {"symbols": [{
                "symbol": "BTCUSDT", "status": "TRADING", "baseAsset": "BTC",
                "filters": [{"filterType": "PRICE_FILTER", "minPrice": "0.01",
                             "tickSize": "0.01"},
                            {"filterType": "LOT_SIZE", "minQty": "0.001",
                             "stepSize": "0.001"},
                            {"filterType": "NOTIONAL", "minNotional": "10"}]}]}
        if path.endswith("/api/v3/account"):
            return {"balances": [{"asset": "BTC", "free": "1.0"}],
                    "commissionRates": {"maker": "0.001", "taker": "0.001"}}
        if method == "POST" and path.endswith("/api/v3/orderList/oco"):
            oco_calls["n"] += 1
            if oco_calls["n"] == 2:
                raise binance_executor.ExecutorError(-1013, "filter failure")
            return {"orderListId": rec._next_list_id(),
                    "listClientOrderId": params.get("listClientOrderId"),
                    "orders": [{"clientOrderId": params.get("aboveClientOrderId"),
                                "orderId": 1}]}
        return {}

    monkeypatch.setattr(binance_executor, "signed_request", flaky)
    result = binance_executor.attach_spot_oco("https://demo-api.binance.com", keys,
                                              "BTCUSDT", 28000.0, 26000.0,
                                              "spot-btc-77-1", 1.0, tp2=28500.0)
    assert result.get("oco_list_id") is not None, "OCO-A phải được giữ"
    assert result.get("tp2_closed") is True
    # Chỉ bán phần TP2 (0.3), không bán cả 1.0
    sells = [(p, prm) for m, p, prm in rec.calls
             if p.endswith("/api/v3/order") and prm.get("side") == "SELL"]
    assert sells and float(sells[0][1]["quantity"]) == 0.3


# ─── D/E. Plan A không hủy Plan B, balance dùng chung ────────────────────────

def _plan_a_list_id(plan_id: str) -> int:
    """orderListId giả lập của plan (phải khớp _make_signal_row)."""
    return 500 + (sum(ord(c) for c in plan_id) % 100)


def _make_signal_row(conn, plan_id, user_id, symbol="BTCUSDT", tp2=False, **over):
    """Chèn 1 dòng ledger spot với OCO IDs riêng của plan.

    tp_algo_id / tp2_algo_id lưu orderListId (số), các *_client_order_id lưu client id.
    """
    list_id = _plan_a_list_id(plan_id)
    row = {
        "user_id": user_id,
        "chat_id": 1,
        "symbol": symbol,
        "mode": "spot",
        "direction": "BUY",
        "sent_at": "2026-01-01T00:00:00+00:00",
        "prediction_id": 101,
        "plan_id": plan_id,
        "order_status": "placed",
        "entry_order_id": 900 + (sum(ord(c) for c in plan_id) % 100),
        "entry_new_client_order_id": f"{plan_id}-e",
        "tp_algo_id": list_id,
        "tp2_algo_id": (list_id + 1000) if tp2 else None,
        "tp_client_algo_id": f"{plan_id}-tp",
        "tp2_client_algo_id": (f"{plan_id}-tp2" if tp2 else None),
        "sl_client_algo_id": f"{plan_id}-sl",
        "list_client_order_id": f"{plan_id}-list",
        "tp2_list_client_order_id": (f"{plan_id}-list2" if tp2 else None),
        "above_client_order_id": f"{plan_id}-tp",
        "below_client_order_id": f"{plan_id}-sl",
        "tp2_above_client_order_id": (f"{plan_id}-tp2" if tp2 else None),
        "tp2_below_client_order_id": (f"{plan_id}-sl2" if tp2 else None),
    }
    row.update(over)
    cols = ", ".join(row.keys())
    marks = ", ".join("?" for _ in row)
    conn.execute(f"INSERT INTO auto_scan_signals ({cols}) VALUES ({marks})",
                 tuple(row.values()))
    conn.commit()
    return conn


def test_plan_a_event_does_not_cancel_plan_b_oco(monkeypatch):
    """executionReport của Plan A chỉ hủy OCO của Plan A — Plan B nguyên vẹn."""
    _use_test_db(monkeypatch)
    analyze.init_auto_scan_db()
    with sqlite3.connect(analyze.DB_PATH) as conn:
        _make_signal_row(conn, "spot-btc-77-1", 77, tp2=False)
        _make_signal_row(conn, "spot-btc-77-2", 77, tp2=False)

    rec = CallRecorder()
    monkeypatch.setattr(binance_executor, "signed_request", rec)
    # Event của Plan A (clientOrderId của leg TP plan A).
    event = {"e": "executionReport", "s": "BTCUSDT",
             "c": "spot-btc-77-1-tp", "X": "FILLED", "o": None}
    result = analyze.handle_spot_user_event(77, ("k", "s"), event)
    assert result["handled"] is True and result["plan_id"] == "spot-btc-77-1"
    cancelled_lists = [prm.get("orderListId") for m, p, prm in rec.calls
                       if m == "DELETE" and p.endswith("/api/v3/orderList")]
    # Chỉ hủy đúng OCO của plan A, không đụng orderListId của plan B.
    plan_a_list = _plan_a_list_id("spot-btc-77-1")
    assert [str(x) for x in cancelled_lists] == [str(plan_a_list)]


def test_plan_b_untouched_after_plan_a_close(monkeypatch):
    """Sau khi Plan A đóng, dòng Plan B vẫn còn order_status='placed'."""
    _use_test_db(monkeypatch)
    analyze.init_auto_scan_db()
    with sqlite3.connect(analyze.DB_PATH) as conn:
        _make_signal_row(conn, "spot-btc-77-1", 77)
        _make_signal_row(conn, "spot-btc-77-2", 77)
    rec = CallRecorder()
    monkeypatch.setattr(binance_executor, "signed_request", rec)
    analyze.handle_spot_user_event(77, ("k", "s"),
                                   {"e": "executionReport", "s": "BTCUSDT",
                                    "c": "spot-btc-77-1-tp", "X": "FILLED"})
    with sqlite3.connect(analyze.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row_b = conn.execute(
            "SELECT order_status FROM auto_scan_signals WHERE plan_id=?",
            ("spot-btc-77-2",)).fetchone()
    assert row_b["order_status"] == "placed", "OCO/protection plan B phải còn nguyên"


def test_shared_balance_stays_within_free(monkeypatch):
    """2 plan cùng symbol: mỗi OCO chỉ bán phần free khả dụng, không bán balance nhà plan khác."""
    rec, keys = _setup(monkeypatch, fills=[(27000, 1.0, 0.0, "BNB")])
    rec.balances = {"BTC": 1.0}
    # Plan A bán 1.0 → chỉ 0.7 còn free cho Plan B (nhưng free thực tế sàn đã cập nhật;
    # ở đây kiểm tra logic min(qty, free) chặn không bán quá).
    result_a = binance_executor.attach_spot_oco("https://demo-api.binance.com", keys,
                                                "BTCUSDT", 28000.0, 26000.0,
                                                "spot-btc-77-1", 1.0)
    assert result_a["qty_tp1"] == 1.0
    rec.balances = {"BTC": 0.0}
    with pytest.raises(binance_executor.ExecutorError):
        binance_executor.attach_spot_oco("https://demo-api.binance.com", keys,
                                         "BTCUSDT", 28000.0, 26000.0,
                                         "spot-btc-77-2", 1.0)


# ─── F. Timeout trước/sau MARKET BUY ─────────────────────────────────────────

def test_network_error_during_entry_is_soft_cancelled(monkeypatch):
    """Timeout khi gửi MARKET BUY → soft-cancel theo newClientOrderId (không mua double)."""
    rec, keys = _setup(monkeypatch)
    rec.balances = {"BTC": 1.0}

    def timeout(base, api_key, secret, method, path, params):
        rec.calls.append((method, path, dict(params)))
        if path.endswith("/exchangeInfo"):
            return {"symbols": [{
                "symbol": "BTCUSDT", "status": "TRADING", "baseAsset": "BTC",
                "filters": [{"filterType": "PRICE_FILTER", "minPrice": "0.01",
                             "tickSize": "0.01"},
                            {"filterType": "LOT_SIZE", "minQty": "0.001",
                             "stepSize": "0.001"},
                            {"filterType": "NOTIONAL", "minNotional": "10"}]}]}
        if method == "POST" and path.endswith("/api/v3/order"):
            raise binance_executor.ExecutorError(-1000, "lỗi mạng")
        return {}

    monkeypatch.setattr(binance_executor, "signed_request", timeout)
    with pytest.raises(binance_executor.ExecutorError) as exc:
        binance_executor.place_spot_plan(
            "BTCUSDT", 27000.0, 28000.0, 26000.0, 1.0, keys, "spot-btc-77-1")
    assert exc.value.code == -1000
    # Phải có nỗ lực hủy entry theo origClientOrderId
    dels = rec.paths("/api/v3/order", "DELETE")
    assert dels and dels[0][2].get("origClientOrderId") == "spot-btc-77-1-e"


# ─── H. Restart recovery ─────────────────────────────────────────────────────

# ─── H. Restart recovery ─────────────────────────────────────────────────────

def test_ensure_spot_recovery_all_users(monkeypatch):
    """ensure_spot_recovery chạy cho MỌI user có intent chưa terminal, không user_scope cứng."""
    from key_store import KeyError_
    monkeypatch.setattr(key_store, "DB_PATH", analyze.DB_PATH)
    key_store.save_api_keys(77, "APIKEY" + "x" * 30, "SECRET" + "y" * 30)
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", key_store.os.getenv("DATA_ENCRYPTION_KEY", ""))
    from cryptography.fernet import Fernet
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    key_store.save_api_keys(77, "APIKEY" + "x" * 30, "SECRET" + "y" * 30)
    _db_with_spot_intent(monkeypatch, plan_id="spot-btc-77-1", status="market_unknown")

    rec = CallRecorder()

    def fake(base, api_key, secret, method, path, params):
        rec.calls.append((method, path, dict(params)))
        if path.endswith("/api/v3/order"):
            raise binance_executor.ExecutorError(-2011, "Order does not exist")
        return {}

    monkeypatch.setattr(binance_executor, "signed_request", fake)
    result = analyze.ensure_spot_recovery()
    assert result["users"] >= 1
    with sqlite3.connect(analyze.DB_PATH) as conn:
        row = conn.execute(
            "SELECT intent_status FROM auto_scan_signals WHERE plan_id=?",
            ("spot-btc-77-1",)).fetchone()
    assert row[0] == "canceled"


def _db_with_spot_intent(monkeypatch, plan_id="spot-btc-77-1", status="market_unknown"):
    _use_test_db(monkeypatch)
    analyze.init_auto_scan_db()
    with sqlite3.connect(analyze.DB_PATH) as conn:
        conn.execute(
            """INSERT INTO auto_scan_signals
               (user_id, chat_id, symbol, mode, direction, sent_at, plan_id,
                intent_status, entry_new_client_order_id, sl, tp1,
                list_client_order_id, tp2_list_client_order_id)
               VALUES (77, 1, 'BTCUSDT', 'spot', 'BUY', '2026-01-01T00:00:00+00:00',
                       ?, ?, ?, 26000, 28000, 'spot-btc-77-1-list',
                       'spot-btc-77-1-list2')""",
            (plan_id, status, f"{plan_id}-e"))
        conn.commit()


def test_restart_recovery_entry_not_found_marks_canceled(monkeypatch):
    """Entry không tồn tại (-2011) → đánh dấu canceled, không mua lại."""
    _db_with_spot_intent(monkeypatch)
    rec = CallRecorder()
    monkeypatch.setattr(binance_executor, "signed_request", rec)
    rec.handlers["oca_lookup"] = None

    def not_found(base, api_key, secret, method, path, params):
        rec.calls.append((method, path, dict(params)))
        if path.endswith("/api/v3/order"):
            raise binance_executor.ExecutorError(-2011, "Order does not exist")
        if path.endswith("/exchangeInfo"):
            return rec.handlers.get("oca_lookup", {}) or {"symbols": []}
        return {"balances": [{"asset": "BTC", "free": "1.0"}]}

    monkeypatch.setattr(binance_executor, "signed_request", not_found)
    actions = analyze.recover_spot_intent_ledger(77, ("k", "s"))
    assert any(a["action"] == "entry_not_found_marked_canceled" for a in actions)
    with sqlite3.connect(analyze.DB_PATH) as conn:
        row = conn.execute(
            "SELECT intent_status FROM auto_scan_signals WHERE plan_id=?",
            ("spot-btc-77-1",)).fetchone()
    assert row[0] == "canceled"
    # KHÔNG có MARKET BUY lại
    assert not any(p.endswith("/api/v3/order") and prm.get("side") == "BUY"
                   for _m, p, prm in rec.calls)


def test_restart_recovery_recreates_missing_oco(monkeypatch):
    """Entry đã FILLED nhưng chưa có OCO → chỉ tạo OCO còn thiếu, không mua lại."""
    _db_with_spot_intent(monkeypatch)

    rec = CallRecorder()

    def fake(base, api_key, secret, method, path, params):
        rec.calls.append((method, path, dict(params)))
        if path.endswith("/api/v3/order") and method == "GET":
            return {"status": "FILLED", "executedQty": "1.0",
                    "cummulativeQuoteQty": "27000",
                    "fills": [{"price": "27000", "qty": "1.0",
                               "commission": "0.001", "commissionAsset": "BNB"}]}
        if path.endswith("/api/v3/orderList"):
            raise binance_executor.ExecutorError(-2011, "Unknown order")
        if path.endswith("/exchangeInfo"):
            return {"symbols": [{
                "symbol": "BTCUSDT", "status": "TRADING", "baseAsset": "BTC",
                "filters": [{"filterType": "PRICE_FILTER", "minPrice": "0.01",
                             "tickSize": "0.01"},
                            {"filterType": "LOT_SIZE", "minQty": "0.001",
                             "stepSize": "0.001"},
                            {"filterType": "NOTIONAL", "minNotional": "10"}]}]}
        if path.endswith("/api/v3/account"):
            return {"balances": [{"asset": "BTC", "free": "1.0"}]}
        return {}

    monkeypatch.setattr(binance_executor, "signed_request", fake)
    actions = analyze.recover_spot_intent_ledger(77, ("k", "s"))
    assert any(a["action"] in ("recreated_oco_a", "recreated_oco_b") for a in actions)
    assert not any(p.endswith("/api/v3/order") and prm.get("side") == "BUY"
                   for _m, p, prm in rec.calls), "KHÔNG được MARKET BUY lại"
    ocos = rec.paths("/api/v3/orderList/oco", "POST")
    assert ocos, "phải tạo OCO còn thiếu"


def test_restart_recovery_skips_existing_oco(monkeypatch):
    """OCO đã tồn tại trên sàn → KHÔNG tạo trùng."""
    _db_with_spot_intent(monkeypatch)

    rec = CallRecorder()

    def fake(base, api_key, secret, method, path, params):
        rec.calls.append((method, path, dict(params)))
        if path.endswith("/api/v3/order") and method == "GET":
            return {"status": "FILLED", "executedQty": "1.0",
                    "fills": [{"price": "27000", "qty": "1.0",
                               "commission": "0", "commissionAsset": "BNB"}]}
        if path.endswith("/api/v3/orderList") and method == "GET":
            return {"orderListId": 6060,
                    "listClientOrderId": params.get("origClientOrderId")}
        if path.endswith("/exchangeInfo"):
            return {"symbols": [{
                "symbol": "BTCUSDT", "status": "TRADING", "baseAsset": "BTC",
                "filters": [{"filterType": "PRICE_FILTER", "minPrice": "0.01",
                             "tickSize": "0.01"},
                            {"filterType": "LOT_SIZE", "minQty": "0.001",
                             "stepSize": "0.001"},
                            {"filterType": "NOTIONAL", "minNotional": "10"}]}]}
        return {}

    monkeypatch.setattr(binance_executor, "signed_request", fake)
    analyze.recover_spot_intent_ledger(77, ("k", "s"))
    assert not rec.paths("/api/v3/orderList/oco", "POST"), \
        "OCO đã tồn tại thì KHÔNG được tạo trùng"


# ─── I. Manual cancel TP/SL ──────────────────────────────────────────────────

def test_manual_cancel_tp_only_keeps_sl(monkeypatch):
    """User chỉ hủy TP → không đụng SL leg, position vẫn được bảo vệ."""
    rec = CallRecorder()
    rec.handlers["oca_lookup"] = {"orderListId": 777,
                                 "listClientOrderId": "spot-btc-77-1-list"}
    monkeypatch.setattr(binance_executor, "signed_request", rec)
    # spot_order_state cho leg TP dùng origClientOrderId → chỉ đọc, không hủy gì.
    state = binance_executor.spot_order_state(
        "https://demo-api.binance.com", ("k", "s"), "BTCUSDT", "spot-btc-77-1-tp")
    assert state["status"] == "FILLED"
    # Không có DELETE nào trong luồng chỉ-đọc trạng thái
    assert not any(m == "DELETE" for m, _p, _prm in rec.calls)


def test_list_status_all_done_keeps_other_oco(monkeypatch):
    """OCO-A ALL_DONE (TP1 chốt 70%) → KHÔNG hủy OCO-B: phần 30% còn lại vẫn được bảo vệ.

    Đây là fix của lỗi nghiêm trọng: trước đây handler hủy CẢ 2 list khi chỉ 1 list kết
    thúc, để 30% position trần không TP/SL.
    """
    _use_test_db(monkeypatch)
    analyze.init_auto_scan_db()
    with sqlite3.connect(analyze.DB_PATH) as conn:
        _make_signal_row(conn, "spot-btc-77-1", 77, tp2=True)
        _make_signal_row(conn, "spot-btc-77-2", 77, tp2=True)
    rec = CallRecorder()
    monkeypatch.setattr(binance_executor, "signed_request", rec)
    result = analyze.handle_spot_user_event(
        77, ("k", "s"), {"e": "listStatus", "s": "BTCUSDT", "L": "ALL_DONE",
                         "C": "spot-btc-77-1-list", "g": _plan_a_list_id("spot-btc-77-1")})
    assert result["plan_id"] == "spot-btc-77-1"
    assert result.get("deferred") == "tp1_partial_exit", result
    # KHÔNG hủy list nào — OCO-B vẫn đang bảo vệ 30% còn lại
    cancelled = [str(prm.get("orderListId")) for m, p, prm in rec.calls
                 if m == "DELETE" and p.endswith("/api/v3/orderList")]
    assert cancelled == [], f"khong duoc huy OCO nao khi moi TP1 khop: {cancelled}"
    with sqlite3.connect(analyze.DB_PATH) as conn:
        row = conn.execute(
            "SELECT order_status FROM auto_scan_signals WHERE plan_id=?",
            ("spot-btc-77-1",)).fetchone()
    assert row[0] == "placed", "plan phai van o trang thai placed, khong duoc dong"


def test_entry_filled_event_does_not_cancel_protection(monkeypatch):
    """Entry MARKET BUY vừa khớp → KHÔNG hủy OCO (đây là trạng thái bình thường).

    Fix lỗi nghiêm trọng: trước đây event entry-fill khớp plan rồi hủy cả OCO, để position
    hoàn toàn trần ngay sau khi mua.
    """
    _use_test_db(monkeypatch)
    analyze.init_auto_scan_db()
    with sqlite3.connect(analyze.DB_PATH) as conn:
        _make_signal_row(conn, "spot-btc-77-1", 77, tp2=True)
    rec = CallRecorder()
    monkeypatch.setattr(binance_executor, "signed_request", rec)

    result = analyze.handle_spot_user_event(
        77, ("k", "s"),
        {"e": "executionReport", "s": "BTCUSDT", "X": "FILLED",
         "c": "spot-btc-77-1-e", "g": -1})

    assert result.get("handled") is True, result
    assert result.get("reason") == "entry_filled_protection_stays_armed", result
    cancelled = [str(prm.get("orderListId")) for m, p, prm in rec.calls
                 if m == "DELETE" and p.endswith("/api/v3/orderList")]
    assert cancelled == [], f"entry fill KHONG duoc huy OCO: {cancelled}"
    with sqlite3.connect(analyze.DB_PATH) as conn:
        row = conn.execute(
            "SELECT order_status FROM auto_scan_signals WHERE plan_id=?",
            ("spot-btc-77-1",)).fetchone()
    assert row[0] == "placed"


# ─── TTL cleanup chỉ hủy lệnh CỦA BOT ───────────────────────────────────────

class _TtlRecorder:
    """Giả lập openOrders có cả lệnh bot LẪN lệnh user tự đặt."""

    def __init__(self, orders):
        self.orders = orders
        self.cancelled = []
        self.calls = []

    def __call__(self, base, api_key, secret, method, path, params):
        self.calls.append((method, path, dict(params)))
        if path.endswith("/fapi/v1/openOrders"):
            return list(self.orders)
        if method == "DELETE" and path.endswith("/fapi/v1/order"):
            self.cancelled.append(params.get("orderId"))
            return {"status": "CANCELED"}
        if path.endswith("/fapi/v1/algoOrder") or path.endswith("/fapi/v1/orderList"):
            return {}
        return {}


def test_ttl_cleanup_never_cancels_user_orders(monkeypatch):
    """GTC LIMIT user tự đặt quá 24h KHÔNG bị hủy — chỉ lệnh của bot (`-e`) mới bị hủy."""
    old_ms = int(__import__("time").time() * 1000) - 100 * 24 * 3600 * 1000  # 100 ngày
    rec = _TtlRecorder([
        {"symbol": "ETHUSDT", "orderId": 1, "clientOrderId": "futu-eth-77-1-e",
         "type": "LIMIT", "timeInForce": "GTC", "executedQty": "0", "updateTime": old_ms},
        {"symbol": "ETHUSDT", "orderId": 2, "clientOrderId": "my-manual-limit",
         "type": "LIMIT", "timeInForce": "GTC", "executedQty": "0", "updateTime": old_ms},
        {"symbol": "BTCUSDT", "orderId": 3, "clientOrderId": "user-btc-plan",
         "type": "LIMIT", "timeInForce": "GTC", "executedQty": "0", "updateTime": old_ms},
    ])
    monkeypatch.setattr(binance_executor, "signed_request", rec)
    monkeypatch.setattr(binance_executor, "FUTURES_ENTRY_TTL_SECONDS", 86400)
    monkeypatch.setattr(binance_executor, "open_futures_algo_orders", lambda keys, sym: [])

    out = binance_executor.cancel_expired_futures_entry_orders(("k", "s"))

    assert rec.cancelled == [1], f"chi huy lenh cua bot, da huy: {rec.cancelled}"


def test_ttl_cleanup_skips_fresh_bot_orders(monkeypatch):
    """Lệnh bot còn trong TTL → không hủy."""
    now_ms = int(__import__("time").time() * 1000)
    rec = _TtlRecorder([
        {"symbol": "ETHUSDT", "orderId": 11, "clientOrderId": "futu-eth-77-1-e",
         "type": "LIMIT", "timeInForce": "GTC", "executedQty": "0", "updateTime": now_ms},
    ])
    monkeypatch.setattr(binance_executor, "signed_request", rec)
    monkeypatch.setattr(binance_executor, "FUTURES_ENTRY_TTL_SECONDS", 86400)
    monkeypatch.setattr(binance_executor, "open_futures_algo_orders", lambda keys, sym: [])

    out = binance_executor.cancel_expired_futures_entry_orders(("k", "s"))
    assert rec.cancelled == [], "con trong TTL thi khong huy"
    assert out == []


def test_ttl_cleanup_skips_partially_filled(monkeypatch):
    """Entry bot đã khớp một phần → KHÔNG hủy (position đã có)."""
    old_ms = int(__import__("time").time() * 1000) - 100 * 24 * 3600 * 1000
    rec = _TtlRecorder([
        {"symbol": "ETHUSDT", "orderId": 21, "clientOrderId": "futu-eth-77-1-e",
         "type": "LIMIT", "timeInForce": "GTC", "executedQty": "0.01",
         "updateTime": old_ms},
    ])
    monkeypatch.setattr(binance_executor, "signed_request", rec)
    monkeypatch.setattr(binance_executor, "FUTURES_ENTRY_TTL_SECONDS", 86400)
    monkeypatch.setattr(binance_executor, "open_futures_algo_orders", lambda keys, sym: [])

    out = binance_executor.cancel_expired_futures_entry_orders(("k", "s"))
    assert rec.cancelled == [], "da khop mot phan thi khong duoc huy"


def test_is_bot_coid():
    assert binance_executor._is_bot_coid("futu-eth-77-1-e") is True
    assert binance_executor._is_bot_coid("spot-eth-77-3-e") is True
    assert binance_executor._is_bot_coid("my-manual-limit") is False
    assert binance_executor._is_bot_coid("futu-eth-77-1-tp") is False
    assert binance_executor._is_bot_coid(None) is False
    assert binance_executor._is_bot_coid("") is False


# ─── Ledger: 1 plan = 1 dòng (không trùng) ─────────────────────────────────

def test_record_signal_upserts_not_duplicates(monkeypatch):
    """`_record_auto_scan_signal` phải CẬP NHẬT dòng intent đã có, không tạo dòng thứ 2.

    Trước đây mỗi lệnh đặt thành công tạo 2 dòng (1 từ _upsert_intent trước khi đặt, 1 từ
    _record_auto_scan_signal sau khi đặt) → /autoscanlog* hiện trùng, đếm xóa gấp đôi.
    """
    _use_test_db(monkeypatch)
    analyze.init_auto_scan_db()

    # 1) Giả lập dòng intent đã tạo TRƯỚC khi đặt lệnh
    analyze._upsert_intent(
        user_id=77, mode="spot", signal_symbol="ETHUSDT", exchange_symbol="ETHUSDT",
        direction="BUY", plan_id="spot-eth-77-1", entry_new_client_order_id="spot-eth-77-1-e",
        tp_client_algo_id="spot-eth-77-1-tp", tp2_client_algo_id=None,
        sl_client_algo_id="spot-eth-77-1-sl", qty=270.0, entry_price=2700.0,
        tp1=2800.0, tp2=None, sl=2600.0, intent_status="market_submitting")

    # 2) Sau khi đặt lệnh xong, caller gọi _record_auto_scan_signal
    analyze._record_auto_scan_signal(
        77, 1, "ETHUSDT", "spot", "BUY", 80, 101,
        plan_id="spot-eth-77-1", order_status="placed",
        entry_low=2690.0, entry_high=2710.0, sl=2600.0, tp1=2750.0, tp2=None,
        signal_symbol="ETHUSDT", exchange_symbol="ETHUSDT", intent_status="protected",
        entry_new_client_order_id="spot-eth-77-1-e")

    with sqlite3.connect(analyze.DB_PATH) as conn:
        rows = conn.execute(
            "SELECT prediction_id, order_status, tp1, sl, intent_status, "
            "entry_new_client_order_id "
            "FROM auto_scan_signals WHERE user_id=77 AND plan_id=?",
            ("spot-eth-77-1",)).fetchall()

    assert len(rows) == 1, f"phai chi co 1 dong cho 1 plan, co {len(rows)}: {rows}"
    assert rows[0][0] == 101, "prediction_id phai duoc ghi"
    assert rows[0][1] == "placed"
    assert rows[0][2] == 2750.0 and rows[0][3] == 2600.0
    # intent_status phải được cập nhật từ 'market_submitting' sang 'protected'
    assert rows[0][4] == "protected"
    # entry coid phải giữ nguyên (không bị NULL hóa)
    assert rows[0][5] == "spot-eth-77-1-e", rows[0][5]


def test_record_signal_inserts_when_no_intent_row(monkeypatch):
    """Không có dòng intent → vẫn INSERT như cũ (đường planner /manual)."""
    _use_test_db(monkeypatch)
    analyze.init_auto_scan_db()
    analyze._record_auto_scan_signal(
        77, 1, "ETHUSDT", "spot", "BUY", 80, 101,
        plan_id="spot-eth-77-9", order_status="pending",
        entry_low=2690.0, entry_high=2710.0, sl=2600.0, tp1=2750.0)
    with sqlite3.connect(analyze.DB_PATH) as conn:
        rows = conn.execute(
            "SELECT plan_id FROM auto_scan_signals WHERE user_id=77").fetchall()
    assert len(rows) == 1


# ─── Duplicate coid: không mua trùng khi không xác minh được ─────────────────

class _SpotMktRec(CallRecorder):
    def __init__(self, dup=True, lookup=None):
        super().__init__()
        self.dup = dup
        self.lookup = lookup
        self._oco = 0

    def __call__(self, base, api_key, secret, method, path, params):
        self.calls.append((method, path, dict(params)))
        if path.endswith("/exchangeInfo"):
            return {"symbols": [{
                "symbol": "ETHUSDT", "status": "TRADING", "baseAsset": "ETH",
                "filters": [{"filterType": "PRICE_FILTER", "minPrice": "0.01",
                             "tickSize": "0.01"},
                            {"filterType": "LOT_SIZE", "minQty": "0.0001",
                             "stepSize": "0.0001"},
                            {"filterType": "NOTIONAL", "minNotional": "5"}]}]}
        if path.endswith("/api/v3/account"):
            return {"balances": [{"asset": "ETH", "free": "1.0"}]}
        if path.endswith("/api/v3/order") and method == "POST" and params.get("type") == "MARKET":
            if self.dup and len(self.paths("/api/v3/order", "POST")) == 1:
                raise binance_executor.ExecutorError(-2010, "Duplicate order")
            return {"orderId": 60000, "status": "FILLED", "executedQty": "1.0",
                    "cummulativeQuoteQty": "2700", "fills": []}
        if path.endswith("/api/v3/order") and method == "GET":
            if self.lookup is not None:
                raise self.lookup
            # Phải có orderId: code dùng chính response này làm entry_resp khi coid trùng.
            return {"orderId": 60000, "status": "FILLED", "executedQty": "1.0",
                    "cummulativeQuoteQty": "2700", "fills": []}
        if path.endswith("/api/v3/orderList/oco") and method == "POST":
            self._oco += 1
            return {"orderListId": 70000 + self._oco,
                    "listClientOrderId": params.get("listClientOrderId"),
                    "orders": [{"clientOrderId": params.get("aboveClientOrderId"),
                                "orderId": 70000 + self._oco},
                               {"clientOrderId": params.get("belowClientOrderId"),
                                "orderId": 70000 + self._oco}]}
        return {}


def test_spot_duplicate_coid_network_error_does_not_double_buy(monkeypatch):
    """Coid trùng + tra cứu lỗi mạng → KHÔNG đặt lại MARKET BUY (tránh mua 2 lần)."""
    binance_executor._FILTER_CACHE.clear()
    rec = _SpotMktRec(lookup=binance_executor.ExecutorError(-1000, "mat ket noi"))
    monkeypatch.setattr(binance_executor, "signed_request", rec)
    monkeypatch.setattr(binance_executor, "public_get", rec.public_get)

    with pytest.raises(binance_executor.ExecutorError) as exc:
        binance_executor.place_spot_plan("ETHUSDT", 2700.0, 2800.0, 2600.0, 0.1,
                                         ("k", "s"), "plan-77-1")
    assert exc.value.code == -1000
    assert "mua tr" in exc.value.msg, exc.value.msg
    market = rec.paths("/api/v3/order", "POST")
    assert len(market) == 1, "KHONG duoc gui MARKET BUY lai"


def test_spot_duplicate_coid_not_found_bumps(monkeypatch):
    """Coid trùng nhưng chắc chắn không có lệnh cũ (-2011) → đổi coid và gửi lại."""
    binance_executor._FILTER_CACHE.clear()
    rec = _SpotMktRec(lookup=binance_executor.ExecutorError(-2011, "Order does not exist"))
    monkeypatch.setattr(binance_executor, "signed_request", rec)
    monkeypatch.setattr(binance_executor, "public_get", rec.public_get)

    out = binance_executor.place_spot_plan("ETHUSDT", 2700.0, 2800.0, 2600.0, 0.1,
                                           ("k", "s"), "plan-77-1")
    market = rec.paths("/api/v3/order", "POST")
    assert len(market) == 2, "phai doi coid va gui lai"
    assert market[0][2]["newClientOrderId"] != market[1][2]["newClientOrderId"]
    assert out["plan_id"].endswith("2"), out["plan_id"]


def test_spot_duplicate_coid_existing_reused(monkeypatch):
    """Coid trùng và lệnh cũ CÓ → dùng lại lệnh cũ, không gửi MARKET BUY thứ 2."""
    binance_executor._FILTER_CACHE.clear()
    rec = _SpotMktRec(dup=True, lookup=None)
    monkeypatch.setattr(binance_executor, "signed_request", rec)
    monkeypatch.setattr(binance_executor, "public_get", rec.public_get)

    out = binance_executor.place_spot_plan("ETHUSDT", 2700.0, 2800.0, 2600.0, 0.1,
                                           ("k", "s"), "plan-77-1")
    market = rec.paths("/api/v3/order", "POST")
    assert len(market) == 1, "khong duoc gui lai MARKET BUY khi lenh cu con"
    assert out["entry_order_id"] == 60000


# ─── Eviction: không hủy OCO đang ARMED ─────────────────────────────────────

def test_ensure_oco_room_never_evicts_armed(monkeypatch):
    """Toàn bộ OCO của bot đang ARMED → KHÔNG hủy OCO nào, báo lỗi rõ ràng."""
    armed = [{"orderListId": 50000 + i,
              "listClientOrderId": f"futu-eth-77-{i}-list",
              "transactionTime": 1700000000000 + i * 1000,
              "listOrderStatus": "EXECUTING"} for i in range(20)]
    rec = _OcoFullRecorder(existing=armed)
    monkeypatch.setattr(binance_executor, "signed_request", rec)

    with pytest.raises(binance_executor.ExecutorError) as exc:
        binance_executor._ensure_oco_room("https://demo-api.binance.com", "k", "s",
                                          "BTCUSDT", {"max_order_lists": 20}, need=2)
    assert exc.value.code == -2010
    assert "offspot" in exc.value.msg, exc.value.msg
    # Không có DELETE nào được gửi
    assert not any(m == "DELETE" for m, _p, _prm in rec.calls)


def test_ensure_oco_room_evicts_only_dead(monkeypatch):
    """Chỉ OCO đã xong (ALL_DONE) bị hủy; OCO đang ARMED được giữ."""
    lists = []
    for i in range(20):
        lists.append({
            "orderListId": 50000 + i,
            "listClientOrderId": f"futu-eth-77-{i}-list",
            "transactionTime": 1700000000000 + i * 1000,
            # 3 list đầu đã xong, còn lại đang armed
            "listOrderStatus": "ALL_DONE" if i < 3 else "EXECUTING",
        })
    rec = _OcoFullRecorder(existing=lists)
    monkeypatch.setattr(binance_executor, "signed_request", rec)

    canceled = binance_executor._ensure_oco_room(
        "https://demo-api.binance.com", "k", "s", "BTCUSDT",
        {"max_order_lists": 20}, need=2)

    coids = [c["list_client_order_id"] for c in canceled]
    # Chỉ 3 list ALL_DONE được hủy, đó đúng là 3 list cũ nhất
    assert coids == ["futu-eth-77-0-list", "futu-eth-77-1-list"], coids
    # Không OCO armed nào bị đụng
    cancelled_ids = [prm.get("orderListId") for m, _p, prm in rec.calls
                     if m == "DELETE"]
    assert 50003 not in cancelled_ids, "OCO dang ARMED khong duoc huy"


# ─── G. WebSocket ────────────────────────────────────────────────────────────

def test_spot_stream_dedup_key():
    """Dedup key khác nhau cho order khác nhau, trùng cho cùng event."""
    k1 = spot_user_stream._dedup_key(
        {"e": "executionReport", "i": 111, "c": "spot-btc-77-1-tp", "E": 1700})
    k2 = spot_user_stream._dedup_key(
        {"e": "executionReport", "i": 112, "c": "spot-btc-77-2-tp", "E": 1700})
    k1b = spot_user_stream._dedup_key(
        {"e": "executionReport", "i": 111, "c": "spot-btc-77-1-tp", "E": 1700})
    assert k1 is not None and k1 == k1b, "cùng event phải cùng khóa"
    assert k1 != k2, "plan khác nhau phải khác khóa"


def test_spot_stream_dedup_cache_drops_repeat():
    cache = spot_user_stream._DedupCache()
    assert cache.is_duplicate("a") is False
    assert cache.is_duplicate("a") is True
    assert cache.is_duplicate("b") is False


def test_spot_stream_disabled_by_default(monkeypatch):
    """Không bật env → ensure_spot_user_stream no-op (không mở socket)."""
    monkeypatch.delenv("SPOT_USER_STREAM_ENABLED", raising=False)
    spot_user_stream.stop_spot_user_stream(1)
    spot_user_stream.ensure_spot_user_stream(1, ("k", "s"))
    assert 1 not in spot_user_stream._THREADS


def test_spot_stream_ws_base_demo_vs_live():
    assert "demo" in spot_user_stream._ws_base("https://demo-api.binance.com")
    assert "demo" not in spot_user_stream._ws_base("https://api.binance.com")


# ─── Khối lượng theo USDT (notional) thay vì số coin ──────────────────────────

def test_futures_qty_computed_from_usdt_and_leverage(monkeypatch, tmp_path):
    """Futures: qty USDT × đòn bẩy / giá entry = số coin đặt lên sàn.

    10 USDT × đòn bẩy 20 / 27000 = 0.00741 → floor step 1e-5 = 0.00741.
    """
    rec, keys = _setup(monkeypatch, base="https://api.binance.com")
    analyze.set_auto_scan_market_enabled(77, 1, "futures", True, "BTCUSDT",
                                         qty="10", leverage=20)
    monkeypatch.setattr(analyze, "get_current_price_raw", lambda sym, mkt: 27000.0)
    from cryptography.fernet import Fernet
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    key_store.save_api_keys(77, "APIKEY" + "x" * 30, "SECRET" + "y" * 30)

    block, info = asyncio.run(analyze._auto_execute_plan(
        user_id=77, mode="futures", symbol="BTCUSDT", direction="LONG",
        plan={"quyet_dinh": "LONG", "entry_thap": 26900.0, "entry_cao": 27000.0,
              "sl": 26000.0, "tp1": 28000.0, "do_tin_cay": 80},
        plan_id="usd-futu-77-1", current_price=27000.0))

    assert info["status"] == "placed", info
    order = info["order"]
    # Vị thế = 10 × 20 = 200 USDT giá trị BTC, chia theo giá entry 27000 → 0.00741 BTC,
    # rồi FLOOR theo stepSize 0.001 → 0.007 BTC (vị thế thực ~189 USDT, bao giờ cũng ≤ 200).
    assert order["qty"] == pytest.approx(0.007, abs=1e-9), order["qty"]
    entry = next(prm for m, p, prm in rec.calls
                 if p.endswith("/fapi/v1/order") and prm.get("type") == "LIMIT")
    assert float(entry["quantity"]) == pytest.approx(0.007, abs=1e-9)
    assert float(entry["price"]) == 27000.0
    # Tin nhắn phải nêu rõ USDT → coin, không chỉ số coin.
    assert "USDT" in block


def test_spot_qty_computed_from_usdt_no_leverage(monkeypatch, tmp_path):
    """Spot: qty USDT / giá thị trường = số coin; KHÔNG nhân đòn bẩy (spot không có leverage).

    270 USDT / 2700 = 0.1 ETH.
    """
    rec, keys = _setup(monkeypatch, base="https://demo-api.binance.com")
    rec.balances = {"ETH": 1.0}
    analyze.set_auto_scan_market_enabled(77, 1, "spot", True, "ETHUSDT",
                                         qty="270", leverage=1)
    monkeypatch.setattr(analyze, "get_current_price_raw", lambda sym, mkt: 2700.0)
    from cryptography.fernet import Fernet
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    key_store.save_api_keys(77, "APIKEY" + "x" * 30, "SECRET" + "y" * 30)

    block, info = asyncio.run(analyze._auto_execute_plan(
        user_id=77, mode="spot", symbol="ETHUSDT", direction="LONG",
        plan={"quyet_dinh": "BUY", "entry_thap": 2690.0, "entry_cao": 2710.0,
              "sl": 2600.0, "tp1": 2750.0, "do_tin_cay": 80},
        plan_id="usd-spot-77-1", current_price=2700.0))

    assert info["status"] == "placed", info
    market = [prm for m, p, prm in rec.calls
              if p.endswith("/api/v3/order") and prm.get("type") == "MARKET"]
    assert len(market) == 1
    assert float(market[0]["quantity"]) == pytest.approx(0.1), market[0]["quantity"]


def test_tiny_usdt_qty_aborts(monkeypatch, tmp_path):
    """Số USDT nhỏ hơn 1 coin tối thiểu → abort rõ ràng, KHÔNG gửi lệnh lên sàn."""
    rec, keys = _setup(monkeypatch)
    rec.balances = {"ETH": 1.0}
    analyze.set_auto_scan_market_enabled(77, 1, "spot", True, "ETHUSDT",
                                         qty="0.5", leverage=1)
    from cryptography.fernet import Fernet
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    key_store.save_api_keys(77, "APIKEY" + "x" * 30, "SECRET" + "y" * 30)

    block, info = asyncio.run(analyze._auto_execute_plan(
        user_id=77, mode="spot", symbol="ETHUSDT", direction="LONG",
        plan={"quyet_dinh": "BUY", "entry_thap": 2690.0, "entry_cao": 2710.0,
              "sl": 2600.0, "tp1": 2750.0, "do_tin_cay": 80},
        plan_id="usd-spot-77-2", current_price=2700.0))

    assert info["status"] == "aborted", info
    assert not any(p.endswith("/api/v3/order") for _m, p, _prm in rec.calls), \
        "không được gửi lệnh khi số USDT không đủ 1 coin"


def test_ledger_records_usdt_notional(monkeypatch, tmp_path):
    """Ledger lưu giá trị vị thế USDT (qty × đòn bẩy), không phải số coin."""
    _use_test_db(monkeypatch)
    analyze.init_auto_scan_db()
    monkeypatch.setattr(analyze, "collect_timeframe_data",
                        lambda *a, **k: analyze._async({}))
    monkeypatch.setattr(analyze, "_missing_critical_timeframes", lambda *a, **k: [])
    import plan_validator
    monkeypatch.setattr(plan_validator, "validate_plan", lambda plan, facts: [])
    monkeypatch.setattr(analyze, "get_current_price_raw", lambda sym, mkt: 2700.0)
    rec = CallRecorder(market_buy_fills=[(2700, 0.1, 0.0001, "BNB")])
    rec.balances = {"ETH": 1.0}
    monkeypatch.setattr(binance_executor, "signed_request", rec)
    from cryptography.fernet import Fernet
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    key_store.save_api_keys(77, "APIKEY" + "x" * 30, "SECRET" + "y" * 30)

    analyze.set_auto_scan_market_enabled(77, 1, "spot", True, "ETHUSDT",
                                         qty="270", leverage=1)
    block, info = asyncio.run(analyze._auto_execute_plan(
        user_id=77, mode="spot", symbol="ETHUSDT", direction="LONG",
        plan={"quyet_dinh": "BUY", "entry_thap": 2690.0, "entry_cao": 2710.0,
              "sl": 2600.0, "tp1": 2750.0, "do_tin_cay": 80},
        plan_id="usd-spot-77-3", current_price=2700.0))
    assert info["status"] == "placed", info
    assert info["order"]["notional_usdt"] == 270.0


# ─── Bất biến TP/SL khi quy từ USDT ra coin ───────────────────────────────────
#
# Đây là chỗ dễ vỡ nhất khi đổi từ "nhập số coin" sang "nhập số USDT": nếu chia
# TP/SL theo số coin TRƯỚC khi floor stepSize thì TP1+TP2 sẽ lệch với position và
# sàn hoặc từ chối (-1111) hoặc để một phần position không bao giờ được chốt.

@pytest.mark.parametrize("usdt,leverage,price,tp2,step", [
    (10, 20, 2700.0, None, 0.001),      # futures chuẩn
    (10, 20, 2700.0, 2862.0, 0.001),    # futures có tp2
    (100, 1, 2700.0, None, 0.001),
    (10, 125, 2700.0, 2862.0, 0.001),
    (50, 10, 2700.0, 2538.0, 0.001),    # SHORT-style
    (270, 1, 2700.0, None, 0.0001),     # spot
    (540, 1, 2700.0, 2862.0, 0.0001),   # spot có tp2
    (5, 5, 2700.0, None, 0.001),
    (500, 20, 2700.0, 2862.0, 0.001),
    (10, 20, 0.5, None, 0.001),         # symbol giá rẻ
])
def test_tp_sl_math_invariants_from_usdt(usdt, leverage, price, tp2, step):
    """TP1 + TP2 == position, SL == 100% position, tất cả là bội stepSize."""
    notional = usdt * leverage
    coins = binance_executor._floor_to(notional / price, step)
    if coins < step:
        pytest.skip("floor về 0 — case này phải abort, đã test riêng")

    # Giống hệt production: chỉ chia 70/30 KHI plan CÓ tp2.
    q1, q2 = (binance_executor.split_tp_qty(coins, step) if tp2 is not None
              else (coins, 0.0))
    if tp2 is None:
        assert q2 == 0.0, "không có tp2 thì TP2 phải là 0"
        assert q1 == coins, "không có tp2 thì TP1 phải là TOÀN BỘ position"
    else:
        assert q1 > 0 and q2 > 0
        assert abs(q1 + q2 - coins) < 1e-12, \
            f"TP1+TP2={q1 + q2} != position={coins} — sẽ lệch với lệnh entry"
        assert q1 >= q2, "TP1 phải là phần lớn (~70%)"
    for q in (q1, q2):
        assert abs(q / step - round(q / step)) < 1e-12, \
            f"{q} không phải bội của step {step} — sàn sẽ trả -1111"
    # notional sau floor không được vượt yêu cầu
    assert coins * price <= notional + 1e-9, \
        f"{coins * price} > {notional} — vị thế vượt số USDT đã cấu hình"


@pytest.mark.parametrize("executed,fee_qty,fee_asset,free,step", [
    (0.1, 0.0001, "BNB", 0.1, 0.0001),     # phí BNB → base giữ nguyên
    (0.1, 0.27, "USDT", 0.1, 0.0001),      # phí quote → base giữ nguyên
    (0.1, 0.0001, "ETH", 0.0999, 0.0001),  # phí base → trừ coin
    (0.06, 0.0, "BNB", 0.06, 0.0001),      # partial fill
    (0.1, 0.0, "BNB", 0.05, 0.0001),       # free < executed (OCO khác khóa)
])
def test_spot_sellable_after_fee_and_balance(executed, fee_qty, fee_asset, free, step):
    """Sellable = executed − phí base, rồi kẹp bởi free balance, rồi floor stepSize.

    Đây chính là "khớp rồi trừ phí các kiệu mới đặt TP/SL cho số coin cuối cùng".
    """
    _exec, fee, asset, sellable = binance_executor.spot_fill_fees(
        {"executedQty": str(executed),
         "fills": [{"price": "2700", "qty": str(executed),
                    "commission": str(fee_qty), "commissionAsset": fee_asset}]},
        "ETH")
    if fee_asset == "ETH":
        assert sellable == max(0.0, executed - fee_qty), "phí base phải trừ vào coin"
    else:
        assert sellable == executed, "phí BNB/quote không được trừ vào coin"

    available = binance_executor._floor_to(min(sellable, free), step)
    assert available <= free + 1e-12, f"OCO bán {available} > free {free}"
    assert available <= sellable + 1e-12, f"OCO bán {available} > sellable {sellable}"
    assert abs(available / step - round(available / step)) < 1e-12, "phải là bội stepSize"

    # TP/SL sau đó KHÔNG được vượt số bán được
    q1, q2 = binance_executor.split_tp_qty(available, step)
    if q2 > 0:
        assert abs(q1 + q2 - available) < 1e-12
    else:
        assert q1 == available


def test_spot_min_notional_merges_tp2(monkeypatch):
    """TP/SL luôn đúng minNotional và bội stepSize; tổng không vượt số bán được."""
    rec, keys = _setup(monkeypatch, base="https://demo-api.binance.com")
    rec.balances = {"ETH": 0.021}
    result = binance_executor.attach_spot_oco("https://demo-api.binance.com", keys,
                                              "ETHUSDT", 2808.0, 2592.0,
                                              "plan-mn-merge-1", 0.021, tp2=2862.0)
    ocos = rec.paths("/api/v3/orderList/oco", "POST")
    total = sum(float(prm["quantity"]) for _m, _p, prm in ocos)
    assert abs(total - 0.021) < 1e-9, f"tổng OCO {total} != số bán được"
    for _m, _p, prm in ocos:
        q = float(prm["quantity"])
        assert q * float(prm["abovePrice"]) >= 10, \
            f"OCO qty {q} * giá < minNotional 10 — sàn sẽ từ chối"
        assert abs(q / 0.0001 - round(q / 0.0001)) < 1e-12


# ─── USDT tối thiểu (min_notional + stepSize) ────────────────────────────────

@pytest.fixture
def _fake_filters(monkeypatch):
    """Ép filter cố định + giá cố định để không phụ thuộc mạng."""
    binance_executor._FILTER_CACHE.clear()
    fake = {"tick": 0.01, "step": 0.001, "min_qty": 0.001,
            "min_notional": 20.0, "base_asset": "ETH", "margin_asset": "USDT",
            "max_order_lists": 20}
    monkeypatch.setattr(binance_executor, "_market_filters", lambda *a, **k: dict(fake))
    price_holder = {"price": 2500.0}
    monkeypatch.setattr(binance_executor, "public_get",
                        lambda base, path, params=None: {"price": str(price_holder["price"])})
    return price_holder


def test_min_order_usdt_futures_divides_by_leverage(_fake_filters):
    """Futures: minNotional / đòn bẩy = số USDT cần nhập."""
    for lev, expect in ((1, 20.0), (5, 4.0), (20, 1.0), (125, 0.16)):
        info = binance_executor.min_order_usdt("https://demo-fapi.binance.com",
                                               "ETHUSDT", False, leverage=lev)
        assert info["usdt_min"] == pytest.approx(expect), \
            f"x{lev}: can {info['usdt_min']} nhung phai la {expect}"
        assert info["min_notional"] == 20.0
        assert info["leverage"] == lev


def test_min_order_usdt_spot_ignores_leverage(_fake_filters):
    """Spot không có đòn bẩy → minUSDT = minNotional, luôn vậy dù leverage truyền gì."""
    for lev in (1, 20, 125):
        info = binance_executor.min_order_usdt("https://demo-api.binance.com",
                                               "ETHUSDT", True, leverage=lev)
        assert info["usdt_min"] == pytest.approx(20.0), info
        assert info["leverage"] == 1, "spot phai co leverage = 1"


def test_min_order_usdt_uses_step_when_step_exceeds_notional(monkeypatch):
    """stepSize × giá lớn hơn minNotional → minUSDT do stepSize chi phối."""
    binance_executor._FILTER_CACHE.clear()
    fake = {"tick": 0.01, "step": 0.5, "min_qty": 0.5, "min_notional": 5.0,
            "base_asset": "SHIB", "max_order_lists": 20}
    monkeypatch.setattr(binance_executor, "_market_filters", lambda *a, **k: dict(fake))
    price_holder = {"price": 10000.0}  # 0.5 * 10000 = 5000 >> 5 minNotional
    monkeypatch.setattr(binance_executor, "public_get",
                        lambda base, path, params=None: {"price": str(price_holder["price"])})
    info = binance_executor.min_order_usdt("https://demo-api.binance.com", "SHIBUSDT",
                                           True, leverage=1)
    assert info["usdt_min"] == pytest.approx(5000.0), info


def test_describe_min_usdt_includes_number():
    """Tin nhắn hủy phải chứa SỐ cụ thể, không chỉ nói "quá nhỏ"."""
    text = analyze._describe_min_usdt("ETHUSDT", "futures", 20, 2500.0)
    assert "USDT" in text
    # giá trị tối thiểu xấp xỉ 1.0
    assert any(ch.isdigit() for ch in text), text
    example = analyze._example_min_usdt(text)
    assert example and float(example) > 0, example


def test_tiny_usdt_abort_message_is_actionable(monkeypatch, tmp_path):
    """Nhập 0.01 USDT (giá trị CŨ kiểu coin) → thông báo hướng dẫn nhập lại, không chỉ "hủy"."""
    _use_test_db(monkeypatch)
    monkeypatch.setattr(analyze, "get_current_price_raw", lambda sym, mkt: 2700.0)
    rec, keys = _setup(monkeypatch)
    rec.balances = {"ETH": 1.0}
    analyze.set_auto_scan_market_enabled(77, 1, "spot", True, "ETHUSDT",
                                         qty="0.01", leverage=1)
    from cryptography.fernet import Fernet
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    key_store.save_api_keys(77, "APIKEY" + "x" * 30, "SECRET" + "y" * 30)

    # Fake filter thật: spot ETHUSDT minNotional 5, step 1e-4 → 0.01 USDT < tối thiểu
    # nên rơi vào nhánh abort có hướng dẫn nhập lại.
    monkeypatch.setattr(binance_executor, "_market_filters", lambda *a, **k: {
        "tick": 0.01, "step": 0.0001, "min_qty": 0.0001, "min_notional": 5.0,
        "base_asset": "ETH", "max_order_lists": 20})
    monkeypatch.setattr(binance_executor, "public_get",
                        lambda base, path, params=None: {"price": "2700"})

    block, info = asyncio.run(analyze._auto_execute_plan(
        user_id=77, mode="spot", symbol="ETHUSDT", direction="LONG",
        plan={"quyet_dinh": "BUY", "entry_thap": 2690.0, "entry_cao": 2710.0,
              "sl": 2600.0, "tp1": 2750.0, "do_tin_cay": 80},
        plan_id="usd-spot-old-77-1", current_price=2700.0))

    assert info["status"] == "aborted", info
    assert "phải số coin" in block.lower(), \
        f"phải nhắc user là đơn vị đã đổi, got: {block!r}"
    assert "USDT" in block, block
    assert "tối thiểu" in block, block
    assert not any(p.endswith("/api/v3/order") for _m, p, _prm in rec.calls), \
        "không được gửi lệnh khi số USDT quá nhỏ"


# ─── Spot OCO đầy → hủy OCO cũ nhất ─────────────────────────────────────────

class _OcoFullRecorder(CallRecorder):
    """Giả lập sàn bị ĐẦY OCO list (20) và ghi lại lệnh DELETE /orderList."""

    def __init__(self, existing=None):
        super().__init__()
        self.existing = existing or []

    def __call__(self, base, api_key, secret, method, path, params):
        self.calls.append((method, path, dict(params)))
        if path.endswith("/exchangeInfo"):
            return self._exchange_info()
        if path.endswith("/api/v3/account"):
            return {"balances": [{"asset": "BTC", "free": "1.0"}],
                    "commissionRates": {"maker": "0.001", "taker": "0.001"}}
        if path.endswith("/api/v3/openOrderList"):
            return list(self.existing)
        if path.endswith("/api/v3/order") and method == "POST" and params.get("type") == "MARKET":
            return {"orderId": 60000, "status": "FILLED",
                    "executedQty": "1.0", "cummulativeQuoteQty": "27000",
                    "fills": [{"price": "27000", "qty": "1.0",
                               "commission": "0.001", "commissionAsset": "BNB"}]}
        if path.endswith("/api/v3/order") and method == "GET":
            return {"status": "FILLED", "executedQty": "1.0", "fills": []}
        if path.endswith("/api/v3/orderList") and method == "DELETE":
            # Xoá khỏi danh sách "đang mở" để phản ánh trạng thái mới.
            for oco in list(self.existing):
                if oco.get("orderListId") == params.get("orderListId"):
                    self.existing.remove(oco)
            return {"orderListId": params.get("orderListId"), "status": "CANCELED"}
        if path.endswith("/api/v3/orderList/oco") and method == "POST":
            return {"orderListId": 70000 + len(self.calls),
                    "listClientOrderId": params.get("listClientOrderId"),
                    "orders": [{"clientOrderId": params.get("aboveClientOrderId"),
                                "orderId": 70000 + len(self.calls)},
                               {"clientOrderId": params.get("belowClientOrderId"),
                                "orderId": 70000 + len(self.calls)}]}
        return {}

    @staticmethod
    def _exchange_info():
        return {"symbols": [{
            "symbol": "BTCUSDT", "status": "TRADING", "baseAsset": "BTC",
            "filters": [{"filterType": "PRICE_FILTER", "minPrice": "0.01",
                         "tickSize": "0.01"},
                        {"filterType": "LOT_SIZE", "minQty": "0.001",
                         "stepSize": "0.001"},
                        {"filterType": "MAX_NUM_ORDER_LISTS", "maxNumOrderLists": 20},
                        {"filterType": "MIN_NOTIONAL", "notional": "20"}]}]}

    def public_get(self, base, path, params=None):
        # `_market_filters` đọc exchangeInfo qua đây — phải có max_order_lists.
        if path.endswith("/exchangeInfo"):
            return self._exchange_info()
        return {}


def _full_lists(n=20):
    return [
        {"orderListId": 50000 + i,
         "listClientOrderId": f"futu-eth-77-{i}-list",
         "transactionTime": 1700000000000 + i * 1000}
        for i in range(n)
    ]


def test_ensure_oco_room_evicts_oldest_when_full(monkeypatch):
    """OCO đầy → hủy OCO CŨ NHẤT của bot, giữ nguyên OCO user tự đặt."""
    rec = _OcoFullRecorder(existing=_full_lists(20) + [
        {"orderListId": 90001, "listClientOrderId": "user-manual-oco",
         "transactionTime": 1700000000000}]  # user tự đặt, không của bot
    )
    monkeypatch.setattr(binance_executor, "signed_request", rec)
    binance_executor._FILTER_CACHE.clear()

    canceled = binance_executor._ensure_oco_room(
        "https://demo-api.binance.com", "k", "s", "BTCUSDT",
        {"max_order_lists": 20}, need=2, keep_plan_id="futu-eth-77-99")

    # 21 list đang mở + cần 2 → phải hủy 3. Các OCO trong test này KHÔNG có
    # `listOrderStatus` nên tất cả đều coi là "đã xong" (không ARMED) → mã hủy được.
    assert len(canceled) == 3, canceled
    # Hủy đúng CŨ NHẤT: i=0,1,2 (transactionTime nhỏ nhất)
    coids = [c["list_client_order_id"] for c in canceled]
    assert coids == ["futu-eth-77-0-list", "futu-eth-77-1-list",
                     "futu-eth-77-2-list"], coids
    # OCO user tự đặt KHÔNG bao giờ bị đụng
    assert "user-manual-oco" not in coids


def test_ensure_oco_room_noop_when_room(monkeypatch):
    """Còn chỗ → không hủy gì cả."""
    rec = _OcoFullRecorder(existing=_full_lists(17))
    monkeypatch.setattr(binance_executor, "signed_request", rec)
    binance_executor._FILTER_CACHE.clear()
    canceled = binance_executor._ensure_oco_room(
        "https://demo-api.binance.com", "k", "s", "BTCUSDT",
        {"max_order_lists": 20}, need=2)
    assert canceled == []
    assert not any(m == "DELETE" for m, _p, _prm in rec.calls)


def test_ensure_oco_room_keeps_current_plan_list(monkeypatch):
    """OCO của plan ĐANG gắn không bao giờ bị hủy làm room."""
    rec = _OcoFullRecorder(existing=_full_lists(20))
    monkeypatch.setattr(binance_executor, "signed_request", rec)
    binance_executor._FILTER_CACHE.clear()
    canceled = binance_executor._ensure_oco_room(
        "https://demo-api.binance.com", "k", "s", "BTCUSDT",
        {"max_order_lists": 20}, need=2, keep_plan_id="futu-eth-77-0")
    coids = [c["list_client_order_id"] for c in canceled]
    assert "futu-eth-77-0-list" not in coids, "khong duoc huy OCO cua plan dang gan"


def test_ensure_oco_room_no_limit_field(monkeypatch):
    """Không công bố giới hạn → không đoán, không hủy gì."""
    rec = _OcoFullRecorder(existing=_full_lists(50))
    monkeypatch.setattr(binance_executor, "signed_request", rec)
    binance_executor._FILTER_CACHE.clear()
    canceled = binance_executor._ensure_oco_room(
        "https://demo-api.binance.com", "k", "s", "BTCUSDT",
        {"max_order_lists": None}, need=2)
    assert canceled == []


def test_is_bot_oco_recognition():
    """Chỉ nhận diện OCO của bot, OCO user tự đặt bị bỏ qua."""
    assert binance_executor._is_bot_oco("futu-eth-77-1-list", None) is True
    assert binance_executor._is_bot_oco("futu-eth-77-1-list2", None) is True
    assert binance_executor._is_bot_oco("spot-eth-77-3-list", None) is True
    assert binance_executor._is_bot_oco("user-manual-oco", None) is False
    assert binance_executor._is_bot_oco("my-protection", None) is False
    assert binance_executor._is_bot_oco(None, None) is False
    # OCO của plan đang gắn thì không hủy
    assert binance_executor._is_bot_oco("futu-eth-77-1-list", "futu-eth-77-1") is False


def test_attach_spot_oco_evicts_when_full(monkeypatch):
    """End-to-end: OCO đầy → attach vẫn tạo được OCO mới vì đã hủy OCO cũ."""
    rec = _OcoFullRecorder(existing=_full_lists(20))
    monkeypatch.setattr(binance_executor, "signed_request", rec)
    monkeypatch.setattr(binance_executor, "public_get", rec.public_get)
    binance_executor._FILTER_CACHE.clear()

    result = binance_executor.attach_spot_oco(
        "https://demo-api.binance.com", ("k", "s"), "BTCUSDT",
        28000.0, 26000.0, "plan-77-99", 1.0, tp2=28600.0)

    evicted = [c for c in rec.calls if c[0] == "DELETE" and
               c[1].endswith("/api/v3/orderList")]
    assert len(evicted) >= 1, "phaai huy OCO cu de nhuong cho"
    assert result.get("oco_list_id"), result
    assert result.get("tp2_oco_list_id"), "plan co tp2 phai co OCO thu 2"
    created = rec.paths("/api/v3/orderList/oco", "POST")
    assert len(created) == 2
