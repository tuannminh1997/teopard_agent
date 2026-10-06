"""Đặt lệnh Binance thật từ plan (futures: limit + TP/SL algo đặt trước khi khớp; spot: limit + OCO sau khi khớp).

Chỉ dùng requests + HMAC-SHA256. Base URL lấy từ env:
  FUTURES_API_BASE (mặc định https://fapi.binance.com — Railway đặt demo host khi test)
  SPOT_API_BASE    (mặc định https://api.binance.com)
"""
import hashlib
import hmac
import math
import os
import time
import urllib.parse

import requests

FUTURES_API_BASE = (os.getenv("FUTURES_API_BASE") or "https://fapi.binance.com").rstrip("/")
SPOT_API_BASE = (os.getenv("SPOT_API_BASE") or "https://api.binance.com").rstrip("/")

_FILTER_CACHE: dict = {}


class ExecutorError(RuntimeError):
    def __init__(self, code, msg):
        super().__init__(f"{code}: {msg}")
        self.code = code
        self.msg = msg


# ─── Hạ tầng request ─────────────────────────────────────────────────────────

def signed_request(base: str, api_key: str, secret: str, method: str, path: str, params: dict) -> dict:
    """Gửi request có chữ ký; ném ExecutorError khi Binance trả code lỗi (<0 hoặc >=400)."""
    params = dict(params)
    params["timestamp"] = int(time.time() * 1000)
    params["recvWindow"] = 5000
    query = urllib.parse.urlencode(params)
    params["signature"] = hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    url = f"{base}{path}"
    if method == "GET":
        r = requests.get(url, params=params, headers={"X-MBX-APIKEY": api_key}, timeout=15)
    else:
        r = requests.post(url, data=params, headers={"X-MBX-APIKEY": api_key}, timeout=15)
    try:
        data = r.json()
    except Exception:
        raise ExecutorError(r.status_code, f"response không phải JSON: {r.text[:200]}")
    if isinstance(data, dict) and data.get("code") is not None:
        try:
            code = int(data["code"])
        except Exception:
            code = r.status_code
        if code != 200:
            raise ExecutorError(code, str(data.get("msg") or data))
    if r.status_code >= 400:
        raise ExecutorError(r.status_code, str(data))
    return data if isinstance(data, dict) else {"data": data}


def public_get(base: str, path: str, params: dict | None = None) -> dict:
    r = requests.get(f"{base}{path}", params=params or {}, timeout=10)
    data = r.json()
    if r.status_code >= 400:
        raise ExecutorError(r.status_code, str(data))
    return data


def _market_filters(base: str, api_key: str, secret: str, symbol: str, spot: bool) -> dict:
    """tickSize + stepSize + minNotional (cache theo phiên process)."""
    key = (base, symbol, spot)
    if key in _FILTER_CACHE:
        return _FILTER_CACHE[key]
    path = "/api/v3/exchangeInfo" if spot else "/fapi/v1/exchangeInfo"
    info = public_get(base, path, {"symbol": symbol})
    sym = next((s for s in info.get("symbols", []) if s.get("symbol") == symbol), None)
    if sym is None:
        raise ExecutorError(-1121, f"{symbol} không tồn tại trên {base}")
    out = {"tick": 0.01, "step": 0.001, "min_notional": None, "margin_asset": None}
    if not spot:
        out["margin_asset"] = sym.get("marginAsset")
    for f in sym.get("filters", []):
        ft = f.get("filterType")
        if ft == "PRICE_FILTER":
            out["tick"] = float(f["tickSize"])
        elif ft == "LOT_SIZE":
            out["step"] = float(f["stepSize"])
        elif ft in ("MIN_NOTIONAL", "NOTIONAL") and not spot:
            out["min_notional"] = float(f.get("notional") or f.get("minNotional") or 0)
    _FILTER_CACHE[key] = out
    return out


def _floor_to(value: float, step: float) -> float:
    if step <= 0:
        return value
    return round(math.floor(value / step + 1e-9) * step, 10)


def _fmt(value: float) -> str:
    s = f"{value:.8f}".rstrip("0").rstrip(".")
    return s or "0"


def _bump_plan_id(plan_id: str) -> str:
    """futu-eth-3 -> futu-eth-4 (chống trùng clientOrderId với phiên trước còn lệnh treo)."""
    head, _, tail = plan_id.rpartition("-")
    try:
        return f"{head}-{int(tail) + 1}"
    except ValueError:
        return f"{plan_id}-2"


# ─── Futures ──────────────────────────────────────────────────────────────────

def _detect_hedge(base: str, api_key: str, secret: str) -> bool:
    resp = signed_request(base, api_key, secret, "GET", "/fapi/v1/positionSide/dual", {})
    return bool(resp.get("dualSidePosition"))


def _set_leverage(base: str, api_key: str, secret: str, symbol: str, leverage: int) -> None:
    try:
        signed_request(base, api_key, secret, "POST", "/fapi/v1/leverage",
                       {"symbol": symbol, "leverage": int(leverage)})
        return
    except ExecutorError as exc:
        # demo-fapi có symbol (ETHU) set leverage luôn trả -1000 dù giá trị hợp lệ —
        # hỏi lại leverage hiện tại: đúng yêu cầu thì cho qua, sai thì chặn (không đặt sai đòn bẩy).
        current = None
        try:
            rows = signed_request(base, api_key, secret, "GET", "/fapi/v2/positionRisk",
                                  {"symbol": symbol})
            items = rows.get("data") if isinstance(rows, dict) else rows
            if isinstance(items, dict):
                items = [items]
            for row in items or []:
                if row.get("symbol") == symbol:
                    current = int(float(row.get("leverage") or 0))
                    break
        except Exception:
            pass
        if current == int(leverage):
            return
        raise ExecutorError(
            exc.code,
            f"{exc.msg} — không set được đòn bẩy {leverage}x cho {symbol} "
            f"(sàn đang giữ {current if current is not None else '?'}x)",
        )


def place_futures_plan(
    symbol: str,
    direction: str,
    entry_price: float,
    tp1: float,
    sl: float,
    qty: float,
    leverage: int | None,
    keys: tuple[str, str],
    plan_id: str,
) -> dict:
    """LIMIT entry + TP/SL (2 lệnh algo) đặt ngay, không chờ khớp. Trả dict ids đã dùng."""
    base, (api_key, secret) = FUTURES_API_BASE, keys
    flt = _market_filters(base, api_key, secret, symbol, spot=False)
    qty = _floor_to(qty, flt["step"])
    price = _floor_to(entry_price, flt["tick"])
    if qty <= 0:
        raise ExecutorError(-4164, f"khối lượng {qty} bị làm tròn về 0 theo step {flt['step']}")
    if flt["min_notional"] and qty * price < flt["min_notional"]:
        raise ExecutorError(
            -4164,
            f"notional {qty * price:.4g} < tối thiểu {flt['min_notional']:g} USDT — tăng khối lượng",
        )
    if leverage:
        _set_leverage(base, api_key, secret, symbol, int(leverage))
    # Symbol margin bằng asset khác USDT (vd ETHU = United Stables) — cần Multi-Assets Mode
    # để pool USDT/USDC làm collateral; bật lần đầu, bật lại sẽ báo "no need" (bỏ qua).
    if flt.get("margin_asset") and flt["margin_asset"] != "USDT":
        try:
            signed_request(base, api_key, secret, "POST", "/fapi/v1/multiAssetsMargin",
                           {"multiAssetsMargin": "true"})
        except ExecutorError:
            pass
    hedge = _detect_hedge(base, api_key, secret)

    side = "BUY" if direction == "LONG" else "SELL"
    close_side = "SELL" if side == "BUY" else "BUY"
    pos_side = direction if hedge else "BOTH"

    # Entry limit — chống trùng clientOrderId bằng cách tăng số thứ tự plan.
    used_plan = plan_id
    entry_resp = None
    for _ in range(6):
        try:
            entry_resp = signed_request(base, api_key, secret, "POST", "/fapi/v1/order", {
                "symbol": symbol, "side": side, "type": "LIMIT",
                "timeInForce": "GTC", "quantity": _fmt(qty), "price": _fmt(price),
                "positionSide": pos_side,
                "clientOrderId": f"{used_plan}-e",
            })
            break
        except ExecutorError as exc:
            if exc.code == -2010 and "duplicate" in exc.msg.lower():
                used_plan = _bump_plan_id(used_plan)
                continue
            raise
    if entry_resp is None:
        raise ExecutorError(-2010, "không tạo được entry sau 6 lần chống trùng clientOrderId")

    def _algo(order_type: str, trigger: float, coid: str) -> int:
        params = {
            "symbol": symbol, "side": close_side, "type": order_type,
            "algoType": "CONDITIONAL",
            "triggerPrice": _fmt(_floor_to(trigger, flt["tick"])),
            "quantity": _fmt(qty),
            "positionSide": pos_side,
            "workingType": "CONTRACT_PRICE",
            "clientAlgoId": coid,
        }
        if not hedge:
            params["reduceOnly"] = "true"
        resp = signed_request(base, api_key, secret, "POST", "/fapi/v1/algoOrder", params)
        return int(resp["algoId"])

    try:
        tp_algo = _algo("TAKE_PROFIT_MARKET", tp1, f"{used_plan}-tp")
        sl_algo = _algo("STOP_MARKET", sl, f"{used_plan}-sl")
    except ExecutorError as exc:
        # Entry có thể đã khớp ngay trong khoảng thời gian chớp nhoáng trước khi TP/SL fail
        # (giá chạy xuyên qua vùng entry) → hủy không được nữa. Hỏi tiếp exchange: entry đã khớp
        # bao nhiêu? Nếu đã khớp thì ĐÓNG NGAY theo thị trường — không bao giờ để position trần
        # mà không có TP/SL. Nếu chưa khớp → hủy entry, không treo lệnh mồ côi.
        executed = 0.0
        try:
            info = signed_request(base, api_key, secret, "GET", "/fapi/v1/order",
                                  {"symbol": symbol, "orderId": entry_resp.get("orderId")})
            executed = float(info.get("executedQty") or 0)
        except Exception:
            executed = 0.0
        if executed > 0:
            # Bất kể thế nào position PHẢI bị đóng — không được để trần (cháy tài khoản).
            close_params = {
                "symbol": symbol, "side": close_side, "type": "MARKET",
                "quantity": _fmt(executed), "positionSide": pos_side,
                **({} if hedge else {"reduceOnly": "true"}),
            }
            closed = False
            for _ in range(3):
                try:
                    signed_request(base, api_key, secret, "POST", "/fapi/v1/order", close_params)
                    closed = True
                    break
                except Exception:
                    time.sleep(1)
            if closed:
                raise ExecutorError(
                    exc.code,
                    f"{exc.msg} — entry đã khớp {_fmt(executed)} nên ĐÃ ĐÓNG NGAY theo giá "
                    "thị trường (không gắn được TP/SL nên không giữ vị thế trần)",
                )
            raise ExecutorError(
                exc.code,
                f"{exc.msg} — ⚠️⚠️ entry đã khớp {_fmt(executed)} NHƯNG ĐÓNG THẤT BẠI sau 3 lần "
                "thử — VÀO GUI ĐÓNG NGAY, vị thế đang trần!",
            )
        # Chưa khớp → hủy theo orderId (bỏ túi từ response) vì demo-fapi có thể không giữ
        # clientOrderId; fallback theo coid.
        for cancel_params in (
            {"symbol": symbol, "orderId": entry_resp.get("orderId")},
            {"symbol": symbol, "origClientOrderId": f"{used_plan}-e"},
        ):
            try:
                signed_request(base, api_key, secret, "DELETE", "/fapi/v1/order", cancel_params)
                break
            except Exception:
                continue
        raise

    return {
        "plan_id": used_plan,
        "entry_order_id": entry_resp.get("orderId"),
        "tp_algo_id": tp_algo,
        "sl_algo_id": sl_algo,
        "qty": qty,
        "entry_price": price,
        "hedge": hedge,
    }


# ─── Spot ────────────────────────────────────────────────────────────────────

def place_spot_plan(
    symbol: str,
    entry_price: float,
    tp1: float,
    sl: float,
    qty: float,
    keys: tuple[str, str],
    plan_id: str,
    fill_timeout: int = 30,
    poll_seconds: int = 2,
) -> dict:
    """LIMIT BUY; chờ khớp (tối đa fill_timeout giây) rồi gắn TP+SL bằng 1 lệnh OCO."""
    base, (api_key, secret) = SPOT_API_BASE, keys
    flt = _market_filters(base, api_key, secret, symbol, spot=True)
    qty = _floor_to(qty, flt["step"])
    price = _floor_to(entry_price, flt["tick"])
    if qty <= 0:
        raise ExecutorError(-4164, f"khối lượng {qty} bị làm tròn về 0 theo step {flt['step']}")

    side = "BUY"
    used_plan = plan_id
    entry_resp = None
    for _ in range(6):
        try:
            entry_resp = signed_request(base, api_key, secret, "POST", "/api/v3/order", {
                "symbol": symbol, "side": side, "type": "LIMIT",
                "timeInForce": "GTC", "quantity": _fmt(qty), "price": _fmt(price),
                "newClientOrderId": f"{used_plan}-e",
            })
            break
        except ExecutorError as exc:
            if exc.code == -2010 and "duplicate" in exc.msg.lower():
                used_plan = _bump_plan_id(used_plan)
                continue
            raise
    entry_id = entry_resp.get("orderId")
    client_id = entry_resp.get("clientOrderId") or f"{used_plan}-e"

    # Chờ lệnh khớp (spot không gắn TP/SL trước được vì cần có coin để bán).
    deadline = time.time() + max(0, fill_timeout)
    status = "NEW"
    while time.time() < deadline:
        order = signed_request(base, api_key, secret, "GET", "/api/v3/order",
                               {"symbol": symbol, "origClientOrderId": client_id})
        status = order.get("status", status)
        if status == "FILLED":
            break
        if status in ("CANCELED", "EXPIRED", "REJECTED"):
            break
        time.sleep(poll_seconds)

    result = {"plan_id": used_plan, "entry_order_id": entry_id, "qty": qty,
              "entry_price": price, "filled": status == "FILLED"}
    if status != "FILLED":
        result["status"] = status
        return result

    try:
        oco = signed_request(base, api_key, secret, "POST", "/api/v3/order/oco", {
            "symbol": symbol, "side": "SELL", "quantity": _fmt(qty),
            "price": _fmt(_floor_to(tp1, flt["tick"])),
            "stopPrice": _fmt(_floor_to(sl, flt["tick"])),
            "timeInForce": "GTC",
            "newClientOrderId": f"{used_plan}-tp",
            "stopClientOrderId": f"{used_plan}-sl",
        })
    except ExecutorError as oco_exc:
        # Đã mua được coin nhưng gắn OCO thất bại → BÁN NGAY thị trường, không giữ trần.
        closed = False
        for _ in range(3):
            try:
                signed_request(base, api_key, secret, "POST", "/api/v3/order", {
                    "symbol": symbol, "side": "SELL", "type": "MARKET",
                    "quantity": _fmt(qty), "newClientOrderId": f"{used_plan}-mc",
                })
                closed = True
                break
            except Exception:
                time.sleep(1)
        if closed:
            raise ExecutorError(
                oco_exc.code,
                f"{oco_exc.msg} — đã mua {_fmt(qty)} {symbol} nhưng gắn TP/SL thất bại → "
                "ĐÃ BÁN NGAY theo thị trường (không giữ vị thế trần)",
            )
        raise ExecutorError(
            oco_exc.code,
            f"{oco_exc.msg} — ⚠️⚠️ đã mua {_fmt(qty)} {symbol} NHƯNG ĐÓNG THẤT BẠI sau 3 lần "
            "thử — VÀO GUI ĐÓNG NGAY, vị thế đang trần!",
        )
    result["oco_list_id"] = oco.get("orderListId")
    result["tp_leg_order_id"] = oco.get("orderId")
    return result


def place_plan(market: str, symbol: str, direction: str, entry_price: float,
               tp1: float, sl: float, qty: float, leverage: int | None,
               keys: tuple[str, str], plan_id: str) -> dict:
    if market == "futures":
        return place_futures_plan(symbol, direction, entry_price, tp1, sl, qty, leverage, keys, plan_id)
    return place_spot_plan(symbol, entry_price, tp1, sl, qty, keys, plan_id)


def current_price(market: str, symbol: str) -> float | None:
    base = FUTURES_API_BASE if market == "futures" else SPOT_API_BASE
    path = "/fapi/v1/ticker/price" if market == "futures" else "/api/v3/ticker/price"
    try:
        return float(public_get(base, path, {"symbol": symbol})["price"])
    except Exception:
        return None


# ─── Hủy lệnh TREO khi autoscan tắt ──────────────────────────────────────────

def cancel_pending_plan_orders(keys: tuple[str, str], market: str, rows: list[dict]) -> list[dict]:
    """Hủy mọi lệnh TREO (chưa khớp) của các plan trong rows:
    - entry còn MỞ (executedQty=0) → hủy entry + TP/SL đi kèm (re-check entry trước khi
      đụng TP/SL để không bao giờ gỡ bảo vệ của position vừa khớp lén giữa chừng);
    - entry ĐÃ KHỚP (FILLED/executed>0) → KHÔNG ĐỤNG gì (position + TP/SL giữ nguyên);
    - entry không còn tồn tại (-2011) → hủy TP/SL mồ côi còn sót;
    - không xác định được trạng thái (mạng lỗi) → bỏ qua row (an toàn).
    Trả về danh sách lệnh đã hủy."""
    is_futures = market == "futures"
    base = FUTURES_API_BASE if is_futures else SPOT_API_BASE
    api_key, secret = keys
    cancelled: list[dict] = []

    def _entry_state(entry_id) -> str:
        path = "/fapi/v1/order" if is_futures else "/api/v3/order"
        try:
            info = signed_request(base, api_key, secret, "GET", path,
                                  {"symbol": symbol, "orderId": entry_id})
            executed = float(info.get("executedQty") or 0)
            return "filled" if (info.get("status") == "FILLED" or executed > 0) else "open"
        except ExecutorError as exc:
            return "gone" if exc.code == -2011 else "unknown"
        except Exception:
            return "unknown"

    for row in rows:
        symbol = row.get("symbol")
        entry_id = row.get("entry_order_id")
        if not symbol or not entry_id:
            continue

        state = _entry_state(entry_id)
        if state in ("filled", "unknown"):
            # Đã khớp (giữ nguyên) hoặc không xác định được (không đụng gì).
            continue

        if state == "open":
            entry_path = "/fapi/v1/order" if is_futures else "/api/v3/order"
            try:
                signed_request(base, api_key, secret, "DELETE", entry_path,
                               {"symbol": symbol, "orderId": entry_id})
                cancelled.append({"kind": "entry", "symbol": symbol, "order_id": entry_id})
            except Exception:
                pass  # có thể vừa khớp giữa chừng → bước re-check bên dưới decides
            # Re-check: entry có lấp trước khi ta gỡ TP/SL không?
            if _entry_state(entry_id) not in ("gone", "open"):
                continue

        # Entry chắc chắn không giữ position nào → TP/SL đi kèm là mồ côi, hủy.
        tp_id = row.get("tp_algo_id")
        sl_id = row.get("sl_algo_id")
        if is_futures:
            for algo_id in (tp_id, sl_id):
                if not algo_id:
                    continue
                try:
                    signed_request(base, api_key, secret, "DELETE", "/fapi/v1/algoOrder",
                                   {"symbol": symbol, "algoid": algo_id})
                    cancelled.append({"kind": "algo", "symbol": symbol, "algo_id": algo_id})
                except Exception:
                    continue
        else:
            if tp_id:  # spot: tp_algo_id giữ oco_list_id
                try:
                    signed_request(base, api_key, secret, "DELETE", "/api/v3/orderList",
                                   {"orderListId": tp_id})
                    cancelled.append({"kind": "oco", "symbol": symbol, "list_id": tp_id})
                except Exception:
                    pass
    return cancelled
