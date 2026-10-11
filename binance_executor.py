"""Đặt lệnh Binance thật từ plan (futures: limit + TP/SL algo; spot: market + OCO).

Chỉ dùng requests + HMAC-SHA256. Base URL lấy từ env:
  FUTURES_API_BASE (mặc định DEMO — xem lưu ý an toàn bên dưới)
  SPOT_API_BASE    (mặc định DEMO)

LƯU Ý AN TOÀN: mặc định cố tình là DEMO, KHÔNG phải live. Nếu biến env bị xoá nhầm
(ví dụ reset biến trên Railway), bot sẽ trỏ về demo — hỏng thì mất tiền ảo, chứ không
bao giờ âm thầm đặt lệnh tiền thật. Muốn chạy live PHẢI chủ động set biến sang host
production và khởi động lại.
"""
import hashlib
import hmac
import math
import os
import time
import urllib.parse

import requests

def _safe_log(msg: str) -> None:
    """Log an toàn với console Windows (cp1252) — tiếng Việt có dấu sẽ UnicodeEncodeError.

    Log cùng cấp bot crash nếu bot coi đó là lỗi — dùng cho mọi log tại thời điểm import.
    """
    try:
        print(msg, flush=True)
    except Exception:
        try:
            print(msg.encode(ascii, replace).decode(ascii), flush=True)
        except Exception:
            pass


FUTURES_API_BASE = (os.getenv("FUTURES_API_BASE")
                    or "https://demo-fapi.binance.com").rstrip("/")
SPOT_API_BASE = (os.getenv("SPOT_API_BASE")
                 or "https://demo-api.binance.com").rstrip("/")

_FILTER_CACHE: dict = {}
# Cache tỷ lệ phí spot theo host (account-level, ít đổi).
# Thời gian tối đa chờ Binance cập nhật executedQty sau MARKET BUY.
SPOT_FILL_TIMEOUT_SECONDS = int(os.getenv("SPOT_FILL_TIMEOUT_SECONDS", "30"))
# Market BUY thường khớp ngay, nhưng Binance cần một khoảng ngắn để cập nhật
# số dư coin trước khi nhận lệnh OCO bán.
SPOT_OCO_DELAY_SECONDS = max(0, int(os.getenv("SPOT_OCO_DELAY_SECONDS", "3")))
# TTL cho LIMIT entry orders futures (mặc định 24h = 86400s). Khi hết hạn:
# - Hủy entry GTC còn mở
# - Hủy TP/SL algo đi kèm (nếu entry chưa khớp)
# - Không hủy nếu entry đã khớp một phần/hết
FUTURES_ENTRY_TTL_SECONDS = int(os.getenv("FUTURES_ENTRY_TTL_SECONDS", "86400"))


class ExecutorError(RuntimeError):
    def __init__(self, code, msg):
        super().__init__(f"{code}: {msg}")
        self.code = code
        self.msg = msg


# ─── Hạ tầng request ─────────────────────────────────────────────────────────

# Chênh lệch serverTime - localTime (ms), lưu theo từng host.
# Clock local lệch vài giây là Binance trả -1021 và CHẶN TOÀN BỘ lệnh (futures lẫn spot).
_TIME_OFFSET_MS: dict[str, int] = {}


def _time_path(path: str) -> str:
    return "/fapi/v1/time" if path.startswith("/fapi") else "/api/v3/time"


def _sync_server_time(base: str, path: str) -> None:
    """Đo serverTime - localTime (endpoint public, không cần chữ ký)."""
    try:
        r = requests.get(f"{base}{_time_path(path)}", timeout=10)
        _TIME_OFFSET_MS[base] = int(r.json()["serverTime"]) - int(time.time() * 1000)
    except Exception:
        _TIME_OFFSET_MS[base] = 0


def signed_request(base: str, api_key: str, secret: str, method: str, path: str, params: dict) -> dict:
    """Gửi request có chữ ký; ném ExecutorError khi Binance trả code lỗi (<0 hoặc >=400).

    - Lỗi mạng (timeout/reset) gói thành ExecutorError(-1000) để mọi nhánh cleanup
      (hủy entry, đóng position) bắt được — nếu để nó thoát thô thì các lệnh
      `except ExecutorError` sẽ bỏ qua cleanup và để position trần.
    - Đồng bộ giờ với server và tự thử lại 1 lần khi gặp -1021 (clock lệch).
    """
    params = dict(params)
    params["recvWindow"] = 5000
    url = f"{base}{path}"

    for attempt in range(2):
        if base not in _TIME_OFFSET_MS:
            _sync_server_time(base, path)
        params["timestamp"] = int(time.time() * 1000) + int(_TIME_OFFSET_MS.get(base, 0))
        params.pop("signature", None)
        query = urllib.parse.urlencode(params)
        params["signature"] = hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()

        try:
            # Dùng requests.request để GIỮ NGUYÊN method. Trước đây nhánh là
            # "if GET ... else requests.post" → MỌI DELETE đều bị gửi thành POST
            # (POST /fapi/v1/order = tạo lệnh, không phải hủy) → hủy lệnh treo chưa từng chạy được.
            r = requests.request(method, url, params=params if method == "GET" else None,
                                 data=None if method == "GET" else params,
                                 headers={"X-MBX-APIKEY": api_key}, timeout=15)
        except requests.RequestException as exc:
            raise ExecutorError(-1000, f"lỗi mạng khi gọi {path}: {exc.__class__.__name__}")

        try:
            data = r.json()
        except Exception:
            raise ExecutorError(r.status_code, f"response không phải JSON: {r.text[:200]}")

        code: int = r.status_code
        msg = str(data)
        if isinstance(data, dict) and data.get("code") is not None:
            try:
                code = int(data["code"])
            except Exception:
                code = r.status_code
            msg = str(data.get("msg") or data)

        if code == -1021 and attempt == 0:
            # Clock local lệch server → đo lại offset rồi thử đúng 1 lần nữa.
            _sync_server_time(base, path)
            continue
        if isinstance(data, dict) and data.get("code") is not None and code != 200:
            raise ExecutorError(code, msg)
        if r.status_code >= 400:
            raise ExecutorError(r.status_code, msg)
        return data if isinstance(data, dict) else {"data": data}

    raise ExecutorError(-1021, "đồng bộ giờ với server thất bại")


def public_get(base: str, path: str, params: dict | None = None) -> dict:
    try:
        r = requests.get(f"{base}{path}", params=params or {}, timeout=10)
    except requests.RequestException as exc:
        raise ExecutorError(-1000, f"lỗi mạng khi gọi {path}: {exc.__class__.__name__}")
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
    out = {"tick": 0.01, "step": 0.001, "min_notional": None, "margin_asset": None,
           "base_asset": None, "min_qty": 0.0, "max_order_lists": None}
    out["base_asset"] = sym.get("baseAsset")
    if not spot:
        out["margin_asset"] = sym.get("marginAsset")
    for f in sym.get("filters", []):
        ft = f.get("filterType")
        if ft == "PRICE_FILTER":
            out["tick"] = float(f["tickSize"])
        elif ft == "LOT_SIZE":
            out["step"] = float(f["stepSize"])
            out["min_qty"] = float(f.get("minQty") or 0)
        elif ft in ("MIN_NOTIONAL", "NOTIONAL"):
            # Spot và Futures đều cần — thiếu check này thì lệnh dưới notional tối thiểu
            # chỉ bị sàn từ chối (-1013) sau khi đã gửi đi.
            raw = f.get("notional") or f.get("minNotional")
            if raw is not None:
                out["min_notional"] = float(raw)
        elif ft == "MAX_NUM_ORDER_LISTS":
            # Spot giới hạn số OCO list đồng thời mỗi symbol. Plan có tp2 cần 2 OCO
            # nên dễ chạm trần — biết trước trần này mới giải phóng được chỗ trước.
            raw = f.get("maxNumOrderLists") or f.get("limit")
            if raw is not None:
                out["max_order_lists"] = int(raw)
    _FILTER_CACHE[key] = out
    return out


def min_order_usdt(base: str, symbol: str, spot: bool, leverage: int = 1,
                   price: float | None = None) -> dict:
    """Số USDT TỐI THIỂU để 1 lệnh qua được filter thật của sàn.

    Lệnh bị từ chối khi vi phạm 1 trong 2 filter:
      1. notional < minNotional   → cần qty_coin * price >= min_notional
      2. qty_coin < stepSize      → floor về 0 thì sàn từ chối luôn

    Từ đó:
      spot:    usdt_min = max(min_notional, step * price)
      futures: usdt_min = max(min_notional, step * price) / leverage

    Trả dict có cả `usdt_min` (số cần nhập) và cách tính để UI giải thích được.
    `price` None → lấy giá hiện tại thật từ ticker để tính cho chính xác.
    """
    flt = _market_filters(base, "", "", symbol, spot)
    if price is None:
        path = "/api/v3/ticker/price" if spot else "/fapi/v1/ticker/price"
        price = float(public_get(base, path, {"symbol": symbol}).get("price") or 0)
    price = float(price or 0)
    min_notional = float(flt.get("min_notional") or 0)
    step = float(flt.get("step") or 0)
    min_qty = float(flt.get("min_qty") or 0)

    if price <= 0:
        return {"usdt_min": None, "reason": f"khong doc duoc gia cua {symbol} tren {base}"}

    need_notional = max(min_notional, min_qty * price, step * price)
    lev = max(1, int(leverage or 1)) if not spot else 1
    usdt_min = need_notional / lev
    return {
        "usdt_min": usdt_min,
        "symbol": symbol,
        "market": "spot" if spot else "futures",
        "leverage": lev,
        "price": price,
        "min_notional": min_notional,
        "step_size": step,
        "min_qty": min_qty,
        "reason": (
            f"notional toi thieu {min_notional:g} USDT"
            + (f" / don bay x{lev}" if not spot else "")
        ),
    }


def _free_base_qty(base: str, api_key: str, secret: str, flt: dict) -> float:
    """Số coin THẬT đang free trong tài khoản spot.

    Sàn trừ phí mua thẳng vào coin (taker ~0.1%) nên số coin nhận được luôn nhỏ hơn qty
    đã khớp. Bán đúng qty lý thuyết sẽ bị từ chối -2010 "insufficient balance".
    """
    asset = flt.get("base_asset")
    if not asset:
        return 0.0
    acct = signed_request(base, api_key, secret, "GET", "/api/v3/account", {})
    for b in acct.get("balances", []):
        if b.get("asset") == asset:
            return float(b.get("free", 0) or 0)
    return 0.0


def spot_fill_fees(order: dict, base_asset: str) -> tuple[float, float, str | None, float]:
    """Tính (executed_qty, fee_qty, fee_asset, sellable_qty) từ response/order spot.

    - Phí trả bằng BASE asset → trừ thẳng vào số coin bán được.
    - Phí trả bằng QUOTE (USDT…) hoặc BNB → số coin giữ nguyên (chỉ ảnh hưởng số dư quote).
    - Cộng dồn commission của MỌI fill (mỗi fill có thể có mức phí khác nhau).
    - KHÔNG dùng requested quantity: MARKET BUY có thể khớp ít hơn đã yêu cầu.

    `sellable_qty` = executedQty - fee_qty_khi_fee_là_base. Đây là số coin lệnh này thực sản
    sinh ra, an toàn để đặt OCO; việc kẹp theo free balance thật làm ở tầng attach_spot_oco
    vì nhiều plan có thể dùng chung balance (phần đã khóa bởi OCO khác không bán được).
    """
    executed = float(order.get("executedQty") or 0)
    fills = order.get("fills") or []
    fee_qty = 0.0
    fee_asset: str | None = None
    for fill in fills:
        commission = float(fill.get("commission") or 0)
        asset = str(fill.get("commissionAsset") or "").upper()
        if commission > 0 and asset == (base_asset or "").upper():
            fee_qty += commission
            fee_asset = asset
        elif commission > 0 and asset and fee_asset is None:
            # Ghi nhận asset phí khác để ledger biết, nhưng KHÔNG trừ vào base qty.
            fee_asset = asset
    sellable = max(0.0, executed - fee_qty)
    return executed, fee_qty, fee_asset, sellable


def _floor_to(value: float, step: float) -> float:
    if step <= 0:
        return value
    return round(math.floor(value / step + 1e-9) * step, 10)


def _fmt(value: float) -> str:
    s = f"{value:.8f}".rstrip("0").rstrip(".")
    return s or "0"


# ─── Tỷ lệ chốt lời + phí theo coin ─────────────────────────────────────────

# TP1 chốt 70%, TP2 chốt 30% (chỉ khi plan có tp2). Đổi bằng biến TP1_SHARE.
def _tp1_share_from_env() -> float:
    """TP1_SHARE phải nằm trong (0, 1] — ngoài khoảng đó chia qty sẽ ra TP1=0 hoặc
    TP1 qty > position, cả hai đều là lỗi ETH hay BTC thật."""
    try:
        value = float(os.getenv("TP1_SHARE", "0.7"))
    except (TypeError, ValueError):
        _safe_log("[CONFIG] TP1_SHARE khong phai so — dung mac dinh 0.7")
        return 0.7
    if not (0.0 < value <= 1.0):
        _safe_log(f"[CONFIG] TP1_SHARE={value} ngoai khoang (0, 1] — dung mac dinh 0.7")
        return 0.7
    return value


TP1_SHARE = _tp1_share_from_env()


def split_tp_qty(qty: float, step: float) -> tuple[float, float]:
    """Chia qty thành (qty_tp1, qty_tp2) theo TP1_SHARE.

    - Tổng LUÔN bằng qty (tp2 = qty - tp1) → không làm lệch vị thế, không bỏ sót Coin.
    - Cả 2 đều phải là bội của stepSize: không thì exchange trả -1111 "Precision is over
      the maximum defined for this asset" (đã gặp thật với 0.0066 khi step=0.001).
    - qty quá nhỏ không chia được (tp1 về 0) → trả (qty, 0.0) để dồn hết vào TP1.
    """
    if qty <= 0:
        return qty, 0.0
    # Tự floor theo step: 0.0075 với step 0.001 không phải bội → tách ra 0.0025 (sai precision,
    # exchange trả -1111). Làm tròn 10 chữ số để hết số râu nổi (0.023-0.016 = 0.006999999...).
    qty = round(_floor_to(qty, step), 10) if step > 0 else round(qty, 10)
    q1 = round(_floor_to(qty * TP1_SHARE, step), 10)
    if q1 <= 0 or (step > 0 and q1 < step):
        return qty, 0.0
    q2 = round(qty - q1, 10)  # qty và q1 đều bội step → q2 cũng bội step
    if q2 <= 0:
        return qty, 0.0
    return q1, q2


def _bump_plan_id(plan_id: str) -> str:
    """futu-eth-3 -> futu-eth-4 (chống trùng clientOrderId với phiên trước còn lệnh treo)."""
    head, _, tail = plan_id.rpartition("-")
    try:
        return f"{head}-{int(tail) + 1}"
    except ValueError:
        return f"{plan_id}-2"


def _cancel_entry_soft(base: str, api_key: str, secret: str, symbol: str, plan_id: str) -> None:
    """Hủy best-effort lệnh entry theo clientOrderId khi không chắc request đã tới exchange hay chưa.
    Mọi lỗi đều bị nuốt — đây chỉ là bước dọn dẹp thêm, không được che lỗi gốc."""
    try:
        signed_request(base, api_key, secret, "DELETE", "/fapi/v1/order",
                       {"symbol": symbol, "origClientOrderId": f"{plan_id}-e"})
    except Exception:
        pass


def _cancel_entry_soft_spot(base: str, api_key: str, secret: str, symbol: str, plan_id: str) -> None:
    """Tương đương _cancel_entry_soft cho spot (endpoint /api/v3/order)."""
    try:
        signed_request(base, api_key, secret, "DELETE", "/api/v3/order",
                       {"symbol": symbol, "origClientOrderId": f"{plan_id}-e"})
    except Exception:
        pass


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
    tp2: float | None = None,
) -> dict:
    """LIMIT entry + TP/SL (algo) đặt ngay, không chờ khớp. Trả dict ids đã dùng.

    - Không có tp2: TP1 = toàn bộ qty (hành vi cũ).
    - Có tp2: TP1 chốt TP1_SHARE (70%), TP2 chốt phần còn lại (30%), SL = toàn bộ qty.
      SL để nguyên 100% là an toàn: đã test trên demo-fapi (hedge mode), algo STOP_MARKET
      với qty LỚN HƠN position vẫn được chấp nhận → sau khi TP1 khớp 1 phần, SL vẫn đóng
      hết phần còn lại.
    - Phí futures tính bằng USDT (không trừ vào qty) → không ảnh hưởng số lượng đặt TP/SL
      (đã xác nhận: đặt market 0.022 → positionAmt = 0.022 đúng bằng).
    """
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
    # Chia sẵn 70/30 — làm tròn theo stepSize ở đây để khỏi gặp -1111 khi gửi lệnh.
    qty_tp1, qty_tp2 = split_tp_qty(qty, flt["step"])
    has_tp2 = tp2 is not None and qty_tp2 > 0
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

    # One-way mode chỉ có một net position cho mỗi symbol. Không được để một
    # plan SHORT mới vô tình giảm/reverse tổng LONG của các plan khác (hoặc
    # ngược lại). Hedge Mode mới cho phép LONG và SHORT cùng tồn tại.
    if not hedge:
        try:
            positions = signed_request(base, api_key, secret, "GET", "/fapi/v2/positionRisk",
                                       {"symbol": symbol})
            positions = positions.get("data") if isinstance(positions, dict) else positions
            if isinstance(positions, dict):
                positions = [positions]
            net_qty = 0.0
            for pos in positions or []:
                if pos.get("symbol") == symbol and str(pos.get("positionSide") or "BOTH") == "BOTH":
                    net_qty += float(pos.get("positionAmt") or 0)
            if (direction == "LONG" and net_qty < 0) or (direction == "SHORT" and net_qty > 0):
                raise ExecutorError(
                    -4061,
                    f"không đặt {direction}: account đang One-way và {symbol} có vị thế ngược chiều "
                    f"({net_qty:g}); bật Hedge Mode hoặc đóng vị thế trước",
                )
        except ExecutorError:
            raise
        except Exception as exc:
            raise ExecutorError(-1000, f"không xác định được vị thế One-way của {symbol}: {exc}")

    side = "BUY" if direction == "LONG" else "SELL"
    close_side = "SELL" if side == "BUY" else "BUY"
    pos_side = direction if hedge else "BOTH"

    # Entry limit — chống trùng clientOrderId bằng cách tăng số thứ tự plan.
    used_plan = plan_id
    entry_resp = None
    entry_time_ms = int(time.time() * 1000)
    for _ in range(6):
        try:
            entry_resp = signed_request(base, api_key, secret, "POST", "/fapi/v1/order", {
                "symbol": symbol, "side": side, "type": "LIMIT",
                "timeInForce": "GTC", "quantity": _fmt(qty), "price": _fmt(price),
                "positionSide": pos_side,
                "newClientOrderId": f"{used_plan}-e",
            })
            break
        except ExecutorError as exc:
            if exc.code == -2010 and "duplicate" in exc.msg.lower():
                # LIMIT entry thì gửi lại bằng coid mới không tạo position ngay (chưa khớp),
                # nhưng vẫn có thể để lệnh treo MỒ CÔI. Hỏi sàn lệnh cũ trước: có rồi thì dùng
                # nó, không thì mới đổi coid.
                existing = None
                try:
                    existing = signed_request(base, api_key, secret, "GET", "/fapi/v1/order", {
                        "symbol": symbol, "origClientOrderId": f"{used_plan}-e",
                    })
                except ExecutorError:
                    existing = None
                if existing:
                    print(f"[FUTURES_LIMIT] {used_plan} coid da ton tai — dung lai lenh cu "
                          f"{existing.get('orderId')} thay vi treo lenh moi", flush=True)
                    entry_resp = existing
                    break
                used_plan = _bump_plan_id(used_plan)
                continue
            # Lỗi khác (mạng/hết quyền/sai tham số): request CÓ THỂ đã tới exchange
            # (vd timeout ở bước gửi) → hủy best-effort theo clientOrderId rồi ném tiếp,
            # để không treo entry mồ côi mà ta không hề biết.
            if exc.code == -1000:
                _cancel_entry_soft(base, api_key, secret, symbol, used_plan)
            raise
    if entry_resp is None:
        raise ExecutorError(-2010, "không tạo được entry sau 6 lần chống trùng clientOrderId")

    def _algo(order_type: str, trigger: float, coid: str, algo_qty: float) -> int:
        params = {
            "symbol": symbol, "side": close_side, "type": order_type,
            "algoType": "CONDITIONAL",
            "triggerPrice": _fmt(_floor_to(trigger, flt["tick"])),
            "quantity": _fmt(algo_qty),
            "positionSide": pos_side,
            "workingType": "CONTRACT_PRICE",
            "clientAlgoId": coid,
        }
        if not hedge:
            params["reduceOnly"] = "true"
        resp = signed_request(base, api_key, secret, "POST", "/fapi/v1/algoOrder", params)
        # Response thiếu `algoId` (lỗi bất định từ sàn/proxy) phải ném ExecutorError để
        # nhánh dọn orphan bên dưới kịp gỡ các algo đã tạo được. KeyError trần sẽ thoát
        # khỏi hàm và bỏ qua toàn bộ dọn dẹp → entry treo mồ côi không TP/SL.
        algo_id = (resp or {}).get("algoId")
        if not algo_id:
            raise ExecutorError(-1000, "algoOrder không trả algoId — không thể đặt TP/SL")
        return int(algo_id)

    tp_algo = None
    tp2_algo = None
    sl_algo = None
    try:
        # TP1 chốt TP1_SHARE (70%), TP2 chốt phần còn lại (30%), SL = TOÀN BỘ qty.
        # Không có tp2 → TP1 phải là TOÀN BỘ qty (không được chia 70% rồi bỏ 30% lại trôi).
        # SL để nguyên 100%: nếu TP1 khớp trước thì còn 30% position, algo STOP_MARKET với
        # qty lớn hơn position vẫn được sàn chấp nhận (đã test trên demo-fapi) nên vẫn đóng hết.
        tp_algo = _algo("TAKE_PROFIT_MARKET", tp1, f"{used_plan}-tp", qty_tp1 if has_tp2 else qty)
        if has_tp2:
            tp2_algo = _algo("TAKE_PROFIT_MARKET", float(tp2), f"{used_plan}-tp2", qty_tp2)
        sl_algo = _algo("STOP_MARKET", sl, f"{used_plan}-sl", qty)
    except ExecutorError as exc:
        # Entry có thể đã khớp ngay trong khoảng thời gian chớp nhoáng trước khi TP/SL fail
        # (giá chạy xuyên qua vùng entry) → hủy không được nữa. Hỏi tiếp exchange: entry đã khớp
        # bao nhiêu? Nếu đã khớp thì ĐÓNG NGAY theo thị trường — không bao giờ để position trần
        # mà không có TP/SL. Nếu chưa khớp → hủy entry, không treo lệnh mồ côi.

        # Gỡ algo đã tạo được (TP đặt xong mà SL fail) — không gỡ thì nó vẫn ARMED và sẽ
        # tự kích hoạt sau, đóng position tương lai của user trên cùng symbol một cách âm thầm.
        for orphan_algo in (tp_algo, tp2_algo, sl_algo):
            if not orphan_algo:
                continue
            try:
                signed_request(base, api_key, secret, "DELETE", "/fapi/v1/algoOrder",
                               {"symbol": symbol, "algoId": orphan_algo})
            except Exception:
                pass

        executed = 0.0
        try:
            info = signed_request(base, api_key, secret, "GET", "/fapi/v1/order",
                                  {"symbol": symbol, "orderId": entry_resp.get("orderId")})
            executed = float(info.get("executedQty") or 0)
        except Exception:
            executed = 0.0
        if executed > 0:
            # Hủy phần CHƯA khớp của entry TRƯỚC khi đóng vị thế đã khớp — nếu bỏ qua,
            # phần còn lại của LIMIT GTC sẽ khớp sau và tạo position không có TP/SL.
            remainder_left = executed + 1e-12 < qty
            if remainder_left:
                try:
                    signed_request(base, api_key, secret, "DELETE", "/fapi/v1/order",
                                   {"symbol": symbol, "orderId": entry_resp.get("orderId")})
                except Exception:
                    remainder_left = False
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
                    "thị trường (không gắn được TP/SL nên không giữ vị thế trần)"
                    + ("" if remainder_left else " — ⚠️ KHÔNG hủy được phần entry còn lại, "
                                                   "kiểm tra trên GUI")
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
        "entry_time_ms": entry_time_ms,
        "tp_algo_id": tp_algo,
        "tp2_algo_id": tp2_algo,
        "sl_algo_id": sl_algo,
        "tp_client_algo_id": f"{used_plan}-tp",
        "tp2_client_algo_id": (f"{used_plan}-tp2" if has_tp2 else None),
        "sl_client_algo_id": f"{used_plan}-sl",
        "qty": qty,
        "qty_tp1": (qty_tp1 if has_tp2 else qty),
        "qty_tp2": (qty_tp2 if has_tp2 else 0.0),
        "entry_price": price,
        "hedge": hedge,
    }


# ─── Spot ────────────────────────────────────────────────────────────────────

def _oco_leg_id(oco: dict, client_order_id: str) -> int | None:
    """Lấy orderId của 1 leg trong OCO theo clientOrderId (chính xác, không đoán theo thứ tự)."""
    for order in oco.get("orders") or []:
        if str(order.get("clientOrderId") or "") == client_order_id:
            return order.get("orderId")
    for report in oco.get("orderReports") or []:
        if str(report.get("clientOrderId") or "") == client_order_id:
            return report.get("orderId")
    return None


def _post_oco(base: str, api_key: str, secret: str, symbol: str, *, tp_price: float,
              sl_price: float, tp_coid: str, sl_coid: str, qty_sell: float, flt: dict,
              list_coid: str | None = None) -> dict:
    """Gửi lệnh OCO qua endpoint MỚI POST /api/v3/orderList/oco (từ 2024-04-02).

    Endpoint cũ POST /api/v3/order/oco đã deprecated; chỉ dùng lại làm fallback khi
    HTTP 404/405 (endpoint không tồn tại → chắc chắn không sinh OCO trùng).

    `list_coid` (listClientOrderId) cho phép tra cứu cả ORDER LIST theo client id —
    restart recovery dùng nó để tìm OCO đã tồn tại mà không cần biết orderListId.
    """
    qty_text = _fmt(qty_sell)
    tp_text = _fmt(_floor_to(tp_price, flt["tick"]))
    sl_text = _fmt(_floor_to(sl_price, flt["tick"]))
    try:
        params = {
            "symbol": symbol, "side": "SELL", "quantity": qty_text,
            # LIMIT_MAKER cho TP (giá limit phía trên) + STOP_LOSS cho SL (market khi chạm).
            "aboveType": "LIMIT_MAKER",
            "abovePrice": tp_text,
            "aboveClientOrderId": tp_coid,
            "belowType": "STOP_LOSS",
            "belowStopPrice": sl_text,
            "belowClientOrderId": sl_coid,
        }
        if list_coid:
            params["listClientOrderId"] = list_coid
        oco = signed_request(base, api_key, secret, "POST", "/api/v3/orderList/oco", params)
    except ExecutorError as oco_exc:
        if oco_exc.code not in (404, 405):
            raise
        print(
            f"[SPOT_OCO] endpoint mới /api/v3/orderList/oco không tồn tại ({oco_exc.code}) "
            "→ thử endpoint cũ /api/v3/order/oco",
            flush=True,
        )
        oco = signed_request(base, api_key, secret, "POST", "/api/v3/order/oco", {
            "symbol": symbol, "side": "SELL", "quantity": qty_text,
            "price": tp_text,
            "stopPrice": sl_text,
            "limitClientOrderId": tp_coid,
            "stopClientOrderId": sl_coid,
        })
    # Giữ nguyên tên key "oco_list_id" — analyze.py đọc key này để ghi tp_algo_id vào ledger.
    return {
        "oco_list_id": oco.get("orderListId"),
        "list_client_order_id": oco.get("listClientOrderId") or list_coid,
        "tp_leg_order_id": _oco_leg_id(oco, tp_coid),
        "sl_leg_order_id": _oco_leg_id(oco, sl_coid),
    }


def _market_sell(base: str, api_key: str, secret: str, symbol: str, qty: float,
                 coid: str) -> bool:
    """Bán ngay thị trường; trả True nếu lệnh đi qua được sàn."""
    for _ in range(3):
        try:
            signed_request(base, api_key, secret, "POST", "/api/v3/order", {
                "symbol": symbol, "side": "SELL", "type": "MARKET",
                "quantity": _fmt(qty), "newClientOrderId": coid,
            })
            return True
        except Exception:
            time.sleep(1)
    return False


def _ensure_oco_room(base: str, api_key: str, secret: str, symbol: str, flt: dict,
                     need: int, keep_plan_id: str | None = None) -> list[dict]:
    """Giải phóng chỗ OCO nếu sắp vượt MAX_NUM_ORDER_LISTS của symbol.

    Spot giới hạn số OCO list đồng thời mỗi symbol (ETHUSDT demo = 20). Plan có tp2 cần
    2 OCO nên 10 plan là đầy. Khi đầy, thay vì để lệnh tạo mới bị sàn từ chối, bot hủy
    OCO CŨ NHẤT do chính bot tạo (nhận diện qua listClientOrderId) để nhường chỗ.

    Chỉ hủy OCO do bot tạo — OCO user tự đặt trên GUI không bao giờ bị đụng tới.
    Trả danh sách OCO đã hủy (để log rõ hành động).
    """
    limit = flt.get("max_order_lists")
    if not limit:
        return []  # không công bố giới hạn → không thể suy luận an toàn, bỏ qua
    try:
        open_lists = signed_request(base, api_key, secret, "GET", "/api/v3/openOrderList",
                                    {"symbol": symbol})
    except Exception as exc:
        print(f"[SPOT_OCO_ROOM] khong doc duoc openOrderList {symbol}: {exc}", flush=True)
        return []
    if not isinstance(open_lists, list):
        return []

    # Chỉ để ý OCO của bot: listClientOrderId do `next_session_plan_id` sinh ra.
    ours = [o for o in open_lists
            if _is_bot_oco(o.get("listClientOrderId"), keep_plan_id)]
    others = len(open_lists) - len(ours)
    if len(ours) + others + need <= limit:
        return []  # còn chỗ

    # Không bao giờ hủy OCO đang ARMED (`listOrderStatus == "EXECUTING"`): nó đang bảo vệ
    # position thật, hủy là để coin trần. Chỉ thu dọn OCO đã xong/hết hạn (`ALL_DONE`,
    # `REJECT`, thiếu trạng thái) — những cái đó không còn bảo vệ gì nữa.
    def _is_armed(oco: dict) -> bool:
        return str(oco.get("listOrderStatus") or "").upper() == "EXECUTING"

    dead = [o for o in ours if not _is_armed(o)]
    live = [o for o in ours if _is_armed(o)]

    # Cần hủy bớt: xếp OCO đã xong theo thời gian, CŨ NHẤT trước.
    def _age(o: dict) -> float:
        for k in ("transactionTime", "workingTime", "transactTime"):
            raw = o.get(k)
            if raw:
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    pass
        # Không có thời gian → coi như CŨ NHẤT (hủy trước), nhưng ghi log rõ.
        return 0.0

    dead.sort(key=_age)
    to_free = (len(ours) + others + need) - limit
    canceled: list[dict] = []
    for oco in dead[:max(0, to_free)]:
        list_id = oco.get("orderListId")
        coid = oco.get("listClientOrderId")
        if not list_id and not coid:
            continue
        try:
            spot_cancel_oco(base, api_key, secret, symbol, order_list_id=list_id,
                            list_client_order_id=coid)
            canceled.append({"kind": "evicted_oco", "symbol": symbol,
                             "list_id": list_id, "list_client_order_id": coid,
                             "age_ms": _age(oco)})
            print(f"[SPOT_OCO_ROOM] het cho OCO tren {symbol} (trần {limit}) → "
                  f"huy OCO cu {coid} de nhuong cho moi", flush=True)
        except Exception as exc:
            print(f"[SPOT_OCO_ROOM] khong huy duoc OCO cu {coid}: {exc}", flush=True)

    # Vẫn thiếu chỗ mà chỉ còn OCO đang ARMED → KHÔNG hủy (sẽ để coin trần).
    # Báo lỗi rõ ràng để user tự dọn bằng /offspot, an toàn hơn nhiều.
    remaining = (len(ours) + others + need) - len(canceled)
    if remaining > limit and live:
        base_asset = (symbol[:-4] if symbol.endswith("USDT") else symbol).upper()
        raise ExecutorError(
            -2010,
            f"symbol {symbol} đã đạt trần {limit} OCO đồng thời "
            f"(hiện có {len(live)} OCO đang bảo vệ vị thế thật) và không có OCO nào đã xong "
            f"để thu dọn. Gõ /offspot {base_asset} để tắt phiên cũ trước khi bật plan mới."
        )
    return canceled


def _is_bot_oco(list_client_order_id, keep_plan_id: str | None) -> bool:
    """OCO có phải do bot tạo không (không phải OCO user tự đặt trên GUI)."""
    if not list_client_order_id:
        return False
    coid = str(list_client_order_id)
    # Không bao giờ hủy OCO của plan ĐANG được gắn.
    if keep_plan_id and coid == f"{keep_plan_id}-list":
        return False
    # Mọi OCO bot đều kết thúc bằng "-list" hoặc "-list2".
    return coid.endswith("-list") or coid.endswith("-list2")


def attach_spot_oco(base: str, keys: tuple[str, str], symbol: str, tp1: float, sl: float,
                    plan_id: str, qty_filled: float, tp2: float | None = None) -> dict:
    """Gắn TP+SL (OCO) cho số coin spot ĐÃ mua; gắn không được → BÁN NGAY thị trường.

    - Không có tp2: 1 OCO (TP = tp1, SL = sl) cho toàn bộ số bán được.
    - Có tp2: 2 OCO riêng — OCO-A chốt TP1_SHARE (70%) ở tp1, OCO-B chốt 30% ở tp2,
      cả 2 đều kèm SL. Nếu gộp vào 1 OCO thì toàn bộ vị thế sẽ đi qua 1 mức giá và
      không còn khái niệm thoát 70/30.

    Sàn trừ phí mua thẳng vào coin nên số coin THẬT free luôn nhỏ hơn qty lý thuyết →
    luôn lấy min(qty, free) rồi floor theo stepSize, không thì sàn trả -2010.
    """
    api_key, secret = keys
    flt = _market_filters(base, api_key, secret, symbol, spot=True)
    free = _free_base_qty(base, api_key, secret, flt)
    available = _floor_to(min(qty_filled, free), flt["step"])
    min_qty = max(float(flt.get("min_qty") or 0), flt["step"])
    if available < min_qty:
        # Không đủ coin để bán → báo rõ và KHÔNG market-sell, vì market-sell cũng bị từ chối
        # đúng lý do đó (tránh bịa thông báo "đã bán ngay" sai).
        raise ExecutorError(
            -4164,
            f"không đủ coin để gắn TP/SL: free={_fmt(free)} "
            f"{flt.get('base_asset') or '?'} (cần tối thiểu {_fmt(min_qty)})",
        )

    q1, q2 = (split_tp_qty(available, flt["step"]) if tp2 is not None else (available, 0.0))
    min_notional = flt.get("min_notional") or 0
    # Mỗi OCO cũng phải vượt min_notional; nếu phần chia quá nhỏ thì gộp về 1 OCO tại tp1.
    if q2 > 0 and min_notional and (q1 * tp1 < min_notional or q2 * float(tp2) < min_notional):
        q1, q2 = available, 0.0

    # Spot giới hạn số OCO list đồng thời mỗi symbol. Nếu sắp đầy thì hủy OCO CŨ NHẤT
    # do bot tạo để nhường chỗ — tránh lệnh bị sàn từ chối vì -2010/limit.
    try:
        _ensure_oco_room(base, api_key, secret, symbol, flt, need=(2 if q2 > 0 else 1),
                         keep_plan_id=plan_id)
    except Exception as exc:
        # Không dọn được chỗ thì vẫn thử tạo; nếu sàn từ chối sẽ rơi vào nhánh
        # market-sell bảo vệ vị thế bên dưới.
        print(f"[SPOT_OCO_ROOM] {plan_id} khong giai phong duoc cho: {exc}", flush=True)

    try:
        first = _post_oco(base, api_key, secret, symbol,
                          tp_price=tp1, sl_price=sl,
                          tp_coid=f"{plan_id}-tp", sl_coid=f"{plan_id}-sl",
                          qty_sell=q1, flt=flt, list_coid=f"{plan_id}-list")
    except ExecutorError as oco_exc:
        if oco_exc.code == -2010 and "duplicate" in oco_exc.msg.lower():
            # Coid da ton tai → tren san DA co OCO cua plan nay (safety net doc lech truoc do).
            # Khong duoc ban: ban luc nay la ban sai vi the da duoc bao ve.
            print(f"[SPOT_OCO] {plan_id} da co OCO tren san (duplicate) — bỏ qua, không bán",
                  flush=True)
            return {"oco_list_id": None, "qty_tp1": available, "qty_tp2": 0.0,
                    "tp2_oco_list_id": None, "already_exists": True,
                    "list_client_order_id": f"{plan_id}-list"}
        if _market_sell(base, api_key, secret, symbol, available, f"{plan_id}-mc"):
            raise ExecutorError(
                oco_exc.code,
                f"{oco_exc.msg} — đã mua {_fmt(available)} {symbol} nhưng gắn TP/SL thất bại → "
                "ĐÃ BÁN NGAY theo thị trường (không giữ vị thế trần)",
            )
        raise ExecutorError(
            oco_exc.code,
            f"{oco_exc.msg} — ⚠️⚠️ đã mua {_fmt(available)} {symbol} NHƯNG ĐÓNG THẤT BẠI sau 3 lần "
            "thử — VÀO GUI ĐÓNG NGAY, vị thế đang trần!",
        )

    result = dict(first)
    result["qty_tp1"] = q1
    result["qty_tp2"] = q2
    result["tp2_oco_list_id"] = None
    result["tp2_list_client_order_id"] = None
    if q2 > 0:
        # OCO-B cho phần 30%. Nếu thất bại thì PHẢI đóng ngay phần đó, nếu không nó nằm
        # trần (OCO-A chỉ bảo vệ 70%). Phần 70% đã có OCO-A → giữ nguyên, không bán cả 2.
        try:
            second = _post_oco(base, api_key, secret, symbol,
                               tp_price=float(tp2), sl_price=sl,
                               tp_coid=f"{plan_id}-tp2", sl_coid=f"{plan_id}-sl2",
                               qty_sell=q2, flt=flt, list_coid=f"{plan_id}-list2")
            result["tp2_oco_list_id"] = second.get("oco_list_id")
            result["tp2_list_client_order_id"] = second.get("list_client_order_id")
            result["tp2_leg_order_id"] = second.get("tp_leg_order_id")
        except ExecutorError as tp2_exc:
            print(f"[SPOT_OCO] gan OCO tp2 that bai: {tp2_exc} — ban phan 30% de khong tran",
                  flush=True)
            if not _market_sell(base, api_key, secret, symbol, q2, f"{plan_id}-mc2"):
                raise ExecutorError(
                    tp2_exc.code,
                    f"{tp2_exc.msg} — ⚠️⚠️ đã mua {symbol} NHƯNG không gắn OCO cho tp2 "
                    "VÀ KHÔNG ĐÓNG ĐƯỢC phần 30% — VÀO GUI ĐÓNG NGAY!",
                )
            result["tp2_closed"] = True
    return result


def spot_pending_state(base: str, api_key: str, secret: str, symbol: str, plan_id: str) -> dict:
    """Trạng thái pending TP/SL (cua OCO/OTOCO) tren san: {"alive", "list_id"}.

    Không tin vào cột tp_algo_id trong DB: OTOCO có thể bị exchange bỏ qua pending khi
    khớp (do phí), hoặc watcher chết khi restart → DB ghi "đã bảo vệ" trong khi trên sàn
    không có gì. Câu hỏi này trả lời đúng sự thật trên sàn.
    """
    try:
        info = signed_request(base, api_key, secret, "GET", "/api/v3/order",
                              {"symbol": symbol, "origClientOrderId": f"{plan_id}-tp"})
    except ExecutorError as exc:
        if exc.code == -2011:  # khong tim thay → pending khong ton tai
            return {"alive": False, "list_id": None}
        # Loi khac (mang...) → khong ket luan duoc; bao la con de khong gan OCO trung.
        return {"alive": True, "list_id": None}
    except Exception:
        return {"alive": True, "list_id": None}
    return {"alive": True, "list_id": info.get("orderListId")}


def spot_find_oco(base: str, api_key: str, secret: str, symbol: str,
                  list_client_order_id: str) -> dict | None:
    """Tìm OCO theo listClientOrderId (GET /api/v3/orderList + origClientOrderId).

    Restart recovery dùng: DB có listClientOrderId nhưng thiếu orderListId (process chết
    giữa lúc OCO đã tạo nhưng chưa kịp ghi DB) → tìm lại orderListId, KHÔNG tạo OCO trùng.
    Trả dict OCO hoặc None nếu không tồn tại.
    """
    try:
        info = signed_request(base, api_key, secret, "GET", "/api/v3/orderList",
                              {"symbol": symbol, "origClientOrderId": list_client_order_id})
    except ExecutorError as exc:
        if exc.code in (-2011, 404):  # không tồn tại
            return None
        raise
    except Exception:
        return None
    return info if isinstance(info, dict) and info.get("orderListId") else None


def spot_cancel_oco(base: str, api_key: str, secret: str, symbol: str,
                    order_list_id=None, list_client_order_id: str | None = None) -> dict:
    """Hủy đúng 1 OCO theo orderListId ƯU TIÊN, fallback listClientOrderId.

    KHÔNG hủy theo symbol tổng thể — làm vậy sẽ xóa luôn OCO các plan khác cùng symbol
    đang bảo vệ position thật của user.
    """
    params = {"symbol": symbol}
    if order_list_id:
        params["orderListId"] = order_list_id
    elif list_client_order_id:
        params["listClientOrderId"] = list_client_order_id
    else:
        raise ExecutorError(-1100, "spot_cancel_oco cần orderListId hoặc listClientOrderId")
    return signed_request(base, api_key, secret, "DELETE", "/api/v3/orderList", params)


def spot_order_state(base: str, keys: tuple[str, str], symbol: str,
                     client_order_id: str) -> dict:
    """Trạng thái lệnh spot theo clientOrderId.

    Dùng cho recovery và vòng chờ khớp: query lại MARKET BUY cũ bằng newClientOrderId
    trước khi dám làm gì khác — tránh mua double khi không rõ lệnh đã lên sàn chưa.
    Trả về cả executed_qty/fee/sellable_qty để khôi phục protection cho đúng số coin.
    """
    api_key, secret = keys
    info = signed_request(base, api_key, secret, "GET", "/api/v3/order",
                          {"symbol": symbol, "origClientOrderId": client_order_id})
    executed, fee_qty, fee_asset, sellable = spot_fill_fees(
        info, _market_filters(base, api_key, secret, symbol, spot=True).get("base_asset") or "")
    return {"status": info.get("status") or "UNKNOWN", "executed": executed,
            "fee_qty": fee_qty, "fee_asset": fee_asset, "sellable_qty": sellable,
            "fills": info.get("fills")}


def futures_entry_state(base: str, keys: tuple[str, str], symbol: str, order_id) -> dict:
    """Trạng thái lệnh entry futures theo orderId: {"status", "executed"}."""
    api_key, secret = keys
    info = signed_request(base, api_key, secret, "GET", "/fapi/v1/order",
                          {"symbol": symbol, "orderId": order_id})
    return {"status": info.get("status") or "UNKNOWN",
            "executed": float(info.get("executedQty") or 0)}


def has_open_futures_position(keys: tuple[str, str], symbol: str) -> bool | None:
    """Symbol đang có vị thế futures nào không. None nếu không đọc được (không phán đoán)."""
    base, (api_key, secret) = FUTURES_API_BASE, keys
    try:
        rows = signed_request(base, api_key, secret, "GET", "/fapi/v2/positionRisk",
                              {"symbol": symbol})
    except Exception:
        return None
    items = rows.get("data") if isinstance(rows, dict) else rows
    if isinstance(items, dict):
        items = [items]
    for row in items or []:
        if row.get("symbol") != symbol:
            continue
        try:
            if abs(float(row.get("positionAmt") or 0)) > 0:
                return True
        except Exception:
            return None
    return False


def cancel_futures_algo_orders(keys: tuple[str, str], symbol: str, algo_ids) -> list[dict]:
    """Hủy các lệnh algo (TP/SL) còn treo của futures.

    Binance KHÔNG tự hủy TP/SL khi position đóng (hạn chế nội bộ sàn — đã có nhiều thread
    xác nhận). Nếu bỏ mặc định thì SL còn treo sẽ đóng cả vị thế MỚI mà user mở sau đó
    trên cùng symbol, đúng tại mức SL của lệnh cũ.
    """
    base, (api_key, secret) = FUTURES_API_BASE, keys
    cancelled: list[dict] = []
    for algo_id in algo_ids:
        if not algo_id:
            continue
        try:
            signed_request(base, api_key, secret, "DELETE", "/fapi/v1/algoOrder",
                           {"symbol": symbol, "algoId": algo_id})
            cancelled.append({"kind": "algo", "symbol": symbol, "algo_id": algo_id})
        except ExecutorError as exc:
            if exc.code not in (-2011, 404):  # -2011 = đã không còn → không phải lỗi
                print(f"[CANCEL_ALGO] khong huy duoc {algo_id} cua {symbol}: {exc}", flush=True)
        except Exception as exc:
            print(f"[CANCEL_ALGO] khong huy duoc {algo_id} cua {symbol}: {exc}", flush=True)
    return cancelled


def open_futures_algo_orders(keys: tuple[str, str], symbol: str) -> list[dict] | None:
    """Đọc các conditional algo đang mở của đúng symbol.

    ``None`` nghĩa là không xác định được trạng thái (lỗi mạng/response lạ), khác
    với ``[]`` là đã đọc thành công và không còn algo mở. Caller không được coi
    ``None`` là danh sách rỗng để tránh hủy nhầm protection.
    """
    api_key, secret = keys
    try:
        raw = signed_request(FUTURES_API_BASE, api_key, secret,
                             "GET", "/fapi/v1/openAlgoOrders", {"symbol": symbol})
    except Exception as exc:
        print(f"[FUTURES_ALGO_READ] khong doc duoc algo mo {symbol}: {exc}", flush=True)
        return None
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("data"), list):
        return raw["data"]
    # Một response dict không có data không đủ căn cứ để kết luận không có lệnh.
    return None



def place_spot_plan(
    symbol: str,
    entry_price: float,
    tp1: float,
    sl: float,
    qty: float,
    keys: tuple[str, str],
    plan_id: str,
    fill_timeout: int | None = None,
    poll_seconds: int = 2,
    tp2: float | None = None,
) -> dict:
    """Market BUY, chờ ngắn, rồi gắn OCO TP/SL cho số coin đã mua."""
    base, (api_key, secret) = SPOT_API_BASE, keys
    flt = _market_filters(base, api_key, secret, symbol, spot=True)
    qty = _floor_to(qty, flt["step"])
    price = _floor_to(entry_price, flt["tick"])
    if qty <= 0:
        raise ExecutorError(-4164, f"khối lượng {qty} bị làm tròn về 0 theo step {flt['step']}")
    if flt["min_notional"] and qty * price < flt["min_notional"]:
        raise ExecutorError(-4164, f"notional {qty * price:.4g} < tối thiểu {flt['min_notional']:g} USDT — tăng khối lượng")

    used_plan = plan_id
    entry_resp = None
    used_plan = plan_id
    entry_resp = None
    for _ in range(6):
        try:
            entry_resp = signed_request(base, api_key, secret, "POST", "/api/v3/order", {
                "symbol": symbol, "side": "BUY", "type": "MARKET",
                "quantity": _fmt(qty), "newClientOrderId": f"{used_plan}-e",
            })
            break
        except ExecutorError as exc:
            if exc.code == -2010 and "duplicate" in exc.msg.lower():
                # Coid đã tồn tại = lệnh TRƯỚC ĐÓ đã được sàn nhận (thường do response bị mất).
                # TUYỆT ĐỐI không gửi lại MARKET BUY với coid mới → sẽ MUA THÊM lần nữa.
                # Hỏi sàn lệnh cũ rồi dùng nó.
                existing = None
                try:
                    existing = signed_request(base, api_key, secret, "GET", "/api/v3/order", {
                        "symbol": symbol, "origClientOrderId": f"{used_plan}-e",
                    })
                except ExecutorError as lookup_exc:
                    # -2011 = chắc chắn KHÔNG có lệnh cũ → đổi coid an toàn.
                    # Mọi lỗi khác (mạng, timeout, quyền...) = KHÔNG RÕ. Không được đoán:
                    # gửi lại MARKET BUY trong lúc không rõ là mua trùng lặp.
                    if lookup_exc.code == -2011:
                        existing = None  # không có lệnh cũ → xuống dưới đổi coid
                    else:
                        raise ExecutorError(
                            -1000,
                            f"coid {used_plan}-e bị trùng nhưng không xác minh được lệnh cũ "
                            f"({lookup_exc.code}: {lookup_exc.msg}) — KHÔNG đặt lại MARKET BUY "
                            "để tránh mua trùng. Vào GUI kiểm tra lệnh của plan này."
                        ) from lookup_exc
                if existing:
                    print(f"[SPOT_MARKET] {used_plan} coid da ton tai — dung lai lenh cu "
                          f"{existing.get('orderId')} thay vi mua tiep", flush=True)
                    entry_resp = existing
                    break
                # Chắc chắn không có lệnh cũ (-2011) → đổi coid.
                used_plan = _bump_plan_id(used_plan)
                continue
            if exc.code == -1000:
                _cancel_entry_soft_spot(base, api_key, secret, symbol, used_plan)
            raise
    if entry_resp is None:
        raise ExecutorError(-2010, "không tạo được market entry sau 6 lần chống trùng")

    entry_id = entry_resp.get("orderId")
    client_id = entry_resp.get("clientOrderId") or f"{used_plan}-e"
    if not entry_id:
        raise ExecutorError(-1000, "market entry tạo xong nhưng không đọc được orderId")

    if fill_timeout is None:
        fill_timeout = SPOT_FILL_TIMEOUT_SECONDS
    deadline = time.time() + max(0, fill_timeout)
    status = entry_resp.get("status") or "NEW"
    executed = float(entry_resp.get("executedQty") or 0)
    order = entry_resp
    while executed <= 0 and time.time() < deadline:
        order = signed_request(base, api_key, secret, "GET", "/api/v3/order",
                               {"symbol": symbol, "orderId": entry_id})
        status = order.get("status", status)
        executed = float(order.get("executedQty") or executed or 0)
        if status == "FILLED" or executed > 0:
            break
        if status in ("CANCELED", "EXPIRED", "REJECTED"):
            break
        time.sleep(poll_seconds)

    if executed <= 0:
        try:
            signed_request(base, api_key, secret, "DELETE", "/api/v3/order",
                           {"symbol": symbol, "orderId": entry_id})
        except Exception:
            pass
        raise ExecutorError(-1000, f"market entry {used_plan} không khớp, không thể gắn OCO")

    if SPOT_OCO_DELAY_SECONDS:
        time.sleep(SPOT_OCO_DELAY_SECONDS)

    avg_price = price
    try:
        quote_qty = float(order.get("cummulativeQuoteQty") or 0)
        if quote_qty > 0:
            avg_price = quote_qty / executed
    except (TypeError, ValueError, ZeroDivisionError):
        pass

    # Phí + số coin THỰC bán được: MARKET BUY có thể khớp ít hơn đã yêu cầu và phí trả
    # bằng base asset sẽ trừ vào coin. Nếu response lệnh thiếu `fills` thì query lại theo
    # orderId — nhưng CHỈ nhận kết quả mới khi nó thực sự có fills hoặc khớp NHIỀU HƠN,
    # tránh trường hợp response cũ đã FILLED mà GET trả thiếu dữ liệu làm mất thông tin.
    executed_qty, fee_qty, fee_asset, sellable_qty = spot_fill_fees(
        order, flt.get("base_asset") or "")
    if not order.get("fills") and executed_qty > 0:
        try:
            full = signed_request(base, api_key, secret, "GET", "/api/v3/order",
                                  {"symbol": symbol, "orderId": entry_id})
            full_executed = float(full.get("executedQty") or 0)
            if full.get("fills") or full_executed > executed_qty:
                new_exec, new_fee, new_asset, new_sell = spot_fill_fees(
                    full, flt.get("base_asset") or "")
                if new_exec > 0 or full.get("fills"):
                    executed_qty, fee_qty, fee_asset, sellable_qty = (
                        new_exec, new_fee, new_asset, new_sell)
                    quote_qty = float(full.get("cummulativeQuoteQty") or 0)
                    if quote_qty > 0 and executed_qty > 0:
                        avg_price = quote_qty / executed_qty
        except Exception:
            pass

    has_tp2 = tp2 is not None and float(tp2) > 0
    result = {
        "plan_id": used_plan, "entry_order_id": entry_id, "qty": qty,
        "entry_price": price, "avg_price": avg_price,
        "filled": status == "FILLED", "status": status, "executed": executed_qty,
        "fee_qty": fee_qty, "fee_asset": fee_asset, "sellable_qty": sellable_qty,
        "client_order_id": client_id, "oco_list_id": None,
        "tp2_oco_list_id": None, "otoco": False, "market_entry": True,
    }
    # Dùng executedQty THỰC sau phí; phí Spot làm số coin bán được nhỏ hơn qty gửi vào.
    result.update(attach_spot_oco(
        base, keys, symbol, tp1, sl, used_plan, sellable_qty,
        tp2=(float(tp2) if has_tp2 else None),
    ))
    result["protected"] = True
    return result


def place_plan(market: str, symbol: str, direction: str, entry_price: float,
               tp1: float, sl: float, qty: float, leverage: int | None,
               keys: tuple[str, str], plan_id: str, tp2: float | None = None) -> dict:
    if market == "futures":
        return place_futures_plan(symbol, direction, entry_price, tp1, sl, qty, leverage,
                                  keys, plan_id, tp2=tp2)
    return place_spot_plan(symbol, entry_price, tp1, sl, qty, keys, plan_id, tp2=tp2)


# ─── Hủy lệnh TREO khi autoscan tắt ──────────────────────────────────────────

def cancel_pending_plan_orders(keys: tuple[str, str], market: str, rows: list[dict]) -> list[dict]:
    """Hủy mọi lệnh TREO (chưa khớp) của các plan trong rows:
    - entry còn MỞ (executedQty=0) → hủy entry + TP/SL đi kèm (re-check entry trước khi
      đụng TP/SL để không bao giờ gỡ bảo vệ của position vừa khớp lén giữa chừng);
    - entry ĐÃ KHỚP HẲN (FILLED) → KHÔNG ĐỤNG gì (position + TP/SL giữ nguyên);
    - entry khớp HẦN (PARTIALLY_FILLED) → hủy phần CHƯA khớp của entry, GIỮ TP/SL vì
      position đã có; nếu coi là "filled" rồi bỏ qua thì phần còn lại vẫn treo và khớp
      sau → position nhân đôi không có bảo vệ;
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
            status = info.get("status")
            if status == "FILLED":
                return "filled"
            if executed > 0:
                return "partial"
            return "open"
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
            # Đã khớp hết (giữ nguyên) hoặc không xác định được (không đụng gì).
            continue

        if state == "partial":
            # Khớp nửa chừng: hủy phần còn lại để không bị khớp thêm, nhưng GIỮ TP/SL
            # vì position đã tồn tại.
            try:
                signed_request(base, api_key, secret, "DELETE",
                               "/fapi/v1/order" if is_futures else "/api/v3/order",
                               {"symbol": symbol, "orderId": entry_id})
                cancelled.append({"kind": "entry_remainder", "symbol": symbol, "order_id": entry_id})
            except Exception:
                pass
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
        tp2_id = row.get("tp2_algo_id")
        sl_id = row.get("sl_algo_id")
        if is_futures:
            for algo_id in (tp_id, tp2_id, sl_id):
                if not algo_id:
                    continue
                try:
                    signed_request(base, api_key, secret, "DELETE", "/fapi/v1/algoOrder",
                                   {"symbol": symbol, "algoId": algo_id})
                    cancelled.append({"kind": "algo", "symbol": symbol, "algo_id": algo_id})
                except Exception:
                    continue
        else:
            # spot: tp_algo_id / tp2_algo_id giữ oco_list_id (1 list cho TP1, 1 list cho TP2).
            # Hủy theo orderListId; nếu DB thiếu orderListId (crash trước khi ghi) thì tra lại
            # bằng listClientOrderId — KHÔNG hủy theo symbol vì sẽ xóa OCO của plan khác.
            list_coids = [row.get("list_client_order_id"), row.get("tp2_list_client_order_id")]
            for list_id, list_coid in ((tp_id, list_coids[0]), (tp2_id, list_coids[1])):
                if not list_id and not list_coid:
                    continue
                if not list_id and list_coid:
                    found = spot_find_oco(base, api_key, secret, symbol, list_coid)
                    list_id = found.get("orderListId") if found else None
                    if not list_id:
                        continue  # OCO không tồn tại nữa → không cần hủy
                try:
                    # `symbol` BẮT BUỘC theo docs (DELETE /api/v3/orderList) — thiếu là sàn
                    # từ chối và lệnh OCO vẫn treo, trong khi người dùng tưởng đã hủy.
                    signed_request(base, api_key, secret, "DELETE", "/api/v3/orderList",
                                   {"symbol": symbol, "orderListId": list_id})
                    cancelled.append({"kind": "oco", "symbol": symbol, "list_id": list_id})
                except ExecutorError as exc:
                    # -2011 = list da khong con (OTOCO bi huy kem entry o buoc tren) → khong phai loi.
                    if exc.code != -2011:
                        print(f"[CANCEL_PENDING] khong huy duoc OCO {list_id} cua {symbol}: {exc}",
                              flush=True)
                except Exception as exc:
                    print(f"[CANCEL_PENDING] khong huy duoc OCO {list_id} cua {symbol}: {exc}",
                          flush=True)
    return cancelled


def _is_bot_coid(client_order_id) -> bool:
    """clientOrderId có do bot sinh ra không (`{plan_id}-e` của entry limit futures).

    Mọi lệnh bot đặt đều kết thúc bằng `-e` (entry). User tự đặt trên GUI thường không có
    hậu tố này. Dùng để TTL cleanup CHỈ hủy lệnh của bot, không hủy lệnh của user.
    """
    coid = str(client_order_id or "")
    return bool(coid) and coid.endswith("-e")


def cancel_expired_futures_entry_orders(keys: tuple[str, str]) -> list[dict]:
    """Hủy các LIMIT entry futures đã mở quá FUTURES_ENTRY_TTL_SECONDS.

    Chỉ hủy khi entry còn MỞ (executedQty=0) VÀ chưa vượt quá TTL.
    Nếu entry đã khớp một phần/hết → KHÔNG hủy (position đã có, TP/SL cần giữ).

    Trả về danh sách lệnh đã hủy: entry + TP/SL algo đi kèm.
    """
    base, (api_key, secret) = FUTURES_API_BASE, keys
    ttl_ms = FUTURES_ENTRY_TTL_SECONDS * 1000
    now_ms = int(time.time() * 1000)
    cancelled: list[dict] = []

    # Đọc tất cả open orders futures
    try:
        open_orders = signed_request(base, api_key, secret, "GET", "/fapi/v1/openOrders", {})
    except Exception as exc:
        print(f"[CANCEL_EXPIRED] khong doc duoc open orders: {exc}", flush=True)
        return cancelled

    if not isinstance(open_orders, list):
        return cancelled

    for order in open_orders:
        if order.get("type") != "LIMIT" or order.get("timeInForce") != "GTC":
            continue
        symbol = order.get("symbol")
        order_id = order.get("orderId")
        client_order_id = order.get("clientOrderId")
        executed_qty = float(order.get("executedQty") or 0)
        update_time = order.get("updateTime") or order.get("time") or 0

        if not symbol or not order_id:
            continue

        # CHỈ động vào entry do BOT tạo (`{plan_id}-e`). User tự đặt GTC LIMIT trên GUI
        # không được bị hủy — hủy nhầm là phá risk management của chính user.
        if not _is_bot_coid(client_order_id):
            continue

        # Đã khớp một phần/hết → bỏ qua (position đã có)
        if executed_qty > 0:
            continue

        # Kiểm tra TTL
        age_ms = now_ms - int(update_time)
        if age_ms < ttl_ms:
            continue

        # Hủy entry
        try:
            signed_request(base, api_key, secret, "DELETE", "/fapi/v1/order",
                           {"symbol": symbol, "orderId": order_id})
            cancelled.append({"kind": "expired_entry", "symbol": symbol, "order_id": order_id,
                              "client_order_id": client_order_id, "age_seconds": age_ms // 1000})
        except Exception as exc:
            print(f"[CANCEL_EXPIRED] khong huy duoc entry {order_id} cua {symbol}: {exc}", flush=True)
            continue

        # Hủy TP/SL algo đi kèm (tìm theo clientAlgoId pattern plan_id-*)
        plan_prefix = client_order_id.rsplit("-", 1)[0] if client_order_id else None
        if plan_prefix:
            try:
                algos = open_futures_algo_orders(keys, symbol)
            except Exception:
                algos = None
            if algos:
                for algo in algos:
                    algo_coid = algo.get("clientAlgoId")
                    if algo_coid and algo_coid.startswith(f"{plan_prefix}-"):
                        try:
                            signed_request(base, api_key, secret, "DELETE", "/fapi/v1/algoOrder",
                                           {"symbol": symbol, "algoId": algo.get("algoId")})
                            cancelled.append({"kind": "expired_algo", "symbol": symbol,
                                              "algo_id": algo.get("algoId"), "client_algo_id": algo_coid})
                        except Exception:
                            pass
    return cancelled
