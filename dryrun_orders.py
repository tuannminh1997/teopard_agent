"""DRY-RUN đặt lệnh futures + spot với FILTER THẬT lấy từ demo-fapi / demo-api.

Không cần API key: dùng `/exchangeInfo` công khai của demo để lấy stepSize/tickSize/
minQty/minNotional THẬT, rồi chạy đúng đường code `place_futures_plan` /
`place_spot_plan` với signed_request bị chặn lại (chỉ ghi payload, không gửi).

Kiểm tra cho TẤT CẢ case:
- futures: có tp2 / không tp2 / LONG / SHORT / qty USDT nhỏ / qty USDT lớn
- spot:    có tp2 / không tp2 / phí base-asset / phí BNB / phí quote / partial fill
           / free balance thấp hơn executedQty / OCO duplicate

Mỗi case assert:
- qty entry khớp stepSize, qty*price >= minNotional
- TP1 + TP2 + SL đúng công thức 70/30/100 và KHÔNG vượt qty
- spot OCO quantity = min(executedQty-sellable, free)
"""
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

import binance_executor
from binance_executor import ExecutorError

FUTU_BASE = os.getenv("FUTURES_API_BASE")
SPOT_BASE = os.getenv("SPOT_API_BASE")

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))


def real_filters(base: str, symbol: str, spot: bool) -> dict:
    path = "/api/v3/exchangeInfo" if spot else "/fapi/v1/exchangeInfo"
    data = requests.get(base + path, params={"symbol": symbol}, timeout=20).json()
    for s in data["symbols"]:
        if s["symbol"] == symbol:
            f = {}
            for flt in s["filters"]:
                t = flt["filterType"]
                if t == "PRICE_FILTER":
                    f["tick"] = float(flt["tickSize"])
                elif t == "LOT_SIZE":
                    f["step"] = float(flt["stepSize"])
                    f["min_qty"] = float(flt["minQty"])
                elif t in ("MIN_NOTIONAL", "NOTIONAL"):
                    # Futures dung key `notional`, Spot dung `minNotional`.
                    raw = flt.get("notional") or flt.get("minNotional")
                    if raw:
                        f["min_notional"] = float(raw)
            f["base_asset"] = s["baseAsset"]
            f["max_order_lists"] = None
            for flt in s["filters"]:
                if flt["filterType"] == "MAX_NUM_ORDER_LISTS":
                    f["max_order_lists"] = flt.get("maxNumOrderLists")
            return f
    raise RuntimeError(f"khong tim thay {symbol}")


class Recorder:
    """Ghi payload mọi request; trả response tối thiểu để code đi tiếp."""

    def __init__(self):
        self.calls = []

    def __call__(self, base, api_key, secret, method, path, params):
        self.calls.append((method, path, dict(params)))
        if path.endswith("/positionSide/dual"):
            return {"dualSidePosition": True}
        if path.endswith("/leverage") or path.endswith("multiAssetsMargin"):
            return {}
        if path.endswith("/algoOrder"):
            return {"algoId": 90000 + len(self.calls)}
        if path.endswith("/fapi/v1/order") and method == "POST":
            return {"orderId": 50000 + len(self.calls),
                    "clientOrderId": params.get("newClientOrderId"),
                    "status": "NEW"}
        if path.endswith("/api/v3/order") and method == "POST" and params.get("type") == "MARKET":
            return {"orderId": 60000 + len(self.calls),
                    "clientOrderId": params.get("newClientOrderId"),
                    "status": "FILLED",
                    "executedQty": params.get("quantity"),
                    "cummulativeQuoteQty": str(
                        float(params.get("quantity") or 0) * float(self.last_price or 0))}
        if path.endswith("/orderList/oco") and method == "POST":
            return {"orderListId": 70000 + len(self.calls),
                    "listClientOrderId": params.get("listClientOrderId"),
                    "orders": [{"clientOrderId": params.get("aboveClientOrderId"),
                                "orderId": 70000 + len(self.calls)},
                               {"clientOrderId": params.get("belowClientOrderId"),
                                "orderId": 70000 + len(self.calls)}]}
        return {}

    def of(self, suffix: str, method=None, pred=None):
        out = [(m, p, prm) for m, p, prm in self.calls
               if p.endswith(suffix) and (method is None or m == method)]
        if pred:
            out = [t for t in out if pred(t[2])]
        return out

    @property
    def last_price(self):
        return getattr(self, "_lp", 0)

    last_price = 27000.0


def payloads(r: "Recorder"):
    return r


print("=" * 78)
print("PHAN 1: FILTER THAT TU DEMO")
print("=" * 78)
for sym, spot in (("ETHUSDT", False), ("ETHUSDT", True)):
    f = real_filters(FUTU_BASE if not spot else SPOT_BASE, sym, spot)
    tag = "SPOT " if spot else "FUTU "
    print(f"  {tag}{sym}: step={f['step']} tick={f['tick']} "
          f"minQty={f['min_qty']} minNotional={f['min_notional']}")

binance_executor._FILTER_CACHE.clear()


# ─── FUTURES ───────────────────────────────────────────────────────────────────

def futu_case(title: str, usdt: float, leverage: int, price: float, tp2=None,
              direction="LONG") -> None:
    print()
    print(f"--- FUTURES: {title}")
    print(f"    config: {usdt} USDT x{leverage} = vi the {usdt * leverage} USDT, "
          f"entry {price}")
    binance_executor._FILTER_CACHE.clear()
    r = Recorder()
    binance_executor.signed_request = r

    plan_entry_top = price if direction == "LONG" else price
    try:
        out = binance_executor.place_futures_plan(
            "ETHUSDT", direction, price, price * 1.04, price * 0.96, usdt * leverage / price,
            leverage, ("k", "s"), f"plan-{title}", tp2=tp2,
        )
    except ExecutorError as exc:
        check(f"{title}: khong loi", False, f"{exc.code}: {exc.msg}")
        return

    flt = real_filters(FUTU_BASE, "ETHUSDT", False)
    step, tick, mn = flt["step"], flt["tick"], flt["min_notional"]

    entry = r.of("/fapi/v1/order", "POST", lambda p: p.get("type") == "LIMIT")
    check(f"{title}: co 1 entry LIMIT", len(entry) == 1, str(len(entry)))
    if not entry:
        return
    eq, ep = float(entry[0][2]["quantity"]), float(entry[0][2]["price"])

    check(f"{title}: entry qty {eq} la boi cua step {step}",
          abs(eq / step - round(eq / step)) < 1e-9)
    check(f"{title}: entry notional {eq * ep:.2f} >= minNotional {mn}",
          eq * ep >= mn, f"{eq * ep:.2f}")
    check(f"{title}: entry notional <= vi the yeu cau {usdt * leverage}",
          eq * ep <= usdt * leverage + 1e-9, f"{eq * ep:.2f} vs {usdt * leverage}")
    check(f"{title}: qty tra ve == entry qty", out["qty"] == eq, f'{out["qty"]} vs {eq}')

    algos = r.of("/fapi/v1/algoOrder", "POST")
    tps = [t for t in algos if t[2]["type"] == "TAKE_PROFIT_MARKET"]
    sls = [t for t in algos if t[2]["type"] == "STOP_MARKET"]
    check(f"{title}: SL duy nhat 1 cai", len(sls) == 1, str(len(sls)))
    sl_qty = float(sls[0][2]["quantity"]) if sls else 0
    check(f"{title}: SL quantity {sl_qty} == position {eq}",
          abs(sl_qty - eq) < 1e-9, f"{sl_qty} vs {eq}")

    if tp2 is None:
        check(f"{title}: khong co tp2 -> dung 1 TP", len(tps) == 1, str(len(tps)))
        if tps:
            check(f"{title}: TP1 == toan bo qty {eq}",
                  abs(float(tps[0][2]["quantity"]) - eq) < 1e-9,
                  str(tps[0][2]["quantity"]))
    else:
        check(f"{title}: co tp2 -> dung 2 TP", len(tps) == 2, str(len(tps)))
        if len(tps) == 2:
            q1, q2 = float(tps[0][2]["quantity"]), float(tps[1][2]["quantity"])
            check(f"{title}: TP1+TP2 == vi the {eq}", abs(q1 + q2 - eq) < 1e-9,
                  f"{q1}+{q2}={q1 + q2}")
            check(f"{title}: TP1 la phan lon (~70%)", q1 >= q2, f"{q1} >= {q2}")
            check(f"{title}: TP1 trigger {tps[0][2]['triggerPrice']} dung gia",
                  float(tps[0][2]["triggerPrice"]) == float(f"{price * 1.04:.2f}"),
                  tps[0][2]["triggerPrice"])
            check(f"{title}: TP2 trigger dung gia",
                  float(tps[1][2]["triggerPrice"]) == float(f"{tp2:.2f}"),
                  tps[1][2]["triggerPrice"])
        check(f"{title}: TP/SL clientAlgoId khac nhau",
              len({t[2]["clientAlgoId"] for t in algos}) == len(algos))


print()
print("=" * 78)
print("PHAN 2: FUTURES — TAT CA CASE")
print("=" * 78)
P = 2700.0
futu_case("futu-usdt-khong-tp2", 10, 20, P)
futu_case("futu-usdt-co-tp2", 10, 20, P, tp2=P * 1.06)
futu_case("futu-usdt-don-bay-1", 100, 1, P)
futu_case("futu-usdt-don-bay-125", 10, 125, P, tp2=P * 1.06)
futu_case("futu-short-co-tp2", 50, 10, P, tp2=P * 0.94, direction="SHORT")
futu_case("futu-usdt-nho", 5, 5, P)
futu_case("futu-usdt-lon", 500, 20, P, tp2=P * 1.06)
# qty quá nhỏ để floor về 0 → phải raise -4164 chứ không được gửi lệnh rác
r0 = Recorder()
binance_executor.signed_request = r0
try:
    binance_executor.place_futures_plan(
        "ETHUSDT", "LONG", P, P * 1.04, P * 0.96, 0.0000001, 20, ("k", "s"), "plan-tiny")
    check("futu-usdt-duoi-1-coin: phai raise", False, "khong raise")
except ExecutorError as exc:
    check("futu-usdt-duoi-1-coin: raise -4164", exc.code == -4164, str(exc.code))
    check("futu-usdt-duoi-1-coin: khong goi san",
          not r0.of("/fapi/v1/order", "POST"), "co request!")

# floor theo stepSize lam notional rot xuoi duoi minNotional that → phai raise, khong duoc
# gui lenh ma san se tu choi -1013.
futu_flt = real_filters(FUTU_BASE, "ETHUSDT", False)
f_min_notional = futu_flt["min_notional"]
for lev in (1, 2):
    for usdt in (1.0, 3.0, 5.0):
        binance_executor._FILTER_CACHE.clear()
        r = Recorder()
        binance_executor.signed_request = r
        try:
            binance_executor.place_futures_plan(
                "ETHUSDT", "LONG", P, P * 1.04, P * 0.96, usdt * lev / P,
                lev, ("k", "s"), f"plan-mn-{lev}-{usdt}")
            qty_sent = float(r.of("/fapi/v1/order", "POST")[0][2]["quantity"])
            notional = qty_sent * P
            check(f"futu-minnotional {usdt} USDT x{lev}: notional {notional:.2f} >= {f_min_notional}",
                  notional >= f_min_notional, f"notional {notional:.2f} < {f_min_notional}")
        except ExecutorError as exc:
            check(f"futu-minnotional {usdt} USDT x{lev}: raise -4164 (khong goi lenh rac)",
                  exc.code == -4164 and not r.of("/fapi/v1/order", "POST"),
                  f"{exc.code} | goi san={bool(r.of('/fapi/v1/order', 'POST'))}")

# SPOT: USDT nho -> floor lam notional duoi minNotional that -> phai raise
# (dat sau khi _SpotRec duoc dinh nghia)


# ─── SPOT ──────────────────────────────────────────────────────────────────────

def spot_case(title: str, usdt: float, price: float, executed: float, free: float,
              fee_qty=0.0, fee_asset="BNB", fills_from_get=None, tp2=None,
              dup_on_oco=False):
    print()
    print(f"--- SPOT: {title}")
    print(f"    config: {usdt} USDT (khong don bay) / gia {price} -> "
          f"{usdt / price:.6f} coin; executed={executed}, free={free}, "
          f"fee={fee_qty} {fee_asset}")
    binance_executor._FILTER_CACHE.clear()
    r = _SpotRec(executed=executed, free=free, fee_qty=fee_qty,
                 fee_asset=fee_asset, fills_from_get=fills_from_get,
                 dup_on_oco=dup_on_oco)
    binance_executor.signed_request = r

    qty_coins = usdt / price
    try:
        out = binance_executor.place_spot_plan(
            "ETHUSDT", price, price * 1.04, price * 0.96, qty_coins,
            ("k", "s"), f"plan-{title}", tp2=tp2,
        )
    except ExecutorError as exc:
        check(f"{title}: khong loi", False, f"{exc.code}: {exc.msg}")
        return

    flt = real_filters(SPOT_BASE, "ETHUSDT", True)
    step, tick, mn = flt["step"], flt["tick"], flt["min_notional"]

    mkt = r.of("/api/v3/order", "POST", lambda p: p.get("type") == "MARKET")
    check(f"{title}: MARKET BUY dung newClientOrderId",
          mkt and "newClientOrderId" in mkt[0][2] and "clientOrderId" not in mkt[0][2])
    mq = float(mkt[0][2]["quantity"]) if mkt else 0
    check(f"{title}: market qty {mq} la boi cua step {step}",
          abs(mq / step - round(mq / step)) < 1e-9)
    check(f"{title}: market notional {mq * price:.2f} >= minNotional {mn}",
          mq * price >= mn, f"{mq * price:.2f}")

    # sellable = executed - phi (neu phi la base) roi floor
    sellable_raw = executed - (fee_qty if fee_asset == "ETH" else 0.0)
    expect_sell = binance_executor._floor_to(min(sellable_raw, free), step)

    ocos = r.of("/api/v3/orderList/oco", "POST")
    if dup_on_oco:
        check(f"{title}: OCO duplicate -> khong ban tiep",
              not r.of("/api/v3/order", "POST", lambda p: p.get("side") == "SELL"))
        return

    check(f"{title}: co OCO", len(ocos) >= 1, str(len(ocos)))
    if not ocos:
        return
    total = sum(float(t[2]["quantity"]) for t in ocos)
    check(f"{title}: tong qty OCO {total} == sellable {expect_sell}",
          abs(total - expect_sell) < 1e-9, f"{total} vs {expect_sell}")
    check(f"{title}: tong qty OCO KHONG vuot free {free}", total <= free + 1e-9,
          f"{total} > {free}")
    for m, p, prm in ocos:
        q = float(prm["quantity"])
        ok_mn = q * float(prm["abovePrice"]) >= mn
        check(f"{title}: OCO qty {q} * gia >= minNotional {mn}", ok_mn,
              f"{q * float(prm['abovePrice']):.2f}")
        check(f"{title}: OCO qty {q} la boi step", abs(q / step - round(q / step)) < 1e-9)
        check(f"{title}: OCO co listClientOrderId rieng",
              bool(prm.get("listClientOrderId")), str(prm))
    if tp2 is not None:
        check(f"{title}: co tp2 -> 2 OCO", len(ocos) == 2, str(len(ocos)))
    else:
        check(f"{title}: khong tp2 -> 1 OCO", len(ocos) == 1, str(len(ocos)))

    check(f"{title}: spot KHONG co leverage trong payload",
          "leverage" not in mkt[0][2])


class _SpotRec(Recorder):
    def __init__(self, executed, free, fee_qty, fee_asset, fills_from_get, dup_on_oco):
        super().__init__()
        self.executed = executed
        self.free = free
        self.fee_qty = fee_qty
        self.fee_asset = fee_asset
        self.fills_from_get = fills_from_get
        self.dup_on_oco = dup_on_oco
        self._oco_n = 0

    def __call__(self, base, api_key, secret, method, path, params):
        self.calls.append((method, path, dict(params)))
        if path.endswith("/exchangeInfo") or path.endswith("/api/v3/account"):
            return {"balances": [{"asset": "ETH", "free": str(self.free), "locked": "0"}],
                    "commissionRates": {"maker": "0.001", "taker": "0.001"}}
        if path.endswith("/api/v3/order") and method == "POST" and params.get("type") == "MARKET":
            return {"orderId": 60000, "clientOrderId": params.get("newClientOrderId"),
                    "status": "FILLED", "executedQty": str(self.executed),
                    "cummulativeQuoteQty": str(self.executed * 2700),
                    "fills": [{"price": "2700", "qty": str(self.executed),
                               "commission": str(self.fee_qty),
                               "commissionAsset": self.fee_asset}]}
        if path.endswith("/api/v3/order") and method == "GET":
            return {"status": "FILLED", "executedQty": str(self.executed),
                    "fills": self.fills_from_get if self.fills_from_get is not None else []}
        if path.endswith("/api/v3/orderList/oco") and method == "POST":
            self._oco_n += 1
            if self.dup_on_oco:
                raise ExecutorError(-2010, "Duplicate order")
            return {"orderListId": 70000 + self._oco_n,
                    "listClientOrderId": params.get("listClientOrderId"),
                    "orders": [{"clientOrderId": params.get("aboveClientOrderId"),
                                "orderId": 70000 + self._oco_n},
                               {"clientOrderId": params.get("belowClientOrderId"),
                                "orderId": 70000 + self._oco_n}]}
        return {}


# SPOT: USDT nho → floor lam notional rot xuoi duoi minNotional that → phai raise,
# khong duoc gui lenh ma san se tu choi -1013.
spot_flt = real_filters(SPOT_BASE, "ETHUSDT", True)
s_min_notional = spot_flt["min_notional"]
for usdt in (1.0, 3.0, 4.9):
    binance_executor._FILTER_CACHE.clear()
    r = _SpotRec(executed=usdt / P, free=usdt / P, fee_qty=0, fee_asset="BNB",
                 fills_from_get=None, dup_on_oco=False)
    binance_executor.signed_request = r
    try:
        binance_executor.place_spot_plan("ETHUSDT", P, P * 1.04, P * 0.96, usdt / P,
                                         ("k", "s"), f"plan-smn-{usdt}")
        check(f"spot-minnotional {usdt} USDT: phai raise hoac notional >= {s_min_notional}",
              False, "khong raise")
    except ExecutorError as exc:
        sent = r.of("/api/v3/order", "POST")
        check(f"spot-minnotional {usdt} USDT: raise -4164 va khong goi lenh rac",
              exc.code == -4164 and not sent,
              f"code={exc.code} goi_san={bool(sent)}")


print()
print("=" * 78)
SP = 2700.0
# phi BNB -> base giu nguyen
spot_case("spot-phi-bnb", 270, SP, executed=0.1, free=0.1, fee_qty=0.0001, fee_asset="BNB")
# phi quote (USDT) -> base giu nguyen
spot_case("spot-phi-usdt", 270, SP, executed=0.1, free=0.1, fee_qty=0.27, fee_asset="USDT")
# phi base (ETH) -> tru vao coin
spot_case("spot-phi-eth", 270, SP, executed=0.1, free=0.0999, fee_qty=0.0001, fee_asset="ETH")
# partial fill: mua 0.1 nhung chi khop 0.06
spot_case("spot-partial-fill", 270, SP, executed=0.06, free=0.06, fee_qty=0, fee_asset="BNB")
# free < executed (phan bi OCO khac khoa)
spot_case("spot-free-thap-hon", 270, SP, executed=0.1, free=0.05, fee_qty=0, fee_asset="BNB")
# co tp2
spot_case("spot-co-tp2", 540, SP, executed=0.2, free=0.2, fee_qty=0, fee_asset="BNB",
          tp2=SP * 1.06)
# OCO duplicate
spot_case("spot-oco-duplicate", 270, SP, executed=0.1, free=0.1, fee_qty=0,
          fee_asset="BNB", dup_on_oco=True)
# qty nho hon 1 coin -> raise
try:
    binance_executor._FILTER_CACHE.clear()
    r = _SpotRec(executed=0.1, free=0.1, fee_qty=0, fee_asset="BNB",
                 fills_from_get=None, dup_on_oco=False)
    binance_executor.signed_request = r
    binance_executor.place_spot_plan("ETHUSDT", SP, SP * 1.04, SP * 0.96, 0.0000001,
                                     ("k", "s"), "plan-tiny-spot")
    check("spot-qty-duoi-1-coin: phai raise", False, "khong raise")
except ExecutorError as exc:
    check("spot-qty-duoi-1-coin: raise -4164", exc.code == -4164, str(exc.code))


# ─── TONG KET ─────────────────────────────────────────────────────────────────

print()
print("=" * 78)
failed = [(n, d) for n, ok, d in RESULTS if not ok]
print(f"TONG: {len(RESULTS)} kiem tra, {len(RESULTS) - len(failed)} PASS, {len(failed)} FAIL")
if failed:
    print()
    print("CAC CASE FAIL:")
    for n, d in failed:
        print(f"  ✗ {n}" + (f" — {d}" if d else ""))
    sys.exit(1)
print("TAT CA CASE DEU PASS")
