"""KIỂM TRA + ĐẶT LỆNH THẬT auto futures/spot dùng ĐÚNG config trong .env và key trong DB.

Mặc định: DRY-RUN (chỉ tính toán + in payload, KHÔNG gửi lệnh lên sàn).
Bật đặt lệnh thật:  LIVE=1 python verify_orders.py --market spot --qty 10

Script đọc:
- FUTURES_API_BASE / SPOT_API_BASE từ .env
- API key của user từ key_store (bot.db) — KHÔNG cần nhập key trong .env
- qty / leverage từ tham số hoặc từ cấu hình phiên đang bật của user

Các bước:
1. Kiểm tra key dùng được trên base đang chọn (GET /account hoặc /fapi/v2/balance)
2. Đọc exchangeInfo THẬT → stepSize/tickSize/minQty/minNotional
3. Quy USDT → coin đúng công thức bot dùng
4. In TOÀN BỘ payload entry + TP/SL sẽ gửi
5. Nếu LIVE=1 → gửi thật rồi đọc lại trạng thái lệnh để verify
"""
import argparse
import json
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

import binance_executor
import key_store
from binance_executor import ExecutorError

RESULTS: list[tuple[str, bool, str]] = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def get_filters(base, symbol, spot):
    path = "/api/v3/exchangeInfo" if spot else "/fapi/v1/exchangeInfo"
    data = requests.get(base + path, params={"symbol": symbol}, timeout=20).json()
    for s in data["symbols"]:
        if s["symbol"] == symbol:
            f = {"base_asset": s["baseAsset"], "tick": 0.01, "step": 0.001,
                 "min_notional": None}
            for flt in s["filters"]:
                t = flt["filterType"]
                if t == "PRICE_FILTER":
                    f["tick"] = float(flt["tickSize"])
                elif t == "LOT_SIZE":
                    f["step"] = float(flt["stepSize"])
                    f["min_qty"] = float(flt["minQty"])
                elif t in ("MIN_NOTIONAL", "NOTIONAL"):
                    raw = flt.get("notional") or flt.get("minNotional")
                    if raw:
                        f["min_notional"] = float(raw)
            return f
    raise SystemExit(f"symbol {symbol} khong ton tai tren {base}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["futures", "spot"], default="futures")
    ap.add_argument("--symbol", default="ETHUSDT")
    ap.add_argument("--qty", type=float, default=None,
                    help="so USDT (khong lay tu config neu bo trong)")
    ap.add_argument("--leverage", type=int, default=None)
    ap.add_argument("--user", type=int, default=None, help="user id (default: user dau tien)")
    ap.add_argument("--live", action="store_true", help="THAT SU DAT LENH (nguy hiem)")
    args = ap.parse_args()

    spot = args.market == "spot"
    base = os.getenv("SPOT_API_BASE") if spot else os.getenv("FUTURES_API_BASE")
    print("=" * 78)
    print(f"MARKET={args.market}  BASE={base}  SYMBOL={args.symbol}")
    print("=" * 78)

    # ── 1. API key ────────────────────────────────────────────────────────────
    uid = args.user
    if uid is None:
        try:
            import sqlite3
            conn = sqlite3.connect(key_store.DB_PATH)
            row = conn.execute("SELECT user_id FROM user_api_keys LIMIT 1").fetchone()
            conn.close()
            uid = int(row[0]) if row else None
        except Exception:
            uid = None
    if uid is None:
        print("\nKHONG TIM THAY user nao co API key trong DB.")
        print("Chay script nay TREN RAILWAY (noi co /data/bot.db), hoac truyen --user <id>.")
        print("\nVan chay che do TINH TOAN KHONG CAN KEY:")
        return compute_only(args, base, spot)

    try:
        keys = key_store.get_api_keys(uid)
    except Exception as exc:
        print(f"khong doc duoc key: {exc}")
        return compute_only(args, base, spot)
    api_key, secret = keys
    check(f"user {uid} co API key trong DB", True)

    # ── 2. Key hoat dong tren base dang chon? ────────────────────────────────
    try:
        if spot:
            acct = binance_executor.signed_request(base, api_key, secret, "GET",
                                                   "/api/v3/account", {})
            bals = {b["asset"]: b for b in acct.get("balances", [])}
            usdt_free = float(bals.get("USDT", {}).get("free", 0) or 0)
            print(f"  key OK tren {base} | USDT free = {usdt_free}")
        else:
            bal = binance_executor.signed_request(base, api_key, secret, "GET",
                                                  "/fapi/v2/balance", {})
            usdt = [b for b in bal if b.get("asset") == "USDT"]
            usdt_free = float(usdt[0].get("availableBalance", 0)) if usdt else 0.0
            print(f"  key OK tren {base} | USDT available = {usdt_free}")
        check("key hoat dong tren base dang chon", True)
    except ExecutorError as exc:
        check("key hoat dong tren base dang chon", False,
              f"{exc.code}: {exc.msg[:200]}")
        print("\n  → key khong dung cho base nay. Neu dang o demo thi can key demo,")
        print("    neu da doi sang live thi can key production. KHONG dat duoc lenh.")
        return compute_only(args, base, spot)

    # ── 3. Config ────────────────────────────────────────────────────────────
    flt = get_filters(base, args.symbol, spot)
    print(f"\n  FILTER THAT {args.symbol}: step={flt['step']} tick={flt['tick']} "
          f"minQty={flt.get('min_qty')} minNotional={flt['min_notional']}")

    qty_usdt = args.qty
    leverage = args.leverage
    if qty_usdt is None or leverage is None:
        try:
            import analyze
            cfg = analyze.get_auto_scan_market_settings(uid, args.market)
        except Exception:
            cfg = None
        if cfg and cfg.get("qty"):
            print(f"  config phien: qty={cfg['qty']} leverage={cfg['leverage']} "
                  f"symbol={cfg['symbol']}")
            qty_usdt = qty_usdt or float(cfg["qty"])
            if leverage is None:
                leverage = int(cfg["leverage"] or 1) if not spot else 1
        else:
            if qty_usdt is None:
                qty_usdt = 10.0
            if leverage is None:
                leverage = 1
    if spot:
        leverage = 1
    print(f"\n  SE DAT: {qty_usdt} USDT"
          + (f" x don bay {leverage} = vi the {qty_usdt * leverage} USDT" if not spot else ""))

    # ── 4. Giá + quy đổi ─────────────────────────────────────────────────────
    if spot:
        price = float(binance_executor.public_get(base, "/api/v3/ticker/price",
                                                  {"symbol": args.symbol})["price"])
    else:
        price = float(binance_executor.public_get(base, "/fapi/v1/ticker/price",
                                                 {"symbol": args.symbol})["price"])
    notional = qty_usdt * leverage if not spot else qty_usdt
    coins_raw = notional / price
    coins = binance_executor._floor_to(coins_raw, flt["step"])
    print(f"  gia hien tai = {price}")
    print(f"  quy doi: {notional} USDT / {price} = {coins_raw:.8f} → floor step "
          f"{flt['step']} = {coins}")
    check("qty la boi cua stepSize",
          abs(coins / flt["step"] - round(coins / flt["step"])) < 1e-9)
    check("notional >= minNotional", coins * price >= (flt["min_notional"] or 0),
          f"{coins * price:.2f} vs {flt['min_notional']}")
    check("notional <= vi the yeu cau", coins * price <= notional + 1e-9,
          f"{coins * price:.2f} vs {notional}")
    if coins <= 0:
        print("\n  → qty floor ve 0: so USDT qua nho cho price nay. Tang qty hoac don bay.")
        return

    q1, q2 = binance_executor.split_tp_qty(coins, flt["step"])
    tp_price = price * 1.04
    sl_price = price * 0.96
    tp1_p = binance_executor._fmt(binance_executor._floor_to(tp_price, flt["tick"]))
    sl_p = binance_executor._fmt(binance_executor._floor_to(sl_price, flt["tick"]))

    # ── 5. In payload se gui ─────────────────────────────────────────────────
    plan_id = f"verify-{args.market}-{uid}-1"
    print("\n  ── PAYLOAD SE GUI ──")
    if spot:
        print(f"  ENTRY  POST /api/v3/order")
        print(f"         symbol={args.symbol} side=BUY type=MARKET "
              f"quantity={binance_executor._fmt(coins)}")
        print(f"         newClientOrderId={plan_id}-e")
        if q2 > 0:
            print(f"  OCO-A  POST /api/v3/orderList/oco qty={binance_executor._fmt(q1)} "
                  f"above=@{tp1_p} belowStop=@{sl_p} list={plan_id}-list")
            print(f"  OCO-B  POST /api/v3/orderList/oco qty={binance_executor._fmt(q2)} "
                  f"above=@{tp1_p} (tp2) belowStop=@{sl_p} list={plan_id}-list2")
        else:
            print(f"  OCO    POST /api/v3/orderList/oco qty={binance_executor._fmt(coins)} "
                  f"above=@{tp1_p} belowStop=@{sl_p} list={plan_id}-list")
        print(f"  (OCO qty se lay min(executedQty - phi, free balance) luc chay that)")
    else:
        print(f"  ENTRY  POST /fapi/v1/order")
        print(f"         symbol={args.symbol} side=BUY type=LIMIT timeInForce=GTC "
              f"quantity={binance_executor._fmt(coins)} price={binance_executor._fmt(price)}")
        print(f"         newClientOrderId={plan_id}-e positionSide=LONG")
        print(f"  TP1    POST /fapi/v1/algoOrder TAKE_PROFIT_MARKET trigger={tp1_p} "
              f"qty={binance_executor._fmt(q1)}")
        if q2 > 0:
            print(f"  TP2    POST /fapi/v1/algoOrder TAKE_PROFIT_MARKET trigger={tp1_p} "
                  f"qty={binance_executor._fmt(q2)}")
        print(f"  SL     POST /fapi/v1/algoOrder STOP_MARKET trigger={sl_p} "
              f"qty={binance_executor._fmt(coins)}  (100% vi the)")

    if not args.live:
        print("\n  ⛔ DRY-RUN: khong gui lenh. Them --live de that su dat (se dat tien ao demo).")
        report()
        return

    # ── 6. ĐẶT THẬT ──────────────────────────────────────────────────────────
    confirm = input("\n  XAC NHAN dat lenh THAT tren demo? goi 'yes' de tiep tuc: ")
    if confirm.strip().lower() != "yes":
        print("  Da huy.")
        return

    print("\n  DANG DAT LENH...")
    try:
        if spot:
            out = binance_executor.place_spot_plan(
                args.symbol, price, tp_price, sl_price, coins, keys, plan_id,
                tp2=(tp_price * 1.02 if q2 > 0 else None))
        else:
            out = binance_executor.place_futures_plan(
                args.symbol, "LONG", price, tp_price, sl_price, coins, leverage,
                keys, plan_id, tp2=(tp_price * 1.02 if q2 > 0 else None))
    except ExecutorError as exc:
        check("dat lenh", False, f"{exc.code}: {exc.msg[:300]}")
        report()
        return

    print(f"\n  KET QUA: {json.dumps(out, indent=2, ensure_ascii=False)}")
    check("dat lenh thanh cong", True)

    # ── 7. Verify bằng cách đọc lại từ sàn ──────────────────────────────────
    print("\n  ── VERIFY DOC LAI TU SAN ──")
    if spot:
        st = binance_executor.spot_order_state(base, keys, args.symbol, f"{plan_id}-e")
        print(f"  entry: {st}")
        open_orders = binance_executor.signed_request(
            base, api_key, secret, "GET", "/api/v3/openOrders", {"symbol": args.symbol})
        ocos = binance_executor.signed_request(
            base, api_key, secret, "GET", "/api/v3/orderList", {"symbol": args.symbol})
        print(f"  openOrders: {len(open_orders)} | orderLists: {len(ocos)}")
        check("OCO da duoc tao tren san", len(ocos) >= 1, str(len(ocos)))
    else:
        st = binance_executor.signed_request(
            base, api_key, secret, "GET", "/fapi/v1/order",
            {"symbol": args.symbol, "origClientOrderId": f"{plan_id}-e"})
        print(f"  entry: status={st.get('status')} qty={st.get('origQty')} "
              f"executed={st.get('executedQty')}")
        algos = binance_executor.open_futures_algo_orders(keys, args.symbol)
        print(f"  algo orders dang mo: {[(a.get('algoId'), a.get('type')) for a in (algos or [])]}")
        check("TP/SL algo da duoc tao tren san", bool(algos), str(len(algos or [])))
        # position co khop voi qty da tinh khong?
        pos = binance_executor.signed_request(
            base, api_key, secret, "GET", "/fapi/v2/positionRisk", {"symbol": args.symbol})
        pos = pos.get("data") if isinstance(pos, dict) else pos
        amt = sum(float(p.get("positionAmt") or 0) for p in (pos or []))
        print(f"  position hien tai: {amt}")
    report()


def compute_only(args, base, spot):
    """Chỉ tính toán, không cần key."""
    flt = get_filters(base, args.symbol, spot)
    print(f"\n  FILTER {args.symbol}: step={flt['step']} tick={flt['tick']} "
          f"minNotional={flt['min_notional']}")
    qty_usdt = args.qty if args.qty is not None else 10.0
    lev = 1 if spot else (args.leverage or 20)
    notional = qty_usdt * lev if not spot else qty_usdt
    price = 2700.0  # gia tham khao
    coins = binance_executor._floor_to(notional / price, flt["step"])
    print(f"  {qty_usdt} USDT" + (f" x{lev}" if not spot else "") + f" / {price} = {coins} coin")
    q1, q2 = binance_executor.split_tp_qty(coins, flt["step"])
    print(f"  TP1={q1} TP2={q2} SL={coins}")
    report()


def report():
    print()
    print("=" * 78)
    failed = [(n, d) for n, ok, d in RESULTS if not ok]
    print(f"TONG: {len(RESULTS)} kiem tra, {len(RESULTS) - len(failed)} PASS, {len(failed)} FAIL")
    for n, d in failed:
        print(f"  FAIL {n}" + (f" — {d}" if d else ""))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
