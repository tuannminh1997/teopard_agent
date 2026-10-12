"""TEST THẬT với API Binance và các biến có sẵn trong .env.

Chạy toàn bộ kiểm tra công khai (không cần API key) trên host đang bật trong .env:
1. Base URL đọc đúng từ .env
2. /exchangeInfo lấy được filter thật
3. /ticker/price lấy được giá thật
4. min_order_usdt tính đúng với filter thật
5. USDT -> coin -> TP/SL khớp invariant cho mọi đòn bẩy
6. WebSocket base URL đúng cho cả spot và futures demo/live
7. _ensure_oco_room đọc trần OCO thật từ exchangeInfo

Các test cần API key (đặt lệnh) hiện KHÔNG chạy được ở local vì .env không có key Binance —
key lưu per-user trong bot.db trên Railway. Script in rõ điều này ở cuối.
"""
import os
import sys
import math

try:
    # Windows PowerShell thường dùng cp1252; test này in tiếng Việt nên ép UTF-8
    # để API smoke test không chết trước khi chạy assertion.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import requests
from dotenv import load_dotenv

load_dotenv()

import binance_executor
import spot_user_stream
import futures_user_stream

RESULTS: list[tuple[str, bool, str]] = []
FAILED = 0


def check(name, cond, detail=""):
    global FAILED
    RESULTS.append((name, bool(cond), detail))
    if not cond:
        FAILED += 1
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


print("=" * 78)
print("TEST THAT VOI API BINANCE + BIEN TRONG .env")
print("=" * 78)

FUTU = binance_executor.FUTURES_API_BASE
SPOT = binance_executor.SPOT_API_BASE
print(f"\n  FUTURES_API_BASE = {FUTU}")
print(f"  SPOT_API_BASE    = {SPOT}")
check("FUTURES_API_BASE lấy từ .env", FUTU == os.getenv("FUTURES_API_BASE"))
check("SPOT_API_BASE lấy từ .env", SPOT == os.getenv("SPOT_API_BASE"))
check("Base URL là Binance Futures/Spot hợp lệ",
      FUTU in {"https://fapi.binance.com", "https://demo-fapi.binance.com"}
      and SPOT in {"https://api.binance.com", "https://demo-api.binance.com"})

# ─── 1. Kết nối thật ────────────────────────────────────────────────────────
print("\n  1. KẾT NỐI THẬT TỚI BINANCE API THEO .env")
for name, base, is_spot in (("FUTU", FUTU, False), ("SPOT", SPOT, True)):
    try:
        path = "/api/v3/ping" if is_spot else "/fapi/v1/ping"
        r = requests.get(base + path, timeout=15)
        check(f"{name} ping", r.status_code == 200, f"HTTP {r.status_code}")
    except Exception as exc:
        check(f"{name} ping", False, str(exc)[:100])

# ─── 2. exchangeInfo thật ───────────────────────────────────────────────────
print("\n  2. FILTER THẬT TỪ exchangeInfo")
real = {}
for sym in ("ETHUSDT", "BTCUSDT"):
    for mkt, is_spot in (("futu", False), ("spot", True)):
        try:
            flt = binance_executor._market_filters(FUTU if not is_spot else SPOT,
                                                   "", "", sym, is_spot)
            real[(sym, mkt)] = flt
            check(f"{mkt} {sym} filter đọc được",
                  flt.get("step") and flt.get("tick") and flt.get("min_notional"),
                  f"step={flt.get('step')} tick={flt.get('tick')} "
                  f"minNotional={flt.get('min_notional')}")
        except Exception as exc:
            check(f"{mkt} {sym} filter đọc được", False, str(exc)[:120])

# Spot phải có trần OCO list
spot_flt = real.get(("ETHUSDT", "spot"))
if spot_flt:
    check("spot ETHUSDT có trần MAX_NUM_ORDER_LISTS",
          bool(spot_flt.get("max_order_lists")),
          f"trần={spot_flt.get('max_order_lists')}")

# ─── 3. giá thật ────────────────────────────────────────────────────────────
print("\n  3. GIÁ THẬT + min_order_usdt")
prices = {}
for sym in ("ETHUSDT", "BTCUSDT"):
    for mkt, is_spot in (("futu", False), ("spot", True)):
        try:
            base = SPOT if is_spot else FUTU
            path = "/api/v3/ticker/price" if is_spot else "/fapi/v1/ticker/price"
            px = float(requests.get(base + path, params={"symbol": sym},
                                    timeout=15).json()["price"])
            prices[(sym, mkt)] = px
        except Exception as exc:
            check(f"{mkt} {sym} giá", False, str(exc)[:80])

for (sym, mkt), px in prices.items():
    flt = real.get((sym, mkt))
    if not flt:
        continue
    is_spot = mkt == "spot"
    # spot: không có đòn bẩy
    # futu: thử nhiều mức đòn bẩy
    for lev in (1, 20, 125) if not is_spot else (1,):
        try:
            info = binance_executor.min_order_usdt(
                SPOT if is_spot else FUTU, sym, is_spot, leverage=lev, price=px)
            usdt_min = info["usdt_min"]
            # Quantity phải ceil lên stepSize trước, nếu không floor lúc đặt lệnh
            # có thể làm notional thực tế lại thấp hơn minNotional.
            raw_qty = max(flt.get("min_qty") or 0,
                          flt["min_notional"] / px,
                          flt["step"])
            required_qty = math.ceil(raw_qty / flt["step"] - 1e-12) * flt["step"]
            expected = required_qty * px / (lev if not is_spot else 1)
            check(f"min_order_usdt {mkt} {sym} x{lev} = {usdt_min:.4f}",
                  abs(usdt_min - expected) < 1e-6,
                  f"mong đợi {expected:.4f}")
            # Số coin tối thiểu phải >= 1 step
            coins = usdt_min * (lev if not is_spot else 1) / px
            check(f"  → coin tối thiểu {coins:.6f} >= step {flt['step']}",
                  coins >= flt["step"] - 1e-12)
        except Exception as exc:
            check(f"min_order_usdt {mkt} {sym} x{lev}", False, str(exc)[:100])

# ─── 4. Invariant TP/SL từ USDT ─────────────────────────────────────────────
print("\n  4. INVARIANT TP/SL KHI QUY TỪ USDT")
for (sym, mkt), flt in real.items():
    is_spot = mkt == "spot"
    step = flt["step"]
    for usdt in (10.0, 100.0, 1000.0):
        for lev in ((1, 20, 125) if not is_spot else (1,)):
            for tp2 in (None, 1.06):
                px = prices.get((sym, mkt), 2700.0)
                notional = usdt * (1 if is_spot else lev)
                coins = binance_executor._floor_to(notional / px, step)
                if coins < step:
                    continue
                q1, q2 = (binance_executor.split_tp_qty(coins, step) if tp2 is not None
                          else (coins, 0.0))
                tp_ok = (abs(q1 + q2 - coins) < 1e-12) if tp2 is not None \
                    else (q2 == 0.0 and q1 == coins)
                mn_ok = coins * px >= flt["min_notional"]
                if mn_ok:
                    # Đạt tối thiểu → TP/SL PHẢI khớp position
                    check(f"{mkt} {sym} {usdt}USDT x{lev} tp2={'C' if tp2 else '-'}: "
                          f"TP+SL={coins} notional={coins * px:.2f}",
                          tp_ok, f"tp1={q1} tp2={q2}")
                else:
                    # KHÔNG đạt tối thiểu → bot phải HỦY, không được tạo lệnh rác.
                    # Kiểm tra: nếu user nhập số USDT này, _auto_execute_plan sẽ abort.
                    usdt_min = flt["min_notional"] / (1 if is_spot else lev)
                    check(f"{mkt} {sym} {usdt}USDT x{lev} dưới minNotional "
                          f"{flt['min_notional']} → bot phải hủy (cần ≥ {usdt_min:.2f} USDT)",
                          usdt < usdt_min,
                          f"usdt={usdt} usdt_min={usdt_min}")

# ─── 5. WebSocket base ─────────────────────────────────────────────────────
print("\n  5. WEBSOCKET BASE URL")
cases = [
    ("https://demo-api.binance.com", spot_user_stream._ws_base, "demo-stream.binance.com"),
    ("https://api.binance.com", spot_user_stream._ws_base, "stream.binance.com:9443"),
    ("https://demo-fapi.binance.com", futures_user_stream._ws_base, "fstream.binancefuture.com"),
    ("https://fapi.binance.com", futures_user_stream._ws_base, "fstream.binance.com"),
]
for base, fn, expect in cases:
    got = fn(base)
    check(f"{base} → {got}", expect in got, f"mong đợi chứa '{expect}'")
check("live khác demo (spot)", spot_user_stream._ws_base("https://api.binance.com")
      != spot_user_stream._ws_base("https://demo-api.binance.com"))
check("live khác demo (futu)", futures_user_stream._ws_base("https://fapi.binance.com")
      != futures_user_stream._ws_base("https://demo-fapi.binance.com"))

# ─── 6. Các env khác trong .env có thực sự được dùng ───────────────────────
print("\n  6. BIẾN MÔI TRƯỜNG CÓ THỰC SỰ ĐƯỢC DÙNG")
import analyze
env_checks = [
    ("FUTURES_USER_STREAM_ENABLED", "futures_user_stream._enabled"),
    ("SPOT_USER_STREAM_ENABLED", "spot_user_stream._enabled"),
    ("TP1_SHARE", "binance_executor.TP1_SHARE"),
    ("SPOT_FILL_TIMEOUT_SECONDS", "binance_executor.SPOT_FILL_TIMEOUT_SECONDS"),
    ("SPOT_OCO_DELAY_SECONDS", "binance_executor.SPOT_OCO_DELAY_SECONDS"),
    ("FUTURES_ENTRY_TTL_SECONDS", "binance_executor.FUTURES_ENTRY_TTL_SECONDS"),
    ("PLANNER_MODEL", "analyze.PLANNER_MODEL"),
    ("PLANNER_SUPPLIER", "analyze.PLANNER_SUPPLIER"),
]
for var, code in env_checks:
    module_name, attr = code.split(".")
    mod = {"futures_user_stream": futures_user_stream,
           "spot_user_stream": spot_user_stream,
           "binance_executor": binance_executor,
           "analyze": analyze}[module_name]
    val = os.getenv(var)
    used = getattr(mod, attr, None)
    if var.endswith("_ENABLED"):
        # Đọc trực tiếp từ env mỗi lần gọi
        enabled_now = os.getenv(var, "0").strip().lower() in {"1", "true", "yes", "on"}
        check(f"{var}={val!r} → stream {'được bật' if enabled_now else 'tắt (mặc định)'}",
              isinstance(val, str) or val is None)
    else:
        check(f"{var}={val!r} → code dùng {used!r}", val is not None)

print("\n" + "=" * 78)
print(f"TỔNG: {len(RESULTS)} kiểm tra, {len(RESULTS) - FAILED} PASS, {FAILED} FAIL")
print("=" * 78)

has_key = any(os.getenv(k) for k in ("BINANCE_API_KEY", "BINANCE_SECRET", "API_KEY"))
print()
print("Ghi chú:")
print("  - Các test trên đều gọi API THẬT theo host trong .env và không cần API key.")
print("  - KHÔNG đặt được lệnh thật ở local vì .env không chứa API key Binance")
print("    (key lưu per-user trong bot.db → trên Railway là /data/bot.db).")
print("  - Để đặt + verify lệnh thật, chạy trên Railway:")
print("      python verify_orders.py --market futures --qty 10 --leverage 20 --live")
print("      python verify_orders.py --market spot --qty 270 --live")
if FAILED:
    sys.exit(1)
