"""Chạy lại agent intraday trên dữ liệu quá khứ để đo thay vì đoán.

Không nằm trong luồng bot. Xem --help.
"""
import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime, timedelta, timezone

import requests

from dotenv import load_dotenv

load_dotenv()

import analyze  # noqa: E402
from plan_validator import validate_plan  # noqa: E402

FEE_RT = 2 * float(os.getenv("FEE_TAKER_PCT", "0.05"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _to_ms(value: str) -> int:
    dt = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_klines_closed(symbol: str, interval: str, end_ms: int, limit: int = 1000):
    """Tải nến đóng hoàn toàn trước end_ms (bỏ nến đang chạy tại thời điểm as_of)."""
    params = {"symbol": symbol, "interval": interval, "endTime": end_ms, "limit": limit}
    r = requests.get(analyze.BINANCE_API_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    # Bỏ nến có open_time >= end_ms (nến chưa đóng tại as_of).
    data = [row for row in data if int(row[0]) < end_ms]
    if not data:
        return None
    import pandas as pd

    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]:
        df[col] = df[col].astype(float)
    return df


def build_frame(symbol: str, interval: str, fetch_limit: int, end_ms: int):
    df = fetch_klines_closed(symbol, interval, end_ms, limit=fetch_limit)
    if df is None:
        return None
    return analyze.add_indicators_intraday(df)


def build_packet_asof(symbol: str, end_ms: int, price: float):
    timeframe_data = {}
    for label, (interval, limit) in analyze.INTRADAY_TIMEFRAMES.items():
        timeframe_data[label] = build_frame(symbol, interval, limit, end_ms)
    if any(v is None for v in timeframe_data.values()):
        return None, None, timeframe_data
    ref: dict = {}
    try:
        daily = fetch_klines_closed(symbol, "1d", end_ms, limit=10)
        if daily is not None and len(daily) >= 1:
            prev = daily.iloc[-2] if len(daily) >= 2 else daily.iloc[-1]
            live = daily.iloc[-1]
            ref = {
                "prev_day_high": float(prev["high"]), "prev_day_low": float(prev["low"]),
                "prev_day_close": float(prev["close"]),
                "today_open": float(live["open"]), "today_high": float(live["high"]),
                "today_low": float(live["low"]),
            }
        weekly = fetch_klines_closed(symbol, "1w", end_ms, limit=3)
        if weekly is not None and len(weekly) >= 2:
            prev_w = weekly.iloc[-2]
            ref["prev_week_high"] = float(prev_w["high"])
            ref["prev_week_low"] = float(prev_w["low"])
    except Exception as exc:
        print(f"[REPLAY] ref levels lỗi: {exc}", flush=True)
    derivs: dict = {}
    if not ARGS.no_derivatives:
        try:
            funding = analyze.get_funding_rate_history(symbol, 4)
            if funding:
                derivs["funding"] = {"latest_pct": funding["latest_pct"], "history_pct": funding["history_pct"]}
                derivs["funding_hist"] = list(funding["history_pct"])
        except Exception:
            pass
        try:
            derivs.update(analyze._intraday_taker_windows(timeframe_data.get("1H")))
        except Exception:
            pass
    btc = None
    if symbol != f"BTC{analyze.BINANCE_QUOTE_ASSET}":
        try:
            btc = analyze.get_btc_intraday_snapshot()
        except Exception:
            btc = None
    text, facts = analyze.build_intraday_packet(timeframe_data, ref, derivs, btc, price, symbol=symbol)
    facts = dict(facts)
    facts["price"] = price
    facts["current_price"] = price
    return text, facts, timeframe_data


def call_model(symbol: str, packet_text: str, model_override: str | None) -> str | None:
    system_prompt = analyze.load_system_prompt("short")
    user_prompt = "\n".join([
        f"PHÂN TÍCH {symbol} — INTRADAY",
        "Packet (dữ liệu lịch sử tại thời điểm phân tích):",
        "",
        packet_text,
        "",
        "Trả về đúng MỘT đối tượng JSON theo system prompt, không thêm chữ ngoài JSON.",
    ])
    if model_override:
        old = analyze.PLANNER_MODEL
        analyze.PLANNER_MODEL = model_override
        try:
            return analyze.request_json_analysis(system_prompt, user_prompt)
        finally:
            analyze.PLANNER_MODEL = old
    return analyze.request_json_analysis(system_prompt, user_prompt)


def fetch_after(symbol: str, start_ms: int, end_ms: int, interval: str = "5m"):
    params = {"symbol": symbol, "interval": interval, "startTime": start_ms, "endTime": end_ms, "limit": 1000}
    r = requests.get(analyze.BINANCE_API_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        return []
    rows = []
    for k in data:
        if len(k) < 7:
            continue
        rows.append({
            "open_time": datetime.fromtimestamp(float(k[0]) / 1000, timezone.utc),
            "close_time": datetime.fromtimestamp(float(k[6]) / 1000, timezone.utc),
            "open": float(k[1]), "high": float(k[2]), "low": float(k[3]), "close": float(k[4]),
        })
    # Chỉ dùng nến mở sau khi plan tồn tại.
    return [r for r in rows if r["open_time"] >= datetime.fromtimestamp(start_ms / 1000, timezone.utc)]


def score_plan(plan: dict, symbol: str, as_of: datetime, hold_hours: float) -> dict:
    direction = str(plan.get("quyet_dinh") or "").upper()
    entry_low, entry_cao = float(plan["entry_thap"]), float(plan["entry_cao"])
    sl, tp1 = float(plan["sl"]), float(plan["tp1"])
    entry_mid = (entry_low + entry_cao) / 2
    start_ms = int(as_of.timestamp() * 1000)
    end_ms = int((as_of + timedelta(hours=hold_hours + 1)).timestamp() * 1000)
    try:
        candles = fetch_after(symbol, start_ms, end_ms, "5m")
    except Exception as exc:
        return {"result": "FETCH_ERROR", "r": None, "note": str(exc)}
    if not candles:
        return {"result": "NO_DATA", "r": None, "note": ""}
    deadline = as_of + timedelta(hours=hold_hours)
    entry_idx = None
    for i, c in enumerate(candles):
        if c["low"] <= entry_cao and c["high"] >= entry_low:
            entry_idx = i
            break
    if entry_idx is None:
        return {"result": "NO_ENTRY", "r": None, "note": ""}
    risk_pct = abs(entry_mid - sl) / entry_mid * 100.0 + FEE_RT
    for c in candles[entry_idx:]:
        if c["open_time"] > deadline:
            return {"result": "EXPIRED", "r": None, "note": ""}
        if direction == "LONG":
            hit_sl = c["low"] <= sl
            hit_tp = c["high"] >= tp1
        else:
            hit_sl = c["high"] >= sl
            hit_tp = c["low"] <= tp1
        if hit_sl:
            # Cùng nến chạm cả hai: tính SL trước (thận trọng). R khi cắt lỗ luôn -1R
            # (risk_pct đã gồm phí khứ hồi, SL nằm đúng mức đã khai báo).
            return {"result": "LOSS", "r": -1.0, "note": ""}
        if hit_tp:
            reward_pct = abs(tp1 - entry_mid) / entry_mid * 100.0 - FEE_RT
            return {"result": "WIN", "r": reward_pct / risk_pct, "note": ""}
    return {"result": "OPEN_AT_END", "r": None, "note": ""}


def baseline_plan(direction: str, price: float, atr: float, entry_atr: float = 0.1) -> dict:
    half = entry_atr * atr
    if direction == "LONG":
        return {"quyet_dinh": "LONG", "entry_thap": price - half, "entry_cao": price + half,
                "sl": price - 1.5 * atr, "tp1": price + 2.5 * atr, "tp2": None}
    return {"quyet_dinh": "SHORT", "entry_thap": price - half, "entry_cao": price + half,
            "sl": price + 1.5 * atr, "tp1": price - 2.5 * atr, "tp2": None}


def summarize(rows: list[dict], label: str) -> dict:
    decided = [r for r in rows if r.get("result") in ("WIN", "LOSS")]
    wins = sum(1 for r in decided if r["result"] == "WIN")
    rs = [r["r"] for r in decided if r.get("r") is not None]
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in rs:
        equity += r
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return {
        "label": label,
        "points": len(rows),
        "plans": sum(1 for r in rows if r.get("decision") in ("LONG", "SHORT")),
        "no_trade_rate": _rate(rows, "NO_TRADE"),
        "validator_reject_rate": _rate(rows, "VALIDATOR_REJECT"),
        "win_rate": (wins / len(decided) * 100.0) if decided else None,
        "expectancy_r": (sum(rs) / len(rs)) if rs else None,
        "max_drawdown_r": max_dd if rs else None,
        "trades": len(decided),
    }


def _rate(rows: list[dict], key: str) -> float | None:
    n = sum(1 for r in rows if r.get("error") == key or r.get("decision") == key)
    return (n / len(rows) * 100.0) if rows else None


def main() -> None:
    global ARGS
    parser = argparse.ArgumentParser(description="Replay agent intraday trên lịch sử")
    parser.add_argument("--symbols", default="BTCUSDT")
    parser.add_argument("--from", dest="date_from", required=True)
    parser.add_argument("--to", dest="date_to", required=True)
    parser.add_argument("--step-hours", type=int, default=4)
    parser.add_argument("--model", default=None)
    parser.add_argument("--packet", choices=["full", "compact"], default="full")
    parser.add_argument("--no-derivatives", action="store_true")
    parser.add_argument("--max-calls", type=int, default=20)
    parser.add_argument("--yes", action="store_true", help="Xác nhận chạy khi số lượt gọi vượt --max-calls")
    parser.add_argument("--out", default="replay_out.csv")
    parser.add_argument("--dry-run", action="store_true", help="Không gọi model, chỉ chạy ống dẫn")
    ARGS = parser.parse_args()

    symbols = [s.strip().upper() for s in ARGS.symbols.replace(",", " ").split() if s.strip()]
    start = datetime.strptime(ARGS.date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(ARGS.date_to, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    slots = []
    cur = start
    while cur < end:
        slots.append(cur)
        cur += timedelta(hours=ARGS.step_hours)
    planned_calls = len(symbols) * len(slots) * (2 if not ARGS.dry_run else 0)
    print(f"Ước tính {planned_calls} lượt gọi model ({len(symbols)} symbol x {len(slots)} mốc x tối đa 2 lượt sửa).")
    if planned_calls > ARGS.max_calls and not ARGS.yes and not ARGS.dry_run:
        print(f"Vượt --max-calls={ARGS.max_calls}. Tăng --max-calls hoặc thêm --yes để chạy.", flush=True)
        sys.exit(2)

    rows: list[dict] = []
    for symbol in symbols:
        for as_of in slots:
            end_ms = int(as_of.timestamp() * 1000)
            rec = {"symbol": symbol, "as_of": as_of.isoformat(), "decision": None, "result": None,
                   "r": None, "error": None, "validator": None, "packet_variant": ARGS.packet}
            try:
                df15 = build_frame(symbol, "15m", 300, end_ms)
                price = float(df15.iloc[-1]["close"]) if df15 is not None and not df15.empty else None
                atr15 = float(df15.iloc[-1]["atr_14"]) if df15 is not None and not df15.empty else None
                if price is None or atr15 is None:
                    rec["error"] = "NO_DATA"
                    rows.append(rec)
                    continue
                packet_text, facts, _tf = build_packet_asof(symbol, end_ms, price)
                if packet_text is None:
                    rec["error"] = "NO_DATA"
                    rows.append(rec)
                    continue
                plan = None
                if ARGS.dry_run:
                    plan = baseline_plan("LONG", price, atr15)
                    rec["decision"] = plan["quyet_dinh"]
                    rec["validator"] = "ok"
                else:
                    raw = call_model(symbol, packet_text, ARGS.model)
                    plan = analyze._extract_json_object((raw or "").strip())
                    if plan is None:
                        rec["error"] = "PARSE_ERROR"
                        rows.append(rec)
                        continue
                    rec["decision"] = str(plan.get("quyet_dinh") or "").upper()
                    errors = validate_plan(plan, facts)
                    if errors:
                        rec["validator"] = "; ".join(errors[:3])
                        if plan.get("quyet_dinh") in ("NO_TRADE", "NO TRADE"):
                            rec["error"] = "NO_TRADE"
                        else:
                            rec["error"] = "VALIDATOR_REJECT"
                            rows.append(rec)
                            continue
                    rec["validator"] = "ok"
                if str(plan.get("quyet_dinh") or "").upper() not in ("LONG", "SHORT"):
                    rec["error"] = rec["error"] or "NO_TRADE"
                    rows.append(rec)
                    continue
                scored = score_plan(plan, symbol, as_of, analyze.TRADE_MAX_HOLD_HOURS["short"])
                rec.update(scored)
            except Exception as exc:
                rec["error"] = f"EXC: {type(exc).__name__}: {exc}"[:300]
            rows.append(rec)
            print(f"{symbol} {as_of.isoformat()} -> {rec.get('decision')} {rec.get('result') or rec.get('error')}", flush=True)

    fieldnames = sorted({k for r in rows for k in r})
    with open(ARGS.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Baseline (a): ngẫu nhiên; (b): luôn LONG theo EMA50 4H.
    rng = random.Random(42)
    base_random, base_long = [], []
    for symbol in symbols:
        for as_of in slots:
            try:
                end_ms = int(as_of.timestamp() * 1000)
                df15 = build_frame(symbol, "15m", 300, end_ms)
                if df15 is None or df15.empty:
                    continue
                price = float(df15.iloc[-1]["close"])
                atr = float(df15.iloc[-1]["atr_14"])
                for bucket, plan in (
                    (base_random, baseline_plan(rng.choice(["LONG", "SHORT"]), price, atr)),
                    (base_long, baseline_plan("LONG", price, atr)),
                ):
                    scored = score_plan(plan, symbol, as_of, analyze.TRADE_MAX_HOLD_HOURS["short"])
                    bucket.append({"decision": plan["quyet_dinh"], "result": scored.get("result"), "r": scored.get("r")})
            except Exception:
                continue

    summary = {
        "agent": summarize(rows, "agent"),
        "baseline_random": summarize(base_random, "baseline_random"),
        "baseline_long_ema50_4h": summarize(base_long, "baseline_long_ema50_4h"),
        "params": {"symbols": symbols, "from": ARGS.date_from, "to": ARGS.date_to,
                   "step_hours": ARGS.step_hours, "packet": ARGS.packet,
                   "no_derivatives": ARGS.no_derivatives, "dry_run": ARGS.dry_run},
    }
    out_json = ARGS.out.rsplit(".", 1)[0] + ".json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
