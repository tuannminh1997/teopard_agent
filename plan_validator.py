import math


def _num(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, str):
            value = value.replace(",", "").strip()
            if not value:
                return None
        num = float(value)
    except Exception:
        return None
    if not math.isfinite(num):
        return None
    return num


def _default_cfg():
    import os

    def _f(name, default):
        try:
            return float(os.getenv(name, str(default)))
        except Exception:
            return default

    return {
        "min_rr": _f("MIN_RR", 1.5),
        "sl_atr_min": _f("SL_ATR_MIN", 0.6),
        "sl_atr_max": _f("SL_ATR_MAX", 3.0),
        "liq_sl_mult": _f("LIQ_SL_MULT", 2.0),
        "entry_ready_atr15": _f("ENTRY_READY_ATR15", 0.25),
        "entry_wait_atr1h": _f("ENTRY_WAIT_ATR1H", 2.0),
        "cite_rel_tol": _f("CITE_REL_TOL", 0.005),
        "fee_rt": 2 * _f("FEE_TAKER_PCT", 0.05),
    }


def validate_plan(plan: dict, facts: dict, cfg: dict | None = None) -> list[str]:
    errors: list[str] = []
    cfg = dict(_default_cfg())
    cfg.update(cfg or {})
    facts = facts or {}
    plan = plan or {}

    quyet_dinh = str(plan.get("quyet_dinh") or "").upper().replace(" ", "_").replace("-", "_")
    trang_thai = str(plan.get("trang_thai") or "").upper()
    if quyet_dinh not in {"LONG", "SHORT", "NO_TRADE"}:
        return [f"quyet_dinh phải là LONG/SHORT/NO_TRADE, nhận được {plan.get('quyet_dinh')!r}."]
    if quyet_dinh == "NO_TRADE":
        if trang_thai != "NO_TRADE":
            errors.append(f"NO_TRADE thì trang_thai phải là NO_TRADE, nhận được {plan.get('trang_thai')!r}.")
        return errors
    if trang_thai not in {"READY_TO_ENTER", "SETUP_WAITING_TRIGGER"}:
        errors.append(f"trang_thai phải là READY_TO_ENTER/SETUP_WAITING_TRIGGER, nhận được {plan.get('trang_thai')!r}.")

    entry_thap = _num(plan.get("entry_thap"))
    entry_cao = _num(plan.get("entry_cao"))
    sl = _num(plan.get("sl"))
    tp1 = _num(plan.get("tp1"))
    tp2 = _num(plan.get("tp2"))
    if entry_thap is None or entry_cao is None or sl is None or tp1 is None:
        missing = [k for k, v in (("entry_thap", entry_thap), ("entry_cao", entry_cao), ("sl", sl), ("tp1", tp1)) if v is None]
        errors.append(f"Thiếu mức giá hợp lệ: {', '.join(missing)}.")
        return errors
    if entry_thap > entry_cao:
        errors.append(f"entry_thap ({entry_thap}) phải ≤ entry_cao ({entry_cao}).")
        return errors

    if quyet_dinh == "LONG":
        if not (sl < entry_thap):
            errors.append(f"LONG: SL ({sl}) phải < Entry thấp ({entry_thap}).")
        if not (entry_cao < tp1):
            errors.append(f"LONG: Entry cao ({entry_cao}) phải < TP1 ({tp1}).")
        if tp2 is not None and not (tp2 > tp1):
            errors.append(f"LONG: TP2 ({tp2}) phải > TP1 ({tp1}).")
        liq = _num(facts.get("liq_long"))
    else:
        if not (sl > entry_cao):
            errors.append(f"SHORT: SL ({sl}) phải > Entry cao ({entry_cao}).")
        if not (entry_thap > tp1):
            errors.append(f"SHORT: Entry thấp ({entry_thap}) phải > TP1 ({tp1}).")
        if tp2 is not None and not (tp2 < tp1):
            errors.append(f"SHORT: TP2 ({tp2}) phải < TP1 ({tp1}).")
        liq = _num(facts.get("liq_short"))

    entry_mid = (entry_thap + entry_cao) / 2
    fee_rt = float(cfg["fee_rt"] or 0.0)
    if entry_mid and entry_mid > 0:
        reward_pct = abs(tp1 - entry_mid) / abs(entry_mid) * 100.0 - fee_rt
        risk_pct = abs(entry_mid - sl) / abs(entry_mid) * 100.0 + fee_rt
        if risk_pct <= 0:
            errors.append("Khoảng cách Entry tới SL bằng 0, không tính được R:R.")
        else:
            rr = reward_pct / risk_pct
            if rr < float(cfg["min_rr"]):
                errors.append(f"R:R TP1 sau phí {rr:.2f} < tối thiểu {float(cfg['min_rr']):.2f} (thưởng {reward_pct:.2f}%, rủi ro {risk_pct:.2f}%).")

    atr_1h = _num(facts.get("atr14_1h"))
    if atr_1h is None:
        print("[VALIDATOR] thiếu atr14_1h trong facts, bỏ qua kiểm tra SL theo ATR.", flush=True)
    elif atr_1h > 0:
        sl_dist_atr = abs(entry_mid - sl) / atr_1h
        if not (float(cfg["sl_atr_min"]) <= sl_dist_atr <= float(cfg["sl_atr_max"])):
            errors.append(
                f"Khoảng cách Entry tới SL {sl_dist_atr:.2f} ATR 1H nằm ngoài [{float(cfg['sl_atr_min']):.2f}, {float(cfg['sl_atr_max']):.2f}].")

    if liq is None:
        print("[VALIDATOR] thiếu giá thanh lý trong facts, bỏ qua kiểm tra thanh lý.", flush=True)
    else:
        sl_dist = abs(entry_mid - sl)
        liq_dist = abs(entry_mid - liq)
        if sl_dist > 0 and liq_dist < float(cfg["liq_sl_mult"]) * sl_dist:
            errors.append(
                f"Khoảng cách thanh lý ({liq_dist:.4g}) < {float(cfg['liq_sl_mult']):.1f} lần khoảng cách SL ({sl_dist:.4g}).")

    price = _num(facts.get("price"))
    if price is None:
        price = _num(facts.get("current_price"))
    if price is not None and trang_thai == "READY_TO_ENTER":
        atr_15m = _num(facts.get("atr14_15m"))
        tol = float(cfg["entry_ready_atr15"]) * atr_15m if atr_15m else 0.0
        if not (entry_thap - tol <= price <= entry_cao + tol):
            errors.append(f"READY_TO_ENTER nhưng giá {price} nằm ngoài vùng Entry [{entry_thap}, {entry_cao}].")
    if price is not None and trang_thai == "SETUP_WAITING_TRIGGER":
        atr_ref = atr_1h
        if atr_ref:
            dist = min(abs(price - entry_thap), abs(price - entry_cao))
            if entry_thap <= price <= entry_cao:
                dist = 0.0
            if dist > float(cfg["entry_wait_atr1h"]) * atr_ref:
                errors.append(f"SETUP_WAITING_TRIGGER nhưng giá {price} cách vùng Entry quá xa ({dist / atr_ref:.2f} ATR 1H).")

    cites = plan.get("dan_chung") or []
    if not isinstance(cites, list) or len(cites) < 3:
        errors.append(f"dan_chung cần tối thiểu 3 dẫn chứng, nhận được {len(cites) if isinstance(cites, list) else 0}.")
    else:
        tol = float(cfg["cite_rel_tol"])
        seen_levels = set()
        for item in cites:
            if not isinstance(item, dict):
                errors.append(f"Dẫn chứng {item!r} không phải object {{ref, value}}.")
                continue
            ref = str(item.get("ref") or "")
            if ref not in facts:
                errors.append(f"dan_chung ref lạ: {ref!r} không có trong packet.")
                continue
            fv = _num(facts.get(ref))
            cv = _num(item.get("value"))
            if fv is None or cv is None:
                errors.append(f"dan_chung {ref}: không so được số (packet={facts.get(ref)!r}, plan={item.get('value')!r}).")
                continue
            denom = abs(fv) if abs(fv) > 1e-12 else 1.0
            if abs(cv - fv) / denom > tol:
                errors.append(f"dan_chung {ref}: plan={cv} khác packet={fv} quá {tol * 100:.1f}%.")
            rl = ref.lower()
            if "sl" in rl or rl == "sl":
                seen_levels.add("sl")
            if "tp1" in rl or "tp_1" in rl or rl == "tp1":
                seen_levels.add("tp1")
            if "entry" in rl or "ema" in rl or "vwap" in rl or "_t0" in rl or "_t-" in rl or "prev_" in rl or "hh_" in rl or "ll_" in rl or "today_" in rl:
                seen_levels.add("entry")
        if "entry" not in seen_levels:
            errors.append("dan_chung thiếu dẫn chứng cho Entry.")
        if "sl" not in seen_levels and not any("sl" in str(plan.get("bang_chung", {}).get("sl", "")).lower() for _ in [0]):
            pass
    return errors
