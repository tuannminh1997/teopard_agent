import pandas as pd

import analyze
from plan_validator import validate_plan


def _sample_df(n=120, start="2026-09-01 00:00"):
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    base = 60000.0
    rows = []
    for i, ts in enumerate(idx):
        o = base + i * 2.0
        h = o + 60.0
        l = o - 60.0
        c = o + 10.0
        rows.append({
            "timestamp": ts, "open": o, "high": h, "low": l, "close": c,
            "volume": 100.0 + (i % 5), "close_time": ts + pd.Timedelta(hours=1),
            "quote_volume": 1000.0, "count": 10,
            "taker_buy_volume": 55.0, "taker_buy_quote_volume": 550.0, "ignore": 0,
        })
    return pd.DataFrame(rows)


def test_intraday_indicators():
    df = analyze.add_indicators_intraday(_sample_df(250))
    assert df is not None and not df.empty
    for col in ("ema_20", "ema_50", "rsi_14", "atr_14", "vol_ratio", "rng", "cl_pct"):
        assert col in df.columns
    assert (df["atr_14"] > 0).all()


def test_intraday_packet_facts_cover_printed_refs():
    import re
    dfs = {
        "4H": analyze.add_indicators_intraday(_sample_df(320)),
        "1H": analyze.add_indicators_intraday(_sample_df(320)),
        "15m": analyze.add_indicators_intraday(_sample_df(320)),
    }
    ref = {"prev_day_high": 61000.0, "prev_day_low": 59000.0, "prev_day_close": 60000.0,
           "today_open": 60000.0, "today_high": 60500.0, "today_low": 59500.0,
           "prev_week_high": 62000.0, "prev_week_low": 58000.0}
    derivs = {"funding_hist": [0.01, 0.01, 0.02, 0.01], "funding": {"latest_pct": 0.01, "history_pct": [0.01, 0.01, 0.02, 0.01]},
              "oi_chg_1h": 1.0, "price_chg_1h": 0.5, "long_short_top": 1.2, "long_short_crowd": 1.0,
              "taker_buy_pct_1h": 52.0}
    text, facts = analyze.build_intraday_packet(dfs, ref, derivs, None, 60000.0, symbol="BTCUSDT")
    assert "OBJECTIVE_MARKET_PACKET" in text
    assert "RỦI RO ĐÒN BẨY" in text
    for key in ("ema20_1h", "ema50_4h", "atr14_1h", "prev_day_high", "liq_long", "liq_short"):
        assert key in facts, key
    assert re.search(r"== 4H:", text) and re.search(r"== 1H:", text) and re.search(r"== 15m:", text)


def test_intraday_packet_missing_ref_levels():
    dfs = {
        "4H": analyze.add_indicators_intraday(_sample_df(320)),
        "1H": analyze.add_indicators_intraday(_sample_df(320)),
        "15m": analyze.add_indicators_intraday(_sample_df(320)),
    }
    text, facts = analyze.build_intraday_packet(dfs, {}, {}, None, 60000.0, symbol="BTCUSDT")
    assert "OBJECTIVE_MARKET_PACKET" in text
    assert "prev_day_high" not in facts


def test_long_mode_untouched():
    assert analyze._mode_frame_roles("long") == ("1D", "1W", "1M")
    assert "1M" in analyze.LONG_TERM_TIMEFRAMES
    prompt = analyze.load_system_prompt("long")
    assert "QUYẾT ĐỊNH" in prompt


def test_system_prompt_placeholders_replaced():
    prompt = analyze.load_system_prompt("short")
    for token in ("{MIN_RR}", "{SL_ATR_MIN}", "{SL_ATR_MAX}", "{LIQ_SL_MULT}", "{FEE_RT}",
                  "{ENTRY_READY_ATR15}", "{CITE_TOL_PCT}", "{MAX_SL_PCT}"):
        assert token not in prompt
    assert "0.25 lần atr14_15m" in prompt
    assert "0.5%" in prompt
    assert "2% giá" in prompt


def test_prompts_have_two_states_only():
    from evaluation_store import normalize_decision_status
    for path in ("analyze_system_prompt.txt", "analyze_system_prompt_long.txt"):
        text = open(path, encoding="utf-8").read()
        for banned in ("READY_TO_ENTER", "SETUP_WAITING_TRIGGER", "STATUS_PARSE_ERROR",
                       "Trạng thái:", "trang_thai"):
            assert banned not in text, f"{path}: còn {banned}"
    # Dữ liệu cũ đi qua normalize; không tự đoán lại.
    assert normalize_decision_status("READY_TO_ENTER") == "TRADE"
    assert normalize_decision_status("TRADE") == "TRADE"
    assert normalize_decision_status("NO_TRADE") == "NO_TRADE"
    assert normalize_decision_status("NO TRADE") == "NO_TRADE"
    assert normalize_decision_status("SETUP_WAITING_TRIGGER") is None
    assert normalize_decision_status("STATUS_PARSE_ERROR") is None
    assert normalize_decision_status(None) is None


def test_no_legacy_status_strings_in_flow_code():
    files = ("analyze.py", "plan_validator.py", "symbol_control.py",
             "bot.py", "auth.py", "mess_control.py")
    banned = ("READY_TO_ENTER", "SETUP_WAITING_TRIGGER", "STATUS_PARSE_ERROR",
              "chờ trigger", "Trigger đã sẵn sàng")
    for fname in files:
        text = open(fname, encoding="utf-8").read()
        for token in banned:
            assert token not in text, f"{fname}: còn {token}"


def _base_facts(price=60000.0, atr=500.0):
    return {
        "price": price, "current_price": price,
        "atr14_1h": atr, "atr14_15m": 180.0,
        "liq_long": price * 0.955, "liq_short": price * 1.045,
        "ema20_1h": price - 300.0, "ema20_15m": price - 100.0,
        "prev_day_high": price + 800.0, "h_15m_t-3": price + 50.0, "c_1h_t0": price - 20.0,
    }


def _base_plan(price=60000.0):
    return {
        "quyet_dinh": "LONG",
        "entry_thap": price - 100.0, "entry_cao": price + 100.0,
        "sl": price - 600.0, "tp1": price + 1200.0, "tp2": price + 1800.0,
        "kich_hoat": "gia nam trong vung", "bang_chung": {"entry": "x", "sl": "y", "tp1": "z"},
        "dan_chung": [
            {"ref": "ema20_1h", "value": price - 300.0},
            {"ref": "ema20_15m", "value": price - 100.0},
            {"ref": "h_15m_t-3", "value": price + 50.0},
        ],
        "rui_ro": ["nhieu"], "do_tin_cay": 70,
    }


def test_validator_accepts_good_plan():
    assert validate_plan(_base_plan(), _base_facts()) == []


def test_validator_rejects_wrong_side_sl():
    plan = _base_plan()
    plan["sl"] = 60000.0 + 500.0
    assert validate_plan(plan, _base_facts())


def test_validator_rejects_low_rr():
    plan = _base_plan()
    plan["tp1"] = 60000.0 + 200.0
    assert validate_plan(plan, _base_facts())


def test_validator_rejects_unknown_ref():
    plan = _base_plan()
    plan["dan_chung"] = [{"ref": "ma_tran", "value": 1}, {"ref": "ema20_1h", "value": 59700.0}, {"ref": "h_15m_t-3", "value": 60050.0}]
    assert validate_plan(plan, _base_facts())


def test_validator_entry_near_price_for_every_trade():
    # Giá trong vùng ± sai số 0.25*atr15m → đạt.
    assert validate_plan(_base_plan(), _base_facts()) == []
    # Giá nằm ngoài vùng Entry → mọi lệnh LONG/SHORT đều bị lỗi (không còn trạng thái "chờ").
    facts = _base_facts()
    facts["price"] = facts["current_price"] = 60000.0 - 1000.0
    errors = validate_plan(_base_plan(), facts)
    assert any("nằm ngoài vùng Entry" in e for e in errors)


def test_render_roundtrip():
    plan = _base_plan()
    text = analyze.render_plan_text(plan, "BTCUSDT", "INTRADAY", 60000.0)
    parsed = analyze.parse_prediction_from_output(text)
    assert parsed["direction"] == "LONG"
    assert abs(parsed["entry_low"] - 59900.0) < 1e-6
    assert abs(parsed["entry_high"] - 60100.0) < 1e-6
    assert abs(parsed["sl"] - 59400.0) < 1e-6
    assert abs(parsed["tp1"] - 61200.0) < 1e-6


def test_render_roundtrip_small_price():
    plan = _base_plan(price=0.08423)
    plan.update({"entry_thap": 0.08300, "entry_cao": 0.08400, "sl": 0.08200, "tp1": 0.08600, "tp2": 0.08800,
                 "dan_chung": [{"ref": "a", "value": 1}, {"ref": "b", "value": 2}, {"ref": "c", "value": 3}]})
    text = analyze.render_plan_text(plan, "XUSDT", "INTRADAY", 0.08423)
    parsed = analyze.parse_prediction_from_output(text)
    assert abs(parsed["entry_low"] - 0.08300) < 1e-9
    assert abs(parsed["tp1"] - 0.08600) < 1e-9
