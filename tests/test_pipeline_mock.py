import asyncio
import json
import os
import tempfile

# DB phải tách khỏi bot.db thật. Gán trực tiếp vào module vì analyze/evaluation_store
# đọc DB_PATH thành global tại thời điểm gọi.
_TMP = tempfile.mkdtemp()
_TEST_DB = os.path.join(_TMP, "test_bot.db")

import analyze  # noqa: E402
import evaluation_store  # noqa: E402

analyze.DB_PATH = _TEST_DB
evaluation_store.DB_PATH = _TEST_DB
analyze._prediction_db_initialized = False
analyze._auto_scan_db_initialized = False

VALID_PLAN = {
    "quyet_dinh": "LONG",
    "entry_thap": 59900.0, "entry_cao": 60100.0,
    "sl": 59400.0, "tp1": 61200.0, "tp2": 61800.0,
    "kich_hoat": "giá đã đóng nến vượt vùng",
    "bang_chung": {"entry": "ema20_1h", "sl": "đáy nến", "tp1": "đỉnh", "tp2": "đỉnh cao"},
    "dan_chung": [
        {"ref": "ema20_1h", "value": 59700.0},
        {"ref": "ema20_15m", "value": 59900.0},
        {"ref": "h_15m_t-3", "value": 60050.0},
    ],
    "rui_ro": ["nhiễu 15m"], "do_tin_cay": 72,
}

BAD_PLAN = dict(VALID_PLAN, sl=60500.0)  # SL sai phía cho LONG

FACTS = {
    "price": 60000.0, "current_price": 60000.0,
    "atr14_1h": 500.0, "atr14_15m": 180.0,
    "ema20_1h": 59700.0, "ema20_15m": 59900.0, "h_15m_t-3": 60050.0,
}


def _canned_ctx():
    return {
        "timeframe_data": {}, "system_prompt": "SP", "fear_greed_info": "",
        "current_price_str": "Giá hiện tại: 60,000", "current_price": 60000.0,
        "open_signals": [], "open_signal_context": None, "feature_block": "",
        "feature_snapshot": "FS", "decision_snapshot": "DS",
        "direction_scorecard": None, "direction_scorecard_payload": None,
        "market_snapshot": "MS", "user_prompt": "UP",
        "funding_context": None, "open_interest_context": None,
        "long_short_context": None,
        "market_context_block": None, "facts": dict(FACTS),
        "ref_levels": None, "derivs": None,
    }


def _run(coro):
    return asyncio.run(coro)


def test_manual_valid_plan_is_saved(monkeypatch, tmp_path):
    monkeypatch.setattr(analyze, "prepare_analysis_context", lambda *a, **k: _async(_canned_ctx()))
    monkeypatch.setattr(analyze, "request_json_analysis", lambda s, u: json.dumps(VALID_PLAN))
    out = _run(analyze.analyze_symbol("BTCUSDT", "futures", user_id=990001, chat_id=1))
    assert '"quyet_dinh": "BUY"' in out["text"]
    assert '"entry_thap": 59900' in out["text"]


def test_manual_invalid_plan_repaired_then_rejected(monkeypatch):
    # Model trả plan sai, lần sửa vẫn sai -> guard từ chối, không lưu.
    calls = {"n": 0}

    def fake_llm(s, u):
        calls["n"] += 1
        return json.dumps(BAD_PLAN)

    monkeypatch.setattr(analyze, "prepare_analysis_context", lambda *a, **k: _async(_canned_ctx()))
    monkeypatch.setattr(analyze, "request_json_analysis", fake_llm)
    out = _run(analyze.analyze_symbol("BTCUSDT", "futures", user_id=990002, chat_id=1))
    assert calls["n"] == 2  # 1 lần chính + 1 lần sửa
    assert "NO TRADE" in out["text"]
    assert "Bot đã tự lưu phân tích này" not in out["text"]


def test_manual_repair_recovers(monkeypatch):
    # Lần 1 sai, lần sửa đúng.
    calls = {"n": 0}

    def fake_llm(s, u):
        calls["n"] += 1
        return json.dumps(BAD_PLAN if calls["n"] == 1 else VALID_PLAN)

    monkeypatch.setattr(analyze, "prepare_analysis_context", lambda *a, **k: _async(_canned_ctx()))
    monkeypatch.setattr(analyze, "request_json_analysis", fake_llm)
    out = _run(analyze.analyze_symbol("BTCUSDT", "futures", user_id=990003, chat_id=1))
    assert calls["n"] == 2
    assert '"quyet_dinh": "BUY"' in out["text"]


def test_manual_no_trade_not_saved(monkeypatch):
    monkeypatch.setattr(analyze, "prepare_analysis_context", lambda *a, **k: _async(_canned_ctx()))
    monkeypatch.setattr(
        analyze, "request_json_analysis",
        lambda s, u: json.dumps({"quyet_dinh": "NO_TRADE", "ly_do": "trend chưa rõ"}))
    out = _run(analyze.analyze_symbol("BTCUSDT", "futures", user_id=990004, chat_id=1))
    assert "NO TRADE" in out["text"]
    assert "Bot đã tự lưu" not in out["text"]


def test_autoscan_valid_plan_sent(monkeypatch):
    monkeypatch.setattr(analyze, "collect_timeframe_data", lambda *a, **k: _async({}))
    monkeypatch.setattr(analyze, "_missing_critical_timeframes", lambda *a, **k: [])
    monkeypatch.setattr(analyze, "prepare_analysis_context", lambda *a, **k: _async(_canned_ctx()))
    monkeypatch.setattr(analyze, "request_json_analysis", lambda s, u: json.dumps(VALID_PLAN))
    monkeypatch.setattr(analyze, "_auto_scan_consume_trend_skip", lambda *a: None)

    class FakeLog:
        def __init__(self):
            self.entries = []

        async def __call__(self, stage, status, reason, **kw):
            self.entries.append((stage, status, reason))
            return {"send": False, "reason": reason, "stage": stage, "status": status, **kw}

    fake_log = FakeLog()
    result = _run(analyze._auto_scan_futures(
        symbol="BTCUSDT", mode="futures", user_id=990005, chat_id=1, scan_slot="s",
        ctx=_canned_ctx(), timeframe_data={}, system_prompt="SP", current_price=60000.0,
        market_snapshot="MS", feature_snapshot="FS",
        facts=dict(FACTS), log_and_return=fake_log,
    ))
    assert result["send"] is True
    assert result["direction"] == "LONG"
    assert "AUTO SCAN" in result["text"]


def test_autoscan_invalid_rejected(monkeypatch):
    calls = {"n": 0}

    def fake_llm(s, u):
        calls["n"] += 1
        return json.dumps(BAD_PLAN)

    monkeypatch.setattr(analyze, "request_json_analysis", fake_llm)

    class FakeLog:
        def __init__(self):
            self.entries = []

        async def __call__(self, stage, status, reason, **kw):
            self.entries.append((stage, status, reason))
            return {"send": False, "reason": reason, "stage": stage, "status": status, **kw}

    fake_log = FakeLog()
    result = _run(analyze._auto_scan_futures(
        symbol="BTCUSDT", mode="futures", user_id=990006, chat_id=1, scan_slot="s",
        ctx=_canned_ctx(), timeframe_data={}, system_prompt="SP", current_price=60000.0,
        market_snapshot="MS", feature_snapshot="FS",
        facts=dict(FACTS), log_and_return=fake_log,
    ))
    assert result["send"] is False
    assert calls["n"] == 2  # chính + sửa, cả 2 vẫn sai -> loại
    assert fake_log.entries[-1][0] == "guard"


def _async(value):
    async def _inner(*a, **k):
        return value
    return _inner()


BUY_JSON_PLAN = """{
  "quyet_dinh": "BUY",
  "entry_thap": 59900,
  "entry_cao": 60100,
  "sl": 59400,
  "tp1": 61200,
  "tp2": 61800,
  "kich_hoat": "Nến 4H đóng giữ hỗ trợ",
  "bang_chung": {"entry": "Hỗ trợ", "sl": "Đáy cấu trúc", "tp1": "Kháng cự", "tp2": "Đỉnh tuần"},
  "dan_chung": [{"ref": "l_4h_t0", "value": 59800}],
  "rui_ro": ["Thị trường đi ngang"],
  "do_tin_cay": 70
}"""


def test_spot_mode_returns_buy_json(monkeypatch):
    """Mode long chạy với template không còn dòng Trạng thái; snapshot ghi TRADE."""
    import sqlite3

    monkeypatch.setattr(analyze, "prepare_analysis_context", lambda *a, **k: _async(_canned_ctx()))
    monkeypatch.setattr(analyze, "request_json_analysis", lambda s, u: BUY_JSON_PLAN)
    out = _run(analyze.analyze_symbol("BTCUSDT", "spot", user_id=990007, chat_id=1))
    assert '"quyet_dinh": "BUY"' in out["text"]
    assert '"entry_thap": 59900' in out["text"]
    conn = sqlite3.connect(_TEST_DB)
    row = conn.execute(
        "SELECT planner_status FROM evaluation_cases WHERE mode='spot' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    assert row is not None and row[0] == "TRADE"


def test_autoscan_spot_sends_buy_json_and_records_signal(monkeypatch):
    """Giai đoạn 3: mock Auto Scan mode spot — gửi tín hiệu và ghi auto_scan_signals (hành vi gốc)."""
    import sqlite3

    import pandas as pd

    fake_dfs = {"1D": pd.DataFrame({"close": [1.0]})}
    monkeypatch.setattr(analyze, "collect_timeframe_data", lambda *a, **k: _async(fake_dfs))
    monkeypatch.setattr(analyze, "_missing_critical_timeframes", lambda *a, **k: [])
    monkeypatch.setattr(analyze, "prepare_analysis_context", lambda *a, **k: _async(_canned_ctx()))
    monkeypatch.setattr(analyze, "request_json_analysis", lambda s, u: BUY_JSON_PLAN)
    conn = sqlite3.connect(_TEST_DB)
    conn.execute("DELETE FROM auto_scan_signals")
    conn.commit()
    conn.close()

    result = _run(analyze.auto_scan_symbol_for_user("BTCUSDT", "spot", 990008, 5, scan_slot="s"))
    assert result.get("send") is True, result
    assert result.get("direction") == "BUY"
    assert '"quyet_dinh": "BUY"' in result["text"]
    conn = sqlite3.connect(_TEST_DB)
    n = conn.execute("SELECT COUNT(*) FROM auto_scan_signals").fetchone()[0]
    status = conn.execute(
        "SELECT planner_status FROM evaluation_cases WHERE mode='spot' ORDER BY id DESC LIMIT 1"
    ).fetchone()[0]
    conn.close()
    assert n == 1
    assert status == "TRADE"
