"""Test cho loi lenh tat phien (offfutu/offspot) do user bao cao.

1. Dang bat ETH ma got /offfutu btc -> PHAI tu choi, khong tat nham phien ETH.
2. Chua bat phien ma got /offfutu eth -> PHAI tu choi, khong tat duoc.
3. Tat phien PHAI giu symbol/qty/leverage de lan sau khoi nhap lai.
4. Tat futures khong duoc dung toi phien spot va nguoc lai.
"""

import asyncio

import pytest

import analyze
import symbol_control


class _FakeMessage:
    chat_id = 1

    def __init__(self):
        self.replies = []

    async def reply_text(self, text, *a, **k):
        self.replies.append(text)

    @property
    def last(self):
        return self.replies[-1] if self.replies else ""


class _FakeChat:
    id = 1


class _FakeUser:
    id = 77


class _FakeUpdate:
    def __init__(self):
        self.effective_message = _FakeMessage()
        self.message = self.effective_message
        self.effective_user = _FakeUser()
        self.effective_chat = _FakeChat()


class _FakeCtx:
    def __init__(self, args):
        self.args = args


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    # File DB, KHONG phai :memory: — code mo nhieu connection toi cung mot path
    # nen :memory: tao DB rieng cho moi connection va bang "bien mat".
    db = str(tmp_path / "off.db")
    monkeypatch.setattr(analyze, "DB_PATH", db)
    monkeypatch.setattr(analyze, "_auto_scan_db_initialized", False)
    analyze.init_auto_scan_db()
    monkeypatch.setattr(analyze, "_AUTO_SCAN_RUN_LOCK", asyncio.Lock())
    monkeypatch.setattr(analyze, "cancel_pending_plan_orders_for",
                        lambda uid, mkt: {"cancelled": [], "no_keys": []})
    monkeypatch.setattr(analyze, "delete_session_signals", lambda uid, mkt, keep=False: 0)
    import auth
    monkeypatch.setattr(auth, "is_account_activated", lambda uid: True)


def _enable(market, symbol, qty="100", leverage=20, enabled=True):
    analyze.set_auto_scan_market_enabled(77, 1, market, enabled, symbol,
                                         qty=qty, leverage=leverage)


def test_off_wrong_symbol_refuses():
    _enable("futures", "ETHUSDT", qty="100", leverage=20)
    upd, ctx = _FakeUpdate(), _FakeCtx(["btc"])
    asyncio.run(symbol_control._autoscan_off_command(upd, ctx, "futures"))
    s = analyze.get_auto_scan_market_settings(77, "futures")
    assert s["enabled"] is True, "PHAI giu phien ETH chay, khong duoc tat nham"
    assert s["symbol"] == "ETHUSDT", "symbol khong duoc bi doi thanh BTC"
    assert s["qty"] == "100" and s["leverage"] == 20, "qty/leverage khong duoc xoa"
    text = upd.effective_message.last
    assert "ETHUSDT" in text, text
    assert "BTCUSDT" in text, text
    assert "/offfutu" in text, "phai goi y dung lenh tat"
    assert "ETH" in text, "phai goi y dung symbol dang chay"


def test_off_correct_symbol_disables():
    _enable("futures", "ETHUSDT", qty="100", leverage=20)
    asyncio.run(symbol_control._autoscan_off_command(
        _FakeUpdate(), _FakeCtx(["eth"]), "futures"))
    assert analyze.get_auto_scan_market_settings(77, "futures")["enabled"] is False


def test_off_never_enabled_refuses():
    _enable("futures", "ETHUSDT", enabled=False)
    upd, ctx = _FakeUpdate(), _FakeCtx(["eth"])
    asyncio.run(symbol_control._autoscan_off_command(upd, ctx, "futures"))
    assert analyze.get_auto_scan_market_settings(77, "futures")["enabled"] is False
    # Bao ro chua bat, va KHONG bao "da tat".
    assert "FUTURES" in upd.effective_message.last
    assert "Auto Scan" not in upd.effective_message.last.split("\n")[0], upd.effective_message.last


def test_off_no_session_refuses():
    upd, ctx = _FakeUpdate(), _FakeCtx(["eth"])
    asyncio.run(symbol_control._autoscan_off_command(upd, ctx, "futures"))
    assert "FUTURES" in upd.effective_message.last
    assert "onfutu" in upd.effective_message.last, "phai goi y lenh bat"


def test_off_missing_args_does_not_disable():
    _enable("futures", "ETHUSDT", enabled=True)
    upd, ctx = _FakeUpdate(), _FakeCtx([])
    asyncio.run(symbol_control._autoscan_off_command(upd, ctx, "futures"))
    assert "/offfutu" in upd.effective_message.last, upd.effective_message.last
    assert analyze.get_auto_scan_market_settings(77, "futures")["enabled"] is True, \
        "thieu symbol thi khong duoc tat luon"


def test_off_preserves_qty_and_leverage():
    _enable("futures", "ETHUSDT", qty="250", leverage=50)
    asyncio.run(symbol_control._autoscan_off_command(
        _FakeUpdate(), _FakeCtx(["eth"]), "futures"))
    s = analyze.get_auto_scan_market_settings(77, "futures")
    assert s["enabled"] is False
    assert s["qty"] == "250", "qty bi mat thanh %r" % s["qty"]
    assert s["leverage"] == 50, "leverage bi mat thanh %r" % s["leverage"]
    assert s["symbol"] == "ETHUSDT"


def test_off_wrong_symbol_keeps_config():
    _enable("futures", "ETHUSDT", qty="250", leverage=50)
    asyncio.run(symbol_control._autoscan_off_command(
        _FakeUpdate(), _FakeCtx(["btc"]), "futures"))
    s = analyze.get_auto_scan_market_settings(77, "futures")
    assert s["enabled"] is True and s["qty"] == "250" and s["leverage"] == 50


def test_off_spot_wrong_symbol_refuses():
    _enable("spot", "ETHUSDT", qty="270", leverage=1)
    upd, ctx = _FakeUpdate(), _FakeCtx(["btc"])
    asyncio.run(symbol_control._autoscan_off_command(upd, ctx, "spot"))
    assert analyze.get_auto_scan_market_settings(77, "spot")["enabled"] is True
    assert "ETHUSDT" in upd.effective_message.last


def test_off_futures_does_not_touch_spot():
    _enable("futures", "ETHUSDT", qty="100", leverage=20)
    _enable("spot", "BTCUSDT", qty="50", leverage=1)
    asyncio.run(symbol_control._autoscan_off_command(
        _FakeUpdate(), _FakeCtx(["eth"]), "futures"))
    assert analyze.get_auto_scan_market_settings(77, "spot")["enabled"] is True, \
        "tat futures ma spot bi tat theo"
    assert analyze.get_auto_scan_market_settings(77, "spot")["qty"] == "50"


def test_off_message_reports_active_symbol():
    _enable("futures", "ETHUSDT", qty="100", leverage=20)
    upd, ctx = _FakeUpdate(), _FakeCtx(["ETH"])
    asyncio.run(symbol_control._autoscan_off_command(upd, ctx, "futures"))
    assert "ETHUSDT" in upd.effective_message.last
    assert "BTCUSDT" not in upd.effective_message.last

