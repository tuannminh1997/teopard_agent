"""Test /onfutu + /offfutu voi NHIỀU coin (toi da 3, cach nhau dau phay)."""
import asyncio

import pytest

import analyze
import binance_executor
import symbol_control

OK = "\u2705"
NO = "\u274c"


def _run(coro):
    try:
        asyncio.run(coro)
    except symbol_control.ApplicationHandlerStop:
        pass


class _M:
    chat_id = 1

    def __init__(self, text=""):
        self.text = text
        self.replies = []

    async def reply_text(self, text, *a, **k):
        self.replies.append(text)

    async def delete(self):
        return True

    @property
    def last(self):
        return self.replies[-1] if self.replies else ""


class _U:
    def __init__(self, uid=77, text=""):
        self.id = uid
        self.effective_message = _M(text)
        self.message = self.effective_message
        self.effective_user = self
        self.effective_chat = _C()


class _C:
    id = 1


class _Ctx:
    def __init__(self, args):
        self.args = args


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    db = str(tmp_path / "on.db")
    monkeypatch.setattr(analyze, "DB_PATH", db)
    monkeypatch.setattr(analyze, "_prediction_db_initialized", False)
    monkeypatch.setattr(analyze, "_auto_scan_db_initialized", False)
    analyze.init_auto_scan_db()
    monkeypatch.setattr(analyze, "_AUTO_SCAN_RUN_LOCK", asyncio.Lock())
    monkeypatch.setattr(analyze, "cancel_pending_plan_orders_for",
                        lambda uid, mkt, symbols=None: {"cancelled": [], "no_keys": []})
    monkeypatch.setattr(analyze, "delete_session_signals",
                        lambda uid, mkt, keep=False, symbol=None: 0)
    import auth
    monkeypatch.setattr(auth, "is_account_activated", lambda uid: True)
    # Danh sách symbol Binance GIẢ LẬP: eth/btc/xrp/sol co, notacoin khong co.
    fake = {"ETHUSDT", "BTCUSDT", "XRPUSDT", "SOLUSDT", "1000SHIBUSDT"}
    monkeypatch.setattr(binance_executor, "min_order_usdt",
                        lambda *a, **k: {"usdt_min": 1.0, "min_notional": 20.0})
    monkeypatch.setattr(analyze, "get_binance_symbols", lambda mkt, refresh=False: fake)
    tmp_path_holder[0] = monkeypatch.setattr
    yield
    symbol_control._AUTO_PENDING.clear()


class _Q:
    def __init__(self, data):
        self.message = _M()
        self.data = data

    async def answer(self):
        pass


class _UQ(_U):
    """Update co ca callback_query de mo phong nhan nut."""

    def __init__(self):
        super().__init__()
        self._query = None

    @property
    def callback_query(self):
        return self._query

    def set_query(self, data):
        self._query = _Q(data)
        return self._query

    @property
    def query(self):
        return self._query


def _enable(symbols, qty="100", leverage=20, market="futures"):
    analyze.set_auto_scan_market_enabled(77, 1, market, True, ",".join(symbols),
                                         qty=qty, leverage=leverage)


def _full_on(args, market="futures", leverage="20", qty="100"):
    """Chay tron luong: /onfutu -> bam nut -> nhap don bay -> nhap qty."""
    from cryptography.fernet import Fernet
    import os
    import key_store
    os.environ["DATA_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
    monkey_q = tmp_path_holder[0]
    monkey_q(key_store, "DB_PATH", analyze.DB_PATH)
    key_store.save_api_keys(77, "APIKEY" + "x" * 30, "SECRET" + "y" * 30)

    upd = _UQ()
    _run(symbol_control._autoscan_on_command(upd, _Ctx(args), market))
    # Mo phong user bam nut "Co": Telegram tu tao callback_query tu callback_data cua nut.
    from analyze import normalize_auto_scan_symbol
    joined = ",".join(normalize_auto_scan_symbol(a) for a in args[0].replace(",", " ").split())
    upd.set_query(f"asauto:yes:{market}:{joined}")
    _run(symbol_control.autoscan_auto_callback(upd, None))
    st = symbol_control._AUTO_PENDING.get(77) or {}
    if st.get("stage") == "leverage":
        _run(symbol_control.autoscan_pending_message(upd, None))
        upd.effective_message.text = leverage
        _run(symbol_control.autoscan_pending_message(upd, None))
    if (symbol_control._AUTO_PENDING.get(77) or {}).get("stage") == "qty":
        upd.effective_message.text = qty
        _run(symbol_control.autoscan_pending_message(upd, None))
    return upd.effective_message.last


tmp_path_holder = [None]


def _syms(market="futures"):
    cfg = analyze.get_auto_scan_market_settings(77, market)
    return analyze._parse_auto_scan_symbols_text((cfg or {}).get("symbol"))


# ─── ON: nhieu coin ────────────────────────────────────────────────────────

def test_on_accepts_three_coins():
    _full_on(["eth,btc,xrp"])
    assert _syms() == ["ETHUSDT", "BTCUSDT", "XRPUSDT"], _syms()
    cfg = analyze.get_auto_scan_market_settings(77, "futures")
    assert cfg["enabled"] is True and float(cfg["qty"]) == 100.0 and cfg["leverage"] == 20


def test_on_rejects_four_coins():
    upd = _U()
    _run(symbol_control._autoscan_on_command(upd, _Ctx(["eth,btc,xrp,sol"]), "futures"))
    text = upd.effective_message.last
    assert NO in text and "3" in text, text


def test_on_rejects_unknown_coin():
    upd = _U()
    _run(symbol_control._autoscan_on_command(upd, _Ctx(["eth,notacoin"]), "futures"))
    text = upd.effective_message.last
    assert NO in text, text
    assert "NOTACOIN" in text.upper(), text
    assert analyze.get_auto_scan_market_settings(77, "futures") is None, \
        "khong duoc bat phien khi co coin khong ton tai"


def test_on_spot_multi_coin():
    _full_on(["eth,btc"], market="spot")
    assert _syms("spot") == ["ETHUSDT", "BTCUSDT"], _syms("spot")
    assert analyze.get_auto_scan_market_settings(77, "spot")["enabled"] is True


def test_on_dedup_symbols():
    _full_on(["eth,ETH,btc"])
    assert _syms() == ["ETHUSDT", "BTCUSDT"], _syms()


# ─── OFF: tung coin ────────────────────────────────────────────────────────

def test_off_one_coin_keeps_others():
    _enable(["ETHUSDT", "BTCUSDT", "XRPUSDT"])
    upd = _U()
    _run(symbol_control._autoscan_off_command(upd, _Ctx(["eth"]), "futures"))

    cfg = analyze.get_auto_scan_market_settings(77, "futures")
    assert cfg["enabled"] is True, "con 2 coin thi phien van phai bat"
    assert _syms() == ["BTCUSDT", "XRPUSDT"], _syms()
    assert cfg["qty"] == "100" and cfg["leverage"] == 20, "qty/don bay khong duoc mat"
    text = upd.effective_message.last
    assert "ETH" in text and "BTC" in text, text


def test_off_two_coins_keeps_one():
    _enable(["ETHUSDT", "BTCUSDT", "XRPUSDT"])
    upd = _U()
    _run(symbol_control._autoscan_off_command(upd, _Ctx(["eth,btc"]), "futures"))
    assert _syms() == ["XRPUSDT"], _syms()
    assert analyze.get_auto_scan_market_settings(77, "futures")["enabled"] is True


def test_off_all_coins_disables_session():
    _enable(["ETHUSDT", "BTCUSDT", "XRPUSDT"])
    upd = _U()
    _run(symbol_control._autoscan_off_command(
        upd, _Ctx(["eth,btc,xrp"]), "futures"))
    cfg = analyze.get_auto_scan_market_settings(77, "futures")
    assert cfg["enabled"] is False, "het coin phai tat phien"
    text = upd.effective_message.last
    assert "hết coin" in text, text


def test_off_single_coin_session_disables():
    """Chi 1 coin thi /offfutu <coin> tat luon ca phien."""
    _enable(["ETHUSDT"])
    upd = _U()
    _run(symbol_control._autoscan_off_command(upd, _Ctx(["eth"]), "futures"))
    assert analyze.get_auto_scan_market_settings(77, "futures")["enabled"] is False


def test_off_coin_not_in_session_warns():
    _enable(["ETHUSDT", "BTCUSDT"])
    upd = _U()
    _run(symbol_control._autoscan_off_command(upd, _Ctx(["xrp"]), "futures"))
    text = upd.effective_message.last
    assert "không có trong phiên" in text, text
    assert "ETH" in text and "BTC" in text, text
    # Khong duoc tat nham coin khac
    assert _syms() == ["ETHUSDT", "BTCUSDT"], _syms()
    assert analyze.get_auto_scan_market_settings(77, "futures")["enabled"] is True


def test_off_multi_not_in_session_warns():
    _enable(["ETHUSDT"])
    upd = _U()
    _run(symbol_control._autoscan_off_command(upd, _Ctx(["btc,xrp"]), "futures"))
    assert "không có trong phiên" in upd.effective_message.last
    assert _syms() == ["ETHUSDT"]


def test_off_spot_per_coin():
    _enable(["ETHUSDT", "BTCUSDT"], market="spot")
    upd = _U()
    _run(symbol_control._autoscan_off_command(upd, _Ctx(["eth"]), "spot"))
    assert _syms("spot") == ["BTCUSDT"], _syms("spot")
    assert analyze.get_auto_scan_market_settings(77, "spot")["enabled"] is True


def test_off_futures_does_not_touch_spot():
    _enable(["ETHUSDT", "BTCUSDT"], market="futures")
    _enable(["ETHUSDT"], market="spot")
    upd = _U()
    _run(symbol_control._autoscan_off_command(upd, _Ctx(["eth"]), "futures"))
    assert _syms("spot") == ["ETHUSDT"], "tat futu khong duoc dung spot"
    assert analyze.get_auto_scan_market_settings(77, "spot")["enabled"] is True


def test_off_no_session_hint_uses_typed_symbol():
    upd = _U()
    _run(symbol_control._autoscan_off_command(upd, _Ctx(["btc"]), "futures"))
    text = upd.effective_message.last
    assert NO in text, text
    assert "/onfutu BTC" in text, text


def test_off_missing_args_shows_syntax_and_keeps_session():
    _enable(["ETHUSDT", "BTCUSDT"])
    upd = _U()
    _run(symbol_control._autoscan_off_command(upd, _Ctx([]), "futures"))
    text = upd.effective_message.last
    assert "/offfutu" in text, text
    assert "eth,btc" in text, text
    assert analyze.get_auto_scan_market_settings(77, "futures")["enabled"] is True


def test_off_preserves_config_across_partial_off():
    _enable(["ETHUSDT", "BTCUSDT", "XRPUSDT"], qty="250", leverage=50)
    _run(symbol_control._autoscan_off_command(_U(), _Ctx(["eth"]), "futures"))
    _run(symbol_control._autoscan_off_command(_U(), _Ctx(["btc"]), "futures"))
    cfg = analyze.get_auto_scan_market_settings(77, "futures")
    assert _syms() == ["XRPUSDT"]
    assert cfg["qty"] == "250" and cfg["leverage"] == 50, \
        "tat tung coin khong duoc mat qty/don bay"


# ─── Parse ────────────────────────────────────────────────────────────────

def test_parse_max_three():
    assert analyze._parse_auto_scan_symbols_text(
        "eth,btc,xrp,sol,ada") == ["ETHUSDT", "BTCUSDT", "XRPUSDT"]
    assert analyze._parse_auto_scan_symbols_text("") == []
    assert analyze._parse_auto_scan_symbols_text("eth; btc , xrp") == [
        "ETHUSDT", "BTCUSDT", "XRPUSDT"]
