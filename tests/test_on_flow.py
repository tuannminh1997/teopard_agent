"""Test thứ tự nhập: đòn bẩy TRƯỚC rồi mới USDT; ép nhập lại khi dưới tối thiểu."""

import asyncio

import pytest

import analyze
import binance_executor
import key_store
import symbol_control

OK = "\u2705"
NO = "\u274c"


def _run(coro):
    """autoscan_pending_message ném ApplicationHandlerStop để nuốt tin — phải bắt."""
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


class _Q:
    def __init__(self, data):
        self.message = _M()
        self.data = data

    async def answer(self):
        pass


class _U:
    """Fake Update. `callback_query` là property trả về query đã gán."""

    def __init__(self, uid=77, text=""):
        self.id = uid
        self.effective_message = _M(text)
        self.message = self.effective_message
        self.effective_user = self
        self.effective_chat = _C()
        self._query = None

    @property
    def callback_query(self):
        return self._query

    def set_query(self, data):
        self._query = _Q(data)
        return self._query


class _C:
    id = 1


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    db = str(tmp_path / "on.db")
    monkeypatch.setattr(analyze, "DB_PATH", db)
    monkeypatch.setattr(analyze, "_prediction_db_initialized", False)
    monkeypatch.setattr(analyze, "_auto_scan_db_initialized", False)
    monkeypatch.setattr(key_store, "DB_PATH", db)
    from cryptography.fernet import Fernet
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    analyze.init_prediction_db()
    analyze.init_auto_scan_db()
    import auth
    monkeypatch.setattr(auth, "is_account_activated", lambda uid: True)
    yield
    symbol_control._AUTO_PENDING.clear()


def _set_min(monkeypatch, usdt_min, min_notional):
    """Ép mức tối thiểu cố định để test không phụ thuộc mạng."""
    monkeypatch.setattr(binance_executor, "min_order_usdt",
                        lambda *a, **k: {"usdt_min": usdt_min,
                                         "min_notional": min_notional})


def _stage(uid=77):
    return (symbol_control._AUTO_PENDING.get(uid) or {}).get("stage")


def _send(upd, text):
    upd.effective_message.text = text
    _run(symbol_control.autoscan_pending_message(upd, None))
    return upd.effective_message.last


def _make_key():
    key_store.save_api_keys(77, "APIKEY" + "x" * 30, "SECRET" + "y" * 30)


# ─── 1. Thứ tự: đòn bẩy TRƯỚC, USDT sau ───────────────────────────────────

def test_futures_asks_leverage_first(monkeypatch):
    _set_min(monkeypatch, 1.0, 20)
    _make_key()
    upd = _U()
    upd.set_query("asauto:yes:futures:ETHUSDT")
    _run(symbol_control.autoscan_auto_callback(upd, None))

    text = upd.callback_query.message.last
    assert "đòn bẩy" in text, "phải hỏi đòn bẩy trước, got: " + text
    assert "Nhập số USDT" not in text, "không được hỏi USDT trước khi có đòn bẩy"
    assert OK in text, "xác nhận phải có dấu tích"


def test_spot_asks_usdt_directly(monkeypatch):
    _set_min(monkeypatch, 5.0, 5)
    _make_key()
    upd = _U()
    upd.set_query("asauto:yes:spot:ETHUSDT")
    _run(symbol_control.autoscan_auto_callback(upd, None))

    text = upd.callback_query.message.last
    assert "Nhập số USDT" in text, "spot không có đòn bẩy -> hỏi USDT luôn"
    assert "đòn bẩy" not in text, "spot không được hỏi đòn bẩy"
    assert OK in text, text


# ─── 2. Ép nhập lại khi dưới tối thiểu ─────────────────────────────────────

def test_below_min_spot_forces_retry(monkeypatch):
    _set_min(monkeypatch, 5.0, 5.0)
    upd = _U()
    symbol_control._new_pending(77, stage="qty", market="spot", symbol="ETHUSDT")
    text = _send(upd, "1")

    assert NO in text, "từ chối phải có dấu ❌, got: " + text
    assert "Nhập lại" in text, text
    assert _stage() == "qty", "phải vẫn ở bước qty để nhập lại"
    assert analyze.get_auto_scan_market_settings(77, "spot") is None, \
        "chưa được bật phiên khi qty dưới tối thiểu"


def test_below_min_futures_forces_retry(monkeypatch):
    _set_min(monkeypatch, 2.5, 50.0)
    upd = _U()
    symbol_control._new_pending(77, stage="qty", market="futures", symbol="ETHUSDT")
    text = _send(upd, "1")

    assert NO in text, text
    assert "Nhập lại" in text, text
    assert _stage() == "qty"
    assert analyze.get_auto_scan_market_settings(77, "futures") is None


def test_short_qty_rejected_then_valid_accepted(monkeypatch):
    """Nhập 1 (từ chối) rồi nhập 10 (hợp lệ) → phiên phải bật được."""
    _set_min(monkeypatch, 5.0, 5.0)
    _make_key()
    upd = _U()
    symbol_control._new_pending(77, stage="qty", market="spot", symbol="ETHUSDT")

    text = _send(upd, "1")
    assert NO in text, text

    text = _send(upd, "10")
    assert OK in text, text
    s = analyze.get_auto_scan_market_settings(77, "spot")
    assert float(s["qty"]) == 10.0 and s["enabled"] is True


# ─── 3. Nhập đúng thì lưu ─────────────────────────────────────────────────

def test_valid_spot_saves(monkeypatch):
    _set_min(monkeypatch, 5.0, 5.0)
    _make_key()
    upd = _U()
    symbol_control._new_pending(77, stage="qty", market="spot", symbol="ETHUSDT")
    text = _send(upd, "50")

    assert OK in text, text
    s = analyze.get_auto_scan_market_settings(77, "spot")
    assert s["enabled"] is True and float(s["qty"]) == 50.0
    assert s["leverage"] == 1, "spot khong hoi don bay; luon bang 1"


def test_leverage_then_qty_futures(monkeypatch):
    monkeypatch.setattr(
        binance_executor, "min_order_usdt",
        lambda *a, **k: {"usdt_min": 50.0 / (k.get("leverage") or 1),
                         "min_notional": 50.0})
    _make_key()
    upd = _U()
    symbol_control._new_pending(77, stage="leverage", market="futures", symbol="ETHUSDT")

    # 1) Nhập đòn bẩy
    text = _send(upd, "20")
    assert OK in text, text
    assert _stage() == "qty", "nhập đòn bẩy xong phải chuyển sang hỏi qty"

    # 2) Prompt qty phải nhở mức tối thiểu THEO đòn bẩy vừa nhập (50/20 = 2.5)
    assert "2.5" in text, "mức tối thiểu phải tính theo đòn bẩy 20, got: " + text

    # 3) Nhập qty đủ -> lưu cả hai
    text = _send(upd, "10")
    assert OK in text, text
    s = analyze.get_auto_scan_market_settings(77, "futures")
    assert float(s["qty"]) == 10.0 and s["leverage"] == 20 and s["enabled"] is True


def test_leverage_then_short_qty_rejected(monkeypatch):
    monkeypatch.setattr(
        binance_executor, "min_order_usdt",
        lambda *a, **k: {"usdt_min": 50.0 / (k.get("leverage") or 1),
                         "min_notional": 50.0})
    upd = _U()
    symbol_control._new_pending(77, stage="leverage", market="futures", symbol="ETHUSDT")
    _send(upd, "20")
    text = _send(upd, "1")

    assert NO in text, text
    assert "Nhập lại" in text, text
    assert _stage() == "qty"
    assert analyze.get_auto_scan_market_settings(77, "futures") is None
