"""Regression tests for multi-coin Auto Scan symbols and per-coin Binance minimums."""

import asyncio

import pytest

import analyze
import binance_executor
import symbol_control


class _Message:
    chat_id = 1

    def __init__(self, text=""):
        self.text = text
        self.replies = []

    async def reply_text(self, text, *args, **kwargs):
        self.replies.append(text)

    async def delete(self):
        return True

    @property
    def last(self):
        return self.replies[-1] if self.replies else ""


class _User:
    id = 77


class _Update:
    def __init__(self, text=""):
        self.effective_user = _User()
        self.effective_message = _Message(text)


@pytest.fixture(autouse=True)
def _db(monkeypatch, tmp_path):
    db = str(tmp_path / "dynamic_min.db")
    monkeypatch.setattr(analyze, "DB_PATH", db)
    monkeypatch.setattr(analyze, "_auto_scan_db_initialized", False)
    analyze.init_auto_scan_db()
    symbol_control._AUTO_PENDING.clear()
    yield
    symbol_control._AUTO_PENDING.clear()


def _run(coro):
    try:
        asyncio.run(coro)
    except symbol_control.ApplicationHandlerStop:
        pass


def test_min_order_usdt_ceils_quantity_to_step(monkeypatch):
    """Minimum suggestion must survive executor's later quantity floor."""
    fake = {
        "tick": 0.01,
        "step": 0.003,
        "min_qty": 0.003,
        "min_notional": 20.0,
        "base_asset": "ETH",
        "margin_asset": "USDT",
    }
    monkeypatch.setattr(binance_executor, "_market_filters", lambda *a, **k: dict(fake))
    info = binance_executor.min_order_usdt(
        "https://demo-api.binance.com", "ETHUSDT", True, price=2500.0)
    # 20 / 2500 = .008 ETH, ceil to .009 ETH because stepSize = .003.
    assert info["required_qty"] == pytest.approx(0.009)
    assert info["required_notional"] == pytest.approx(22.5)
    assert info["usdt_min"] == pytest.approx(22.5)


def test_dynamic_minimums_are_reported_for_every_coin(monkeypatch):
    values = {"ETHU": 0.25, "BTCU": 0.42, "XRPUSDT": 0.30}
    monkeypatch.setattr(binance_executor, "FUTURES_API_BASE",
                        "https://demo-fapi.binance.com")
    monkeypatch.setattr(analyze, "_resolve_demo_symbol",
                        lambda base, symbol: {"ETHUSDT": "ETHU", "BTCUSDT": "BTCU"}.get(symbol, symbol))
    monkeypatch.setattr(binance_executor, "min_order_usdt",
                        lambda base, symbol, spot, leverage=1: {
                            "usdt_min": values[symbol], "min_notional": 5.0,
                            "exchange_symbol": symbol,
                        })

    infos = analyze.auto_scan_min_order_infos("eth,btc,xrp", "futures", leverage=20)
    assert [x["analysis_symbol"] for x in infos] == ["ETHUSDT", "BTCUSDT", "XRPUSDT"]
    assert [x["exchange_symbol"] for x in infos] == ["ETHU", "BTCU", "XRPUSDT"]
    assert [x["usdt_min"] for x in infos] == [0.25, 0.42, 0.30]

    prompt = symbol_control._usdt_prompt("ETHUSDT,BTCUSDT,XRPUSDT", "futures", leverage=20)
    assert "ETH ≥ 0.25 USDT" in prompt
    assert "BTC ≥ 0.42 USDT" in prompt
    assert "XRP ≥ 0.30 USDT" in prompt
    assert "ít nhất ≈ 0.42 USDT" in prompt


def test_multi_coin_qty_must_reach_largest_minimum(monkeypatch):
    monkeypatch.setattr(analyze, "auto_scan_min_order_infos", lambda symbols, market, leverage=1: [
        {"analysis_symbol": "ETHUSDT", "usdt_min": 2.0},
        {"analysis_symbol": "BTCUSDT", "usdt_min": 7.5},
        {"analysis_symbol": "XRPUSDT", "usdt_min": 3.0},
    ])
    update = _Update("5")
    symbol_control._new_pending(
        77, stage="qty", market="spot", symbol="ETHUSDT,BTCUSDT,XRPUSDT")
    _run(symbol_control.autoscan_pending_message(update, None))
    assert "BTC ≥ 7.50 USDT" in update.effective_message.last
    assert "Nhập lại" in update.effective_message.last
    assert (symbol_control._AUTO_PENDING[77]["stage"] == "qty")
    assert analyze.get_auto_scan_market_settings(77, "spot") is None


def test_multi_coin_qty_above_largest_minimum_enables_session(monkeypatch):
    monkeypatch.setattr(analyze, "auto_scan_min_order_infos", lambda symbols, market, leverage=1: [
        {"analysis_symbol": "ETHUSDT", "usdt_min": 2.0},
        {"analysis_symbol": "BTCUSDT", "usdt_min": 7.5},
    ])
    update = _Update("8")
    symbol_control._new_pending(
        77, stage="qty", market="spot", symbol="ETHUSDT,BTCUSDT")
    _run(symbol_control.autoscan_pending_message(update, None))
    cfg = analyze.get_auto_scan_market_settings(77, "spot")
    assert cfg["enabled"] is True
    assert cfg["symbol"] == "ETHUSDT,BTCUSDT"
    assert float(cfg["qty"]) == 8.0


def test_symbol_normalization_is_market_specific():
    assert analyze.normalize_auto_scan_symbol("shib", "spot") == "SHIBUSDT"
    assert analyze.normalize_auto_scan_symbol("shib", "futures") == "1000SHIBUSDT"


def test_demo_futures_symbol_validation_accepts_exchange_alias(monkeypatch):
    monkeypatch.setattr(binance_executor, "FUTURES_API_BASE",
                        "https://demo-fapi.binance.com")
    monkeypatch.setattr(analyze, "get_binance_symbols", lambda market, refresh=False: frozenset({"ETHU"}))
    monkeypatch.setattr(analyze, "_resolve_demo_symbol", lambda base, symbol: "ETHU")
    ok, normalized = analyze.check_binance_symbol("eth", "futures")
    assert ok is True
    assert normalized == "ETHUSDT"


def test_cancel_pending_uses_exchange_symbol_for_demo_futures(monkeypatch):
    calls = []
    get_count = {"n": 0}

    def fake_signed(base, key, secret, method, path, params):
        calls.append((method, path, dict(params)))
        if method == "GET":
            get_count["n"] += 1
            if get_count["n"] == 1:
                return {"status": "NEW", "executedQty": "0"}
            raise binance_executor.ExecutorError(-2011, "gone")
        return {}

    monkeypatch.setattr(binance_executor, "signed_request", fake_signed)
    out = binance_executor.cancel_pending_plan_orders(
        ("k", "s"), "futures", [{
            "symbol": "ETHUSDT", "exchange_symbol": "ETHU",
            "entry_order_id": 99, "tp_algo_id": 101,
            "tp2_algo_id": None, "sl_algo_id": 102,
        }])
    assert out
    assert all(call[2].get("symbol") == "ETHU" for call in calls)
