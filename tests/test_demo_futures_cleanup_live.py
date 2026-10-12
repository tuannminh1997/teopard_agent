"""Opt-in integration test against Binance Futures Demo.

Run explicitly with:
    $env:FUTURES_API_BASE='https://demo-fapi.binance.com'
    $env:RUN_DEMO_FUTURES_TESTS='1'
    py -m pytest -q tests/test_demo_futures_cleanup_live.py

The test only creates unreachable LIMIT/protection orders with a unique client
prefix, then verifies per-plan cleanup. It never touches pre-existing orders.
"""
import os
import time

import pytest
from dotenv import load_dotenv

load_dotenv()

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DEMO_FUTURES_TESTS", "0").lower() not in {"1", "true", "yes", "on"},
    reason="explicit demo integration test is disabled",
)


def _setup():
    import binance_executor as ex

    base = os.getenv("FUTURES_API_BASE", "")
    if "demo-fapi.binance.com" not in base:
        pytest.fail(f"Refusing non-demo Futures base: {base!r}")
    keys = (os.getenv("KEY_API", "").strip(), os.getenv("SECRET_API", "").strip())
    if not all(keys):
        pytest.fail("KEY_API/SECRET_API chưa có trong môi trường test")
    return ex, keys


def _open_by_prefix(ex, keys, symbol, prefix):
    rows = ex.open_futures_algo_orders(keys, symbol)
    assert rows is not None, "Không đọc được openAlgoOrders"
    return [r for r in rows if str(r.get("clientAlgoId") or "").startswith(prefix)]


def _cancel_entry(ex, keys, symbol, entry_id):
    if not entry_id:
        return
    try:
        ex.signed_request(ex.FUTURES_API_BASE, keys[0], keys[1], "DELETE",
                          "/fapi/v1/order", {"symbol": symbol, "orderId": entry_id})
    except ex.ExecutorError as exc:
        if exc.code not in (-2011, 404):
            raise


def test_demo_futures_cleanup_matrix_isolated_by_plan():
    ex, keys = _setup()
    symbol = "ETHU"
    prefix = f"codex-clean-{int(time.time() * 1000)}-"
    created = []

    def create(plan, tp2=None):
        result = ex.place_futures_plan(
            symbol=symbol, direction="LONG", entry_price=2000.0,
            tp1=3000.0, sl=1900.0, qty=0.003, leverage=20,
            keys=keys, plan_id=prefix + plan,
            tp2=(3100.0 if tp2 is not None else None),
        )
        created.append(result)
        rows = _open_by_prefix(ex, keys, symbol, prefix + plan)
        return result, rows

    try:
        # Plan A: có TP2. Xóa TP1 trước vẫn phải giữ TP2 + SL.
        a, a_rows = create("a", tp2=5200.0)
        a_ids = {str(r["algoId"]): r for r in a_rows}
        assert len(a_ids) == 3
        ex.cancel_futures_algo_orders(keys, symbol, [a["tp_algo_id"]])
        left = _open_by_prefix(ex, keys, symbol, prefix + "a")
        assert {str(r["algoId"]) for r in left} == {
            str(a["tp2_algo_id"]), str(a["sl_algo_id"])
        }
        ex.cancel_futures_algo_orders(keys, symbol, [a["tp2_algo_id"], a["sl_algo_id"]])
        assert _open_by_prefix(ex, keys, symbol, prefix + "a") == []

        # Plan B: không có TP2. Xóa TP1 rồi xóa SL; không được còn protection.
        b, b_rows = create("b")
        assert len(b_rows) == 2
        ex.cancel_futures_algo_orders(keys, symbol, [b["tp_algo_id"]])
        assert {str(r["algoId"]) for r in _open_by_prefix(ex, keys, symbol, prefix + "b")} == {
            str(b["sl_algo_id"])
        }
        ex.cancel_futures_algo_orders(keys, symbol, [b["sl_algo_id"]])
        assert _open_by_prefix(ex, keys, symbol, prefix + "b") == []

        # Plan C/D/E trộn có/không TP2: cleanup từng plan không chạm plan khác.
        c, _ = create("c", tp2=5200.0)
        d, _ = create("d")
        e, _ = create("e", tp2=5200.0)
        ex.cancel_futures_algo_orders(keys, symbol, [c["sl_algo_id"]])
        remaining_d = _open_by_prefix(ex, keys, symbol, prefix + "d")
        remaining_e = _open_by_prefix(ex, keys, symbol, prefix + "e")
        assert len(remaining_d) == 2
        assert len(remaining_e) == 3
        ex.cancel_futures_algo_orders(
            keys, symbol,
            [c["tp_algo_id"], c["tp2_algo_id"], d["tp_algo_id"], d["sl_algo_id"],
             e["tp_algo_id"], e["tp2_algo_id"], e["sl_algo_id"]],
        )
        assert _open_by_prefix(ex, keys, symbol, prefix + "c") == []
        assert _open_by_prefix(ex, keys, symbol, prefix + "d") == []
        assert _open_by_prefix(ex, keys, symbol, prefix + "e") == []
    finally:
        for order in created:
            ex.cancel_futures_algo_orders(keys, symbol, [
                order.get("tp_algo_id"), order.get("tp2_algo_id"), order.get("sl_algo_id")
            ])
            _cancel_entry(ex, keys, symbol, order.get("entry_order_id"))
