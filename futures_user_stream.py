"""Small Binance USD-M Futures user-stream supervisor.

The stream is an acceleration path; REST reconciliation remains authoritative
after reconnects and process restarts. It is deliberately lazy-imported so the
bot can still run without websocket-client when user streams are disabled.

Improvements:
- Event deduplication using (event_type, orderId/algoId, event_time)
- listenKeyExpired handling with immediate reconnect
- Exponential backoff reconnect logic
- Daemon thread crash protection with auto-restart
- Proper handling of ORDER_TRADE_UPDATE, ALGO_UPDATE, CONDITIONAL_ORDER_TRADE_UPDATE
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque

import requests

_LOCK = threading.Lock()
_THREADS: dict[int, threading.Thread] = {}
_STOP_EVENTS: dict[int, threading.Event] = {}


def _enabled() -> bool:
    return (os.getenv("FUTURES_USER_STREAM_ENABLED", "0").strip().lower()
            in {"1", "true", "yes", "on"})


def _ws_base(api_base: str) -> str:
    """Base WebSocket cho user stream futures.

    Theo docs Binance, host demo-fapi (demo trading) và testnet CÙNG dùng
    `wss://fstream.binancefuture.com` — khác host REST nhưng host WS giống nhau.
    Production dùng `wss://fstream.binance.com`.
    """
    if "demo-fapi" in api_base or "testnet" in api_base:
        return "wss://fstream.binancefuture.com/ws/"
    return "wss://fstream.binance.com/ws/"


def _listen_key(api_base: str, api_key: str) -> str:
    response = requests.post(
        f"{api_base}/fapi/v1/listenKey",
        headers={"X-MBX-APIKEY": api_key}, timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    key = data.get("listenKey")
    if not key:
        raise RuntimeError("Binance không trả listenKey Futures")
    return str(key)


def _keepalive(api_base: str, api_key: str, listen_key: str) -> None:
    requests.put(
        f"{api_base}/fapi/v1/listenKey",
        headers={"X-MBX-APIKEY": api_key}, params={"listenKey": listen_key}, timeout=15,
    ).raise_for_status()


def _dedup_key(event: dict) -> str | None:
    """Tạo khóa trùng lặp cho event. Trả None nếu không thể xác định.

    ORDER_TRADE_UPDATE và ALGO_UPDATE đều để payload trong `o`, nhưng field id KHÁC NHAU:
    - ORDER_TRADE_UPDATE: `i` = orderId, `c` = clientOrderId
    - ALGO_UPDATE: `aid` = algoId, `caid` = clientAlgoId
    """
    e_type = str(event.get("e") or "")
    inner = event.get("o") if isinstance(event.get("o"), dict) else {}
    if e_type == "ORDER_TRADE_UPDATE":
        order_id = inner.get("i") or inner.get("c")
        event_time = event.get("E") or event.get("T") or inner.get("T")
        if order_id and event_time:
            return f"order:{order_id}:{event_time}"
    elif e_type == "ALGO_UPDATE":
        algo_id = inner.get("aid") or inner.get("caid")
        event_time = event.get("E") or event.get("T") or inner.get("tt")
        if algo_id and event_time:
            return f"algo:{algo_id}:{event_time}"
    return None


class _DedupCache:
    """Cache trùng lặp đơn giản với giới hạn kích thước và TTL."""

    def __init__(self, max_size: int = 5000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._data: dict[str, float] = {}

    def is_duplicate(self, key: str) -> bool:
        now = time.time()
        # Dọn cache cũ
        if len(self._data) > self.max_size:
            self._data = {k: v for k, v in self._data.items() if now - v < self.ttl_seconds}
        if key in self._data:
            return True
        self._data[key] = now
        return False


_DEDUP = _DedupCache()


def _run(user_id: int, keys: tuple[str, str]) -> None:
    stop_event = _STOP_EVENTS.get(int(user_id))
    if not stop_event:
        stop_event = threading.Event()
        with _LOCK:
            _STOP_EVENTS[int(user_id)] = stop_event

    try:
        import websocket
    except ImportError:
        print("[FUTURES_STREAM] websocket-client missing; REST reconciliation remains active", flush=True)
        return

    from binance_executor import FUTURES_API_BASE

    api_key, _secret = keys
    reconnect_delay = 1.0
    max_reconnect_delay = 60.0

    while _enabled() and not stop_event.is_set():
        ws = None
        listen_key = None
        try:
            listen_key = _listen_key(FUTURES_API_BASE, api_key)
            ws = websocket.create_connection(
                _ws_base(FUTURES_API_BASE) + listen_key,
                timeout=35,
                enableTrace=False,
            )
            ws.settimeout(30)
            last_keepalive = time.time()
            reconnect_delay = 1.0  # Reset delay on successful connect
            print(f"[FUTURES_STREAM] user={user_id} connected", flush=True)

            try:
                while _enabled() and not stop_event.is_set():
                    if time.time() - last_keepalive >= 25 * 60:
                        try:
                            _keepalive(FUTURES_API_BASE, api_key, listen_key)
                        except Exception as exc:
                            print(f"[FUTURES_STREAM] user={user_id} keepalive failed: {exc}", flush=True)
                            break  # Force reconnect
                        last_keepalive = time.time()

                    try:
                        raw = ws.recv()
                    except Exception as exc:
                        if "timed out" in str(exc).lower():
                            continue
                        raise

                    if not raw:
                        raise RuntimeError("empty Futures user-stream frame")

                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    event_type = str(event.get("e") or "")

                    # Xử lý listenKeyExpired - cần reconnect ngay lập tức
                    if event_type == "listenKeyExpired":
                        print(f"[FUTURES_STREAM] user={user_id} listenKey expired, reconnecting...", flush=True)
                        break  # Force reconnect with new listenKey

                    # Deduplication
                    dedup_key = _dedup_key(event)
                    if dedup_key and _DEDUP.is_duplicate(dedup_key):
                        continue

                    if event_type in {"ORDER_TRADE_UPDATE", "ALGO_UPDATE",
                                      "CONDITIONAL_ORDER_TRADE_UPDATE"}:
                        from analyze import handle_futures_algo_update
                        try:
                            handle_futures_algo_update(user_id, keys, event)
                        except Exception as exc:
                            print(f"[FUTURES_STREAM] user={user_id} handler error: {exc}", flush=True)
            finally:
                if ws:
                    try:
                        ws.close()
                    except Exception:
                        pass
        except Exception as exc:
            print(f"[FUTURES_STREAM] user={user_id} disconnected: {exc}", flush=True)

        # Exponential backoff với jitter
        if _enabled() and not stop_event.is_set():
            sleep_time = min(reconnect_delay, max_reconnect_delay)
            time.sleep(sleep_time)
            reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)


def ensure_futures_user_stream(user_id: int, keys: tuple[str, str]) -> None:
    """Start at most one daemon stream per user; no-op unless explicitly enabled."""
    if not _enabled():
        return
    with _LOCK:
        uid = int(user_id)
        thread = _THREADS.get(uid)
        stop_event = _STOP_EVENTS.get(uid)
        if thread and thread.is_alive() and stop_event and not stop_event.is_set():
            return
        # Cleanup old stop event
        if stop_event:
            stop_event.set()
        stop_event = threading.Event()
        _STOP_EVENTS[uid] = stop_event
        thread = threading.Thread(target=_run, args=(uid, keys),
                                  name=f"futures-stream-{uid}", daemon=True)
        _THREADS[uid] = thread
        thread.start()


def stop_futures_user_stream(user_id: int, keys: tuple[str, str] | None = None) -> None:
    """Stop the user stream for a specific user and drop the listenKey.

    KHÔNG gửi listenKey lên DELETE khi `keys=None`; nếu có key thì gọi DELETE
    `/fapi/v1/listenKey` để không rò rỉ listenKey sau khi người dùng tắt autoscan.
    """
    uid = int(user_id)
    with _LOCK:
        stop_event = _STOP_EVENTS.get(uid)
        if stop_event:
            stop_event.set()
        thread = _THREADS.get(uid)
    if thread and thread.is_alive():
        thread.join(timeout=5)
    with _LOCK:
        if uid in _THREADS:
            del _THREADS[uid]
        if uid in _STOP_EVENTS:
            del _STOP_EVENTS[uid]
    if keys:
        try:
            from binance_executor import FUTURES_API_BASE
            requests.delete(f"{FUTURES_API_BASE}/fapi/v1/listenKey",
                            headers={"X-MBX-APIKEY": keys[0]}, params={}, timeout=10)
        except Exception:
            pass