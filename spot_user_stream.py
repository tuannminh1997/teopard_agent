"""Binance Spot user-stream supervisor.

The stream is an acceleration path; REST reconciliation remains authoritative
after reconnects and process restarts. It is deliberately lazy-imported so the
bot can still run without websocket-client when user streams are disabled.

Differences vs Futures stream:
- Events are REST-style: executionReport / listStatus / outboundAccountPosition
- OCO lists use orderListId/listClientOrderId instead of algoId/clientAlgoId
- A single symbol can hold several plans; events must be mapped by clientOrderId
  so one plan's cancel never touches another plan's protection

Improvements:
- Event deduplication using (event_type, clientOrderId/lastOrderId, event_time)
- listenKeyExpired handling with immediate reconnect
- Exponential backoff reconnect logic
- Daemon thread crash protection with auto-restart
- Stream failure never fails an order — REST reconciliation covers it
"""
from __future__ import annotations

import json
import os
import threading
import time

import requests

_LOCK = threading.Lock()
_THREADS: dict[int, threading.Thread] = {}
_STOP_EVENTS: dict[int, threading.Event] = {}


def _enabled() -> bool:
    return (os.getenv("SPOT_USER_STREAM_ENABLED", "0").strip().lower()
            in {"1", "true", "yes", "on"})


def _ws_base(api_base: str) -> str:
    if "demo-api" in api_base:
        return "wss://demo-stream.binance.com/ws/"
    if "testnet" in api_base:
        return "wss://testnet.binance.vision/ws/"
    return "wss://stream.binance.com:9443/ws/"


def _listen_key(api_base: str, api_key: str) -> str:
    response = requests.post(
        f"{api_base}/api/v3/userDataStream",
        headers={"X-MBX-APIKEY": api_key}, timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    key = data.get("listenKey")
    if not key:
        raise RuntimeError("Binance không trả listenKey Spot")
    return str(key)


def _keepalive(api_base: str, api_key: str, listen_key: str) -> None:
    requests.put(
        f"{api_base}/api/v3/userDataStream",
        headers={"X-MBX-APIKEY": api_key}, params={"listenKey": listen_key}, timeout=15,
    ).raise_for_status()


def _dedup_key(event: dict) -> str | None:
    """Tạo khóa trùng lặp cho event Spot. Trả None nếu không thể xác định.

    Theo docs Spot user-data-stream:
    - executionReport: `i` = orderId, `c` = clientOrderId, `E` = event time
    - listStatus: `g` = orderListId, `C` = listClientOrderId, `E` = event time
    - outboundAccountPosition: `E` = event time (không có order id)
    """
    e_type = str(event.get("e") or "")
    event_time = event.get("E") or event.get("T")
    if e_type == "executionReport":
        order_id = event.get("i") or event.get("c")
        if order_id and event_time:
            return f"exec:{order_id}:{event_time}"
    elif e_type == "listStatus":
        list_id = event.get("g") or event.get("C")
        if list_id and event_time:
            return f"list:{list_id}:{event_time}"
    elif e_type == "outboundAccountPosition":
        if event_time:
            return f"bal:{event_time}"
    return None


class _DedupCache:
    """Cache trùng lặp đơn giản với giới hạn kích thước và TTL."""

    def __init__(self, max_size: int = 5000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._data: dict[str, float] = {}

    def is_duplicate(self, key: str) -> bool:
        now = time.time()
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
        print("[SPOT_STREAM] websocket-client missing; REST reconciliation remains active", flush=True)
        return

    from binance_executor import SPOT_API_BASE

    api_key, _secret = keys
    reconnect_delay = 1.0
    max_reconnect_delay = 60.0

    while _enabled() and not stop_event.is_set():
        ws = None
        listen_key = None
        try:
            listen_key = _listen_key(SPOT_API_BASE, api_key)
            ws = websocket.create_connection(
                _ws_base(SPOT_API_BASE) + listen_key,
                timeout=35,
                enableTrace=False,
            )
            ws.settimeout(30)
            last_keepalive = time.time()
            reconnect_delay = 1.0  # Reset delay on successful connect
            print(f"[SPOT_STREAM] user={user_id} connected", flush=True)

            try:
                while _enabled() and not stop_event.is_set():
                    if time.time() - last_keepalive >= 25 * 60:
                        try:
                            _keepalive(SPOT_API_BASE, api_key, listen_key)
                        except Exception as exc:
                            print(f"[SPOT_STREAM] user={user_id} keepalive failed: {exc}", flush=True)
                            break  # Force reconnect
                        last_keepalive = time.time()

                    try:
                        raw = ws.recv()
                    except Exception as exc:
                        if "timed out" in str(exc).lower():
                            continue
                        raise

                    if not raw:
                        raise RuntimeError("empty Spot user-stream frame")

                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    event_type = str(event.get("e") or "")

                    # listenKeyExpired → reconnect ngay lập tức
                    if event_type == "listenKeyExpired":
                        print(f"[SPOT_STREAM] user={user_id} listenKey expired, reconnecting...", flush=True)
                        break

                    # Deduplication — event trùng (reconnect/mạng) không được xử lý 2 lần
                    dedup_key = _dedup_key(event)
                    if dedup_key and _DEDUP.is_duplicate(dedup_key):
                        continue

                    if event_type in {"executionReport", "listStatus", "outboundAccountPosition"}:
                        from analyze import handle_spot_user_event
                        try:
                            handle_spot_user_event(user_id, keys, event)
                        except Exception as exc:
                            print(f"[SPOT_STREAM] user={user_id} handler error: {exc}", flush=True)
            finally:
                if ws:
                    try:
                        ws.close()
                    except Exception:
                        pass
        except Exception as exc:
            print(f"[SPOT_STREAM] user={user_id} disconnected: {exc}", flush=True)

        # Exponential backoff với jitter
        if _enabled() and not stop_event.is_set():
            sleep_time = min(reconnect_delay, max_reconnect_delay)
            time.sleep(sleep_time)
            reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)


def ensure_spot_user_stream(user_id: int, keys: tuple[str, str]) -> None:
    """Start at most one daemon stream per user; no-op unless explicitly enabled."""
    if not _enabled():
        return
    with _LOCK:
        uid = int(user_id)
        thread = _THREADS.get(uid)
        stop_event = _STOP_EVENTS.get(uid)
        if thread and thread.is_alive() and stop_event and not stop_event.is_set():
            return
        if stop_event:
            stop_event.set()
        stop_event = threading.Event()
        _STOP_EVENTS[uid] = stop_event
        thread = threading.Thread(target=_run, args=(uid, keys),
                                  name=f"spot-stream-{uid}", daemon=True)
        _THREADS[uid] = thread
        thread.start()


def stop_spot_user_stream(user_id: int, keys: tuple[str, str] | None = None) -> None:
    """Stop the user stream for a specific user and drop the listenKey.

    KHÔNG gửi listenKey lên DELETE khi `keys=None`; nếu có key thì gọi DELETE
    `/api/v3/userDataStream` để không rò rỉ listenKey sau khi người dùng tắt autoscan.
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
            from binance_executor import SPOT_API_BASE
            requests.delete(f"{SPOT_API_BASE}/api/v3/userDataStream",
                            headers={"X-MBX-APIKEY": keys[0]}, timeout=10)
        except Exception:
            pass
