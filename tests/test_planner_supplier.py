"""OpenRouter supplier failover (PLANNER_SUPPLIER).

Tests khẳng định:
- Không có PLANNER_SUPPLIER → payload KHÔNG có `provider` (hành vi cũ).
- Có PLANNER_SUPPLIER → mỗi request giới hạn đúng 1 nhà qua `provider.only`
  và `allow_fallbacks=false`.
- Nhà đầu timeout/lỗi 5xx → tự chuyển sang nhà tiếp theo trong danh sách.
- Lỗi 4xx (400/401/404/422) → KHÔNG đổi nhà, raise ngay (đổi cũng vậy).
- Thứ tự trong list = thứ tự ưu tiên.
- `_parse_supplier_list` lọc trùng/rỗng và giữ đúng thứ tự.
"""
import json

import pytest
import requests

import analyze


class _Resp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} error", response=self)


def _ok_payload(text="NO_TRADE"):
    return {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


@pytest.fixture
def no_key(monkeypatch):
    """Cho phép test không cần OPENROUTER_API_KEY thật."""
    monkeypatch.setattr(analyze, "OPENROUTER_API_KEY", "test-key")
    return monkeypatch


def _fake_post(monkeypatch, script):
    """script: list of (exception_or_None, status_code_or_None). Ghi lại payload mỗi lần."""
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        action, status = script[len(calls) - 1] if len(calls) <= len(script) else (None, 200)
        if action is not None:
            raise action
        if status:
            return _Resp(status, text="upstream error")
        return _Resp(200, _ok_payload())

    monkeypatch.setattr(analyze.requests, "post", fake_post)
    return calls


# ─── _parse_supplier_list ─────────────────────────────────────────────────────

def test_parse_supplier_list_normalizes_slugs():
    """Ten mien (inference.net) phai duoc doi thanh slug OpenRouter (inference-net).

    Neu khong doi, provider.only khong khop provider nao va OpenRouter tra 404
    "No allowed providers" — chan toan bo phan tich ke ca luong manual.
    """
    assert analyze._parse_supplier_list("inference.net, Modal, Baseten") == \
        ["inference-net", "modal", "baseten"]


def test_parse_supplier_list_dedup_and_order():
    # Chu thuong + loc trung ca khi viet khac dang (inference.net va inference-net la mot).
    assert analyze._parse_supplier_list("inference.net, Modal , Baseten ,inference-net") == \
        ["inference-net", "modal", "baseten"]


def test_parse_supplier_list_empty_and_separators():
    assert analyze._parse_supplier_list("") == []
    assert analyze._parse_supplier_list("   ") == []
    assert analyze._parse_supplier_list(None) == []  # type: ignore[arg-type]
    assert analyze._parse_supplier_list("a;b, c") == ["a", "b", "c"]


# ─── Không có PLANNER_SUPPLIER: giữ nguyên hành vi cũ ─────────────────────────

def test_no_supplier_sends_no_provider_field(monkeypatch, no_key):
    monkeypatch.setattr(analyze, "PLANNER_SUPPLIER", [])
    calls = _fake_post(monkeypatch, [(None, None)])

    analyze.llm_create_once("sys", [{"role": "user", "content": "hi"}],
                            max_tokens=10, timeout=5)

    assert len(calls) == 1
    assert "provider" not in calls[0], "không có PLANNER_SUPPLIER thì không được gửi provider"


# ─── Có PLANNER_SUPPLIER: giới hạn đúng 1 nhà mỗi request ─────────────────────

def test_supplier_restricts_request(monkeypatch, no_key):
    monkeypatch.setattr(analyze, "PLANNER_SUPPLIER", ["inference.net", "Modal", "Baseten"])
    calls = _fake_post(monkeypatch, [(None, None)])

    result = analyze.llm_create_once("sys", [{"role": "user", "content": "hi"}],
                                     max_tokens=10, timeout=5)

    assert len(calls) == 1, "nhà đầu tiên thành công thì chỉ gọi 1 lần"
    assert calls[0]["provider"] == {"only": ["inference.net"], "allow_fallbacks": False}
    assert result["provider"] == "inference.net"


def test_supplier_order_is_priority_order(monkeypatch, no_key):
    monkeypatch.setattr(analyze, "PLANNER_SUPPLIER", ["Modal", "Baseten"])
    calls = _fake_post(monkeypatch, [(None, None)])

    analyze.llm_create_once("sys", [], max_tokens=10, timeout=5)

    assert calls[0]["provider"]["only"] == ["Modal"], "phải thử nhà ĐẦU danh sách trước"


# ─── Failover khi nhà đầu lỗi ─────────────────────────────────────────────────

def test_timeout_failsover_to_next_supplier(monkeypatch, no_key):
    """Nhà đầu timeout → sang nhà thứ 2."""
    monkeypatch.setattr(analyze, "PLANNER_SUPPLIER", ["inference.net", "Modal", "Baseten"])
    calls = _fake_post(monkeypatch, [
        (requests.exceptions.ReadTimeout("read timed out"), None),
        (None, None),
    ])

    result = analyze.llm_create_once("sys", [], max_tokens=10, timeout=5)

    assert len(calls) == 2
    assert calls[0]["provider"]["only"] == ["inference.net"]
    assert calls[1]["provider"]["only"] == ["Modal"]
    assert result["provider"] == "Modal"


def test_connection_error_failsover_to_next_supplier(monkeypatch, no_key):
    monkeypatch.setattr(analyze, "PLANNER_SUPPLIER", ["A", "B"])
    calls = _fake_post(monkeypatch, [
        (requests.exceptions.ConnectionError("connection reset"), None),
        (None, None),
    ])

    result = analyze.llm_create_once("sys", [], max_tokens=10, timeout=5)

    assert result["provider"] == "B"
    assert len(calls) == 2


def test_two_suppliers_down_then_third_works(monkeypatch, no_key):
    monkeypatch.setattr(analyze, "PLANNER_SUPPLIER", ["A", "B", "C"])
    calls = _fake_post(monkeypatch, [
        (requests.exceptions.ReadTimeout("t1"), None),
        (requests.exceptions.ReadTimeout("t2"), None),
        (None, None),
    ])

    result = analyze.llm_create_once("sys", [], max_tokens=10, timeout=5)

    assert result["provider"] == "C"
    assert [c["provider"]["only"][0] for c in calls] == ["A", "B", "C"]


@pytest.mark.parametrize("status", [500, 502, 503, 504, 429])
def test_5xx_and_429_failsover(monkeypatch, no_key, status):
    monkeypatch.setattr(analyze, "PLANNER_SUPPLIER", ["A", "B"])
    calls = _fake_post(monkeypatch, [(None, status), (None, None)])

    result = analyze.llm_create_once("sys", [], max_tokens=10, timeout=5)

    assert result["provider"] == "B", f"status {status} phải chuyển nhà"
    assert len(calls) == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_4xx_does_not_failsover(monkeypatch, no_key, status):
    """Lỗi cấu hình/key/model: đổi nhà cũng lỗi y như vậy → raise ngay."""
    monkeypatch.setattr(analyze, "PLANNER_SUPPLIER", ["A", "B"])
    calls = _fake_post(monkeypatch, [(None, status)])

    with pytest.raises(RuntimeError):
        analyze.llm_create_once("sys", [], max_tokens=10, timeout=5)

    assert len(calls) == 1, "4xx không được đổi nhà"


def test_all_suppliers_down_raises(monkeypatch, no_key):
    monkeypatch.setattr(analyze, "PLANNER_SUPPLIER", ["A", "B"])
    calls = _fake_post(monkeypatch, [
        (requests.exceptions.ReadTimeout("t1"), None),
        (requests.exceptions.ReadTimeout("t2"), None),
    ])

    with pytest.raises(requests.exceptions.ReadTimeout):
        analyze.llm_create_once("sys", [], max_tokens=10, timeout=5)

    assert len(calls) == 2, "phải thử HẾT danh sách rồi mới bỏ cuộc"


def test_single_supplier_list_no_failsover(monkeypatch, no_key):
    monkeypatch.setattr(analyze, "PLANNER_SUPPLIER", ["only-one"])
    calls = _fake_post(monkeypatch, [(requests.exceptions.ReadTimeout("t"), None)])

    with pytest.raises(requests.exceptions.ReadTimeout):
        analyze.llm_create_once("sys", [], max_tokens=10, timeout=5)

    assert len(calls) == 1


# ─── Fallback khi PLANNER_SUPPLIER sai slug (lam chet ca luong manual) ───────

class _Err404(Exception):
    def __init__(self):
        super().__init__(
            'OpenRouter API error: 404 - {"error":{"message":"No allowed providers are '
            'available for the selected model. Providers serving x: a, b, inference-net, '
            'but your request\'s provider.only preference permits only: inference.net"}}'
        )


def test_no_provider_error_detected(monkeypatch):
    monkeypatch.setattr(analyze, "OPENROUTER_API_KEY", "k")
    assert analyze._is_no_provider_error(_Err404()) is True
    assert analyze._is_no_provider_error(RuntimeError("network down")) is False
    assert analyze._is_no_provider_error(Exception("401 unauthorized")) is False


def test_all_suppliers_mismatch_falls_back_unrestricted(monkeypatch):
    """Moi supplier deu 404 'no allowed providers' -> thu lai KHONG gioi han provider.

    Truoc fix nay, mot chu thai trong PLANNER_SUPPLIER lam 404 va chan TOAN BO phan tich
    (ke ca luong manual). User phai van nhan duoc ket qua.
    """
    monkeypatch.setattr(analyze, "OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(analyze, "PLANNER_SUPPLIER", ["wrong-slug-1", "wrong-slug-2"])
    calls = []

    def fake(system, messages, model, max_tokens, timeout=None,
             reasoning_effort=None, response_format=None, provider_slug=None):
        calls.append(provider_slug)
        if provider_slug:
            raise _Err404()
        return {"text": "NO_TRADE", "provider": None, "model": model}

    monkeypatch.setattr(analyze, "_openrouter_create_once", fake)

    result = analyze.llm_create_once("s", [], max_tokens=10, timeout=5)

    assert result["text"] == "NO_TRADE"
    # 2 lan gioi han (that bai) + 1 lan khong gioi han (thanh cong)
    assert calls == ["wrong-slug-1", "wrong-slug-2", None], calls


def test_regular_failure_still_raises(monkeypatch):
    """Loi thuong (mang/timeout) van raise nhu cu — fallback chi ap dung cho loi cau hinh."""
    import requests
    monkeypatch.setattr(analyze, "OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(analyze, "PLANNER_SUPPLIER", ["only-one"])
    monkeypatch.setattr(analyze, "_openrouter_create_once",
                        lambda *a, **k: (_ for _ in ()).throw(
                            requests.exceptions.ReadTimeout("t")))
    with pytest.raises(requests.exceptions.ReadTimeout):
        analyze.llm_create_once("s", [], max_tokens=10, timeout=5)
