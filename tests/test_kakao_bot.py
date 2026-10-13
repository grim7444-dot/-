"""KakaoNotifier (2026-09-28, replaces telegram_bot.py entirely -- user
request: "카톡으로 해줄 수 있나"). Covers the two things unique to Kakao's
"나에게 보내기" API versus the old static-token Telegram bot: automatic
access_token refresh from a refresh_token, and persisting a rotated
refresh_token back into .env so a later restart doesn't need the browser
login redone.
"""

from __future__ import annotations

import json
import sys
import types

import pytest


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeRequests:
    """Records every POST and returns queued responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def post(self, url, headers=None, data=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "data": data, "json": json})
        if not self._responses:
            raise AssertionError("no more fake responses queued")
        return self._responses.pop(0)


def _install_fake_requests(monkeypatch, responses):
    fake = _FakeRequests(responses)
    monkeypatch.setitem(sys.modules, "requests", fake)
    return fake


def _notifier(tmp_path, **kwargs):
    from kakao_bot import KakaoNotifier

    env_path = tmp_path / ".env"
    env_path.write_text(kwargs.pop("env_text", ""), encoding="utf-8")
    return KakaoNotifier(
        rest_api_key=kwargs.pop("rest_api_key", "REST_KEY"),
        refresh_token=kwargs.pop("refresh_token", "REFRESH_TOKEN"),
        dry_run=kwargs.pop("dry_run", False),
        env_path=env_path,
    )


def test_dry_run_never_calls_requests(tmp_path, monkeypatch):
    notifier = _notifier(tmp_path, dry_run=True)
    monkeypatch.delitem(sys.modules, "requests", raising=False)

    result = notifier.send_with_result("hello")

    assert result == {"sent": False, "reason": "dry-run", "preview": "hello"}


def test_missing_credentials_previews_without_calling_requests(tmp_path, monkeypatch):
    notifier = _notifier(tmp_path, dry_run=False, rest_api_key="", refresh_token="")
    monkeypatch.delitem(sys.modules, "requests", raising=False)

    result = notifier.send_with_result("hello")

    assert result["sent"] is False
    assert result["reason"] == "no Kakao credentials configured"


def test_successful_send_refreshes_token_then_posts_memo(tmp_path, monkeypatch):
    notifier = _notifier(tmp_path)
    fake = _install_fake_requests(
        monkeypatch,
        [
            _FakeResponse(200, {"access_token": "ACCESS1", "expires_in": 21600}),
            _FakeResponse(200, {"result_code": 0}),
        ],
    )

    result = notifier.send_with_result("*진입* `005930`")

    assert result == {"sent": True, "reason": "", "preview": "진입 005930"}
    assert len(fake.calls) == 2
    assert fake.calls[0]["url"].startswith("https://kauth.kakao.com")
    assert fake.calls[0]["data"]["refresh_token"] == "REFRESH_TOKEN"
    memo_call = fake.calls[1]
    assert memo_call["url"].startswith("https://kapi.kakao.com")
    assert memo_call["headers"]["Authorization"] == "Bearer ACCESS1"
    template = json.loads(memo_call["data"]["template_object"])
    assert template["text"] == "진입 005930"


def test_cached_access_token_is_reused_within_its_lifetime(tmp_path, monkeypatch):
    notifier = _notifier(tmp_path)
    fake = _install_fake_requests(
        monkeypatch,
        [
            _FakeResponse(200, {"access_token": "ACCESS1", "expires_in": 21600}),
            _FakeResponse(200, {"result_code": 0}),
            _FakeResponse(200, {"result_code": 0}),
        ],
    )

    notifier.send_with_result("first")
    notifier.send_with_result("second")

    # Only one token-refresh call across both sends -- the second send
    # reused the cached, still-valid access_token.
    refresh_calls = [c for c in fake.calls if "kauth" in c["url"]]
    assert len(refresh_calls) == 1


def test_401_triggers_one_retry_with_a_fresh_token(tmp_path, monkeypatch):
    notifier = _notifier(tmp_path)
    fake = _install_fake_requests(
        monkeypatch,
        [
            _FakeResponse(200, {"access_token": "STALE", "expires_in": 21600}),
            _FakeResponse(401, {}),
            _FakeResponse(200, {"access_token": "FRESH", "expires_in": 21600}),
            _FakeResponse(200, {"result_code": 0}),
        ],
    )

    result = notifier.send_with_result("hello")

    assert result["sent"] is True
    memo_calls = [c for c in fake.calls if "kapi" in c["url"]]
    assert memo_calls[-1]["headers"]["Authorization"] == "Bearer FRESH"


def test_token_refresh_failure_is_reported_without_crashing(tmp_path, monkeypatch):
    notifier = _notifier(tmp_path)
    _install_fake_requests(
        monkeypatch,
        [_FakeResponse(400, {"error_description": "invalid_grant"})],
    )

    result = notifier.send_with_result("hello")

    assert result["sent"] is False
    assert result["reason"] == "token refresh failed"


def test_rotated_refresh_token_is_persisted_to_env_file(tmp_path, monkeypatch):
    notifier = _notifier(
        tmp_path,
        env_text="KIWOOM_PAPER=true\nKAKAO_REFRESH_TOKEN=REFRESH_TOKEN\nDART_API_KEY=x\n",
    )
    _install_fake_requests(
        monkeypatch,
        [
            _FakeResponse(
                200,
                {"access_token": "ACCESS1", "expires_in": 21600, "refresh_token": "ROTATED"},
            ),
            _FakeResponse(200, {"result_code": 0}),
        ],
    )

    notifier.send_with_result("hello")

    assert notifier._refresh_token == "ROTATED"
    env_text = notifier._env_path.read_text(encoding="utf-8")
    assert "KAKAO_REFRESH_TOKEN=ROTATED" in env_text
    assert "KIWOOM_PAPER=true" in env_text
    assert "DART_API_KEY=x" in env_text
    assert env_text.count("KAKAO_REFRESH_TOKEN=") == 1


def test_rotated_refresh_token_is_appended_if_the_key_was_missing(tmp_path, monkeypatch):
    notifier = _notifier(tmp_path, env_text="KIWOOM_PAPER=true\n")
    _install_fake_requests(
        monkeypatch,
        [
            _FakeResponse(
                200,
                {"access_token": "ACCESS1", "expires_in": 21600, "refresh_token": "ROTATED"},
            ),
            _FakeResponse(200, {"result_code": 0}),
        ],
    )

    notifier.send_with_result("hello")

    env_text = notifier._env_path.read_text(encoding="utf-8")
    assert "KAKAO_REFRESH_TOKEN=ROTATED" in env_text


def test_build_kakao_reads_config_dry_run_and_credentials(tmp_path):
    from kakao_bot import build_kakao
    from settings import Credentials, Secret

    credentials = Credentials(
        app_key=Secret(""),
        secret_key=Secret(""),
        account_no=Secret(""),
        kakao_rest_api_key=Secret("REST"),
        kakao_refresh_token=Secret("REFRESH"),
        dart_api_key=Secret(""),
        loaded_for="PAPER",
    )
    notifier = build_kakao(credentials, {"kakao": {"dry_run": False}})

    assert notifier.dry_run is False
    assert notifier._rest_api_key == "REST"
    assert notifier._refresh_token == "REFRESH"


def test_build_kakao_defaults_to_dry_run_with_no_config():
    from kakao_bot import build_kakao

    notifier = build_kakao(None, {})

    assert notifier.dry_run is True
    assert notifier._rest_api_key == ""
    assert notifier._refresh_token == ""


def test_alert_helpers_format_and_forward_to_send(tmp_path, monkeypatch):
    notifier = _notifier(tmp_path, dry_run=True)
    sent: list[str] = []
    monkeypatch.setattr(notifier, "send", lambda text: sent.append(text))

    notifier.alert_entry("005930", "삼성전자", "LONG", 10, 70_000.0, 68_000.0, 1_000_000.0)
    notifier.alert_exit("005930", "삼성전자", 10, 71_000.0, 70_000.0, "익절")
    notifier.alert_stop_hit("005930", "삼성전자", 67_900.0, 68_000.0)
    notifier.alert_kill_switch(900_000.0, 1_000_000.0)

    assert len(sent) == 4
    assert "005930" in sent[0] and "진입" in sent[0]
    assert "청산" in sent[1]
    assert "손절 발동" in sent[2]
    assert "킬스위치" in sent[3]
