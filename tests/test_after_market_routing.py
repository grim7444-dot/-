"""Orders placed during the 16:00-20:00 NXT/KRX after-market (live from
2026-09-14, user request: "16시부터 20시까지 매매할 수 있게 만들어줘") must
route to NXT -- KRX's own matching engine is closed after 15:30, so an order
still tagged dmst_stex_tp="KRX" would simply be refused there. Everywhere
else, routing stays "KRX", unchanged.
"""

from __future__ import annotations

from datetime import datetime

import pytest

import broker as broker_module
from broker import KiwoomBroker
from market.calendar import KST
from settings import Credentials, Secret, resolve_mode


class _FixedDatetime(datetime):
    """Stands in for the module-level ``datetime`` broker.py imports, so
    ``datetime.now(KST)`` inside submit_order/cancel_all_orders returns a
    moment the test controls instead of the real wall clock."""

    _fixed: datetime

    @classmethod
    def now(cls, tz=None):
        return cls._fixed if tz is None else cls._fixed.astimezone(tz)


def _broker(config):
    decision = resolve_mode({}, cli_live=False)
    credentials = Credentials(
        app_key=Secret("K" * 43),
        secret_key=Secret("S" * 43),
        account_no=Secret("12345678"),
        telegram_token=Secret(""),
        telegram_chat_id=Secret(""),
        dart_api_key=Secret(""),
        loaded_for="PAPER",
    )
    return KiwoomBroker(decision, credentials, config, allowed_codes=list(config["universe"]))


def _freeze(monkeypatch, hour: int, minute: int = 0):
    fixed = _FixedDatetime(2026, 9, 14, hour, minute, tzinfo=KST)
    _FixedDatetime._fixed = fixed
    monkeypatch.setattr(broker_module, "datetime", _FixedDatetime)


def test_submit_order_routes_to_nxt_during_the_after_market(config, monkeypatch):
    b = _broker(config)
    _freeze(monkeypatch, 18, 0)  # 18:00 -- inside 16:00-20:00
    captured = {}

    def fake_call(category, api_id_key, body, label):
        captured["body"] = body
        return {"ord_no": "123", "return_code": "0", "return_msg": ""}

    monkeypatch.setattr(b, "_call", fake_call)
    code = next(iter(config["universe"]))
    b.submit_order(code, "LONG", 1, price=10_000)
    assert captured["body"]["dmst_stex_tp"] == "NXT"


def test_submit_order_routes_to_krx_during_the_regular_session(config, monkeypatch):
    b = _broker(config)
    _freeze(monkeypatch, 10, 0)  # 10:00 -- regular continuous session
    captured = {}

    def fake_call(category, api_id_key, body, label):
        captured["body"] = body
        return {"ord_no": "123", "return_code": "0", "return_msg": ""}

    monkeypatch.setattr(b, "_call", fake_call)
    code = next(iter(config["universe"]))
    b.submit_order(code, "LONG", 1, price=10_000)
    assert captured["body"]["dmst_stex_tp"] == "KRX"


@pytest.mark.parametrize("hour,minute,expected", [(15, 59, "KRX"), (16, 0, "NXT"), (19, 59, "NXT"), (20, 0, "KRX")])
def test_submit_order_routing_follows_the_after_market_boundary(config, monkeypatch, hour, minute, expected):
    b = _broker(config)
    _freeze(monkeypatch, hour, minute)
    captured = {}

    def fake_call(category, api_id_key, body, label):
        captured["body"] = body
        return {"ord_no": "123", "return_code": "0", "return_msg": ""}

    monkeypatch.setattr(b, "_call", fake_call)
    code = next(iter(config["universe"]))
    b.submit_order(code, "LONG", 1, price=10_000)
    assert captured["body"]["dmst_stex_tp"] == expected


def test_cancel_all_orders_also_routes_to_nxt_during_the_after_market(config, monkeypatch):
    b = _broker(config)
    _freeze(monkeypatch, 18, 0)
    code = next(iter(config["universe"]))
    monkeypatch.setattr(b, "open_order_codes", lambda: [code])
    captured = {}

    def fake_call(category, api_id_key, body, label):
        captured["body"] = body
        return {}

    monkeypatch.setattr(b, "_call", fake_call)
    b.cancel_all_orders()
    assert captured["body"]["dmst_stex_tp"] == "NXT"
