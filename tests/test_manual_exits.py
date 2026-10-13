"""risk.manual_exits (2026-09-21, user request: "손절과 익절은 내가 할테니까
종목만 선정하게 하는건 어떨까"; briefly widened 2026-09-28 to also cover the
strategy's own stop_pct, then narrowed back 2026-10-13: "손절은 3%고 익절은
내가 할 수 있게 해줘"): every take-profit-style exit decision becomes a
rate-limited KakaoTalk notice instead of an actual sell, while every
stop-loss stays fully automatic -- the strategy's own "X% 손절" tier (see
_exit_should_defer_to_manual), its live-quote equivalent in
_live_protective_breach, the engine's own ATR-derived hard stop
(position.effective_stop(), capped by risk.max_stop_pct), and TIME EXIT
(day-trade flatten / planned overnight exit, a scheduled rather than a
price-based rule either way).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from market.calendar import KST
from portfolio import LONG, Portfolio, Position


def _paths(tmp_path):
    return dict(
        state_path=tmp_path / "state.json",
        trades_path=tmp_path / "trades.csv",
        daily_path=tmp_path / "daily_pnl.csv",
    )


def _engine(tmp_path, *, manual_exits: bool, notify_minutes: int = 10):
    from main import TradingEngine

    engine = TradingEngine.__new__(TradingEngine)
    engine.rt = SimpleNamespace(
        portfolio=Portfolio(**_paths(tmp_path), mode_label="DRY-RUN"),
        name_of=lambda code: "테스트종목",
        config={
            "risk": {
                "manual_exits": manual_exits,
                "manual_exits_notify_minutes": notify_minutes,
            }
        },
    )
    sent: list[str] = []
    engine._tg_notifier = SimpleNamespace(send=lambda msg: sent.append(msg))
    engine._sent = sent
    return engine


def _open_long(engine, code: str, entry: float) -> Position:
    engine.rt.portfolio.open_position(
        Position(
            symbol=code, side=LONG, qty=10, entry_price=entry,
            stop_price=entry * 0.98, stop_distance=entry * 0.02,
        )
    )
    return engine.rt.portfolio.get(code)


def test_manual_exits_reads_the_risk_config_flag(tmp_path):
    on = _engine(tmp_path, manual_exits=True)
    off = _engine(tmp_path, manual_exits=False)
    assert on._manual_exits() is True
    assert off._manual_exits() is False


def test_defer_exit_notifies_and_never_sells(tmp_path, monkeypatch):
    engine = _engine(tmp_path, manual_exits=True)
    _open_long(engine, "005930", 10_000.0)
    position = engine.rt.portfolio.get("005930")

    monkeypatch.setattr(
        engine, "_submit_exit", lambda *a, **k: pytest.fail("must not sell")
    )

    engine._defer_exit("005930", position, 10_200.0, "확정 익절", "005930 테스트")

    assert len(engine._sent) == 1
    assert "직접 매도 판단 필요" in engine._sent[0]
    assert "확정 익절" in engine._sent[0]

    updated = engine.rt.portfolio.get("005930")
    assert updated.last_manual_exit_notice != ""


def test_exit_should_defer_to_manual_never_defers_a_stop_loss_reason():
    """2026-10-13: a reason string containing "손절" is the strategy's own
    soft stop_pct tier -- stop-loss stays automatic regardless of
    manual_exits, so this must always come back False."""
    from main import _exit_should_defer_to_manual

    assert _exit_should_defer_to_manual("3% 손절", manual_exits=True) is False
    assert _exit_should_defer_to_manual("3% 손절", manual_exits=False) is False


def test_exit_should_defer_to_manual_defers_a_take_profit_reason_when_on():
    from main import _exit_should_defer_to_manual

    assert _exit_should_defer_to_manual("확정 익절", manual_exits=True) is True
    assert _exit_should_defer_to_manual("확정 익절", manual_exits=False) is False


def test_defer_exit_is_rate_limited(tmp_path):
    engine = _engine(tmp_path, manual_exits=True, notify_minutes=10)
    _open_long(engine, "005930", 10_000.0)
    position = engine.rt.portfolio.get("005930")

    engine._defer_exit("005930", position, 10_200.0, "확정 익절", "label")
    assert len(engine._sent) == 1

    # Same cycle's worth of re-checks (e.g. fast_exit_check_seconds=5) must
    # not spam a second notice.
    position = engine.rt.portfolio.get("005930")
    engine._defer_exit("005930", position, 10_210.0, "확정 익절", "label")
    assert len(engine._sent) == 1


def test_defer_exit_renotifies_after_the_cooldown(tmp_path):
    engine = _engine(tmp_path, manual_exits=True, notify_minutes=10)
    _open_long(engine, "005930", 10_000.0)
    position = engine.rt.portfolio.get("005930")

    stale = (datetime.now(KST) - timedelta(minutes=11)).isoformat()
    position.last_manual_exit_notice = stale
    engine.rt.portfolio.update_position(position)

    position = engine.rt.portfolio.get("005930")
    engine._defer_exit("005930", position, 10_200.0, "확정 익절", "label")
    assert len(engine._sent) == 1


def test_position_last_manual_exit_notice_defaults_empty_and_round_trips():
    p = Position(
        symbol="TEST", side=LONG, qty=1, entry_price=10_000.0,
        stop_price=9_800.0, stop_distance=200.0,
    )
    assert p.last_manual_exit_notice == ""
    p.last_manual_exit_notice = "2026-09-21T09:00:00+09:00"
    restored = Position.from_dict(p.to_dict())
    assert restored.last_manual_exit_notice == "2026-09-21T09:00:00+09:00"
