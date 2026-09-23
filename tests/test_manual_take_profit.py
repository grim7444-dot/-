"""risk.manual_take_profit (2026-09-21, user request: "손절과 익절은 내가
할테니까 종목만 선정하게 하는건 어떨까" -> "완전 수동 -- 손절선 하나만
자동"): every take-profit-style exit becomes a rate-limited Telegram notice
instead of an actual sell, while the hard stop and any strategy exit whose
reason contains "손절" keep selling exactly as before.
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


def _engine(tmp_path, *, manual_take_profit: bool, notify_minutes: int = 10):
    from main import TradingEngine

    engine = TradingEngine.__new__(TradingEngine)
    engine.rt = SimpleNamespace(
        portfolio=Portfolio(**_paths(tmp_path), mode_label="DRY-RUN"),
        name_of=lambda code: "테스트종목",
        config={
            "risk": {
                "manual_take_profit": manual_take_profit,
                "manual_take_profit_notify_minutes": notify_minutes,
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


def test_manual_take_profit_reads_the_risk_config_flag(tmp_path):
    on = _engine(tmp_path, manual_take_profit=True)
    off = _engine(tmp_path, manual_take_profit=False)
    assert on._manual_take_profit() is True
    assert off._manual_take_profit() is False


def test_defer_profit_exit_notifies_and_never_sells(tmp_path, monkeypatch):
    engine = _engine(tmp_path, manual_take_profit=True)
    _open_long(engine, "005930", 10_000.0)
    position = engine.rt.portfolio.get("005930")

    monkeypatch.setattr(
        engine, "_submit_exit", lambda *a, **k: pytest.fail("must not sell")
    )

    engine._defer_profit_exit("005930", position, 10_200.0, "확정 익절", "005930 테스트")

    assert len(engine._sent) == 1
    assert "직접 매도 판단 필요" in engine._sent[0]
    assert "확정 익절" in engine._sent[0]

    updated = engine.rt.portfolio.get("005930")
    assert updated.last_manual_exit_notice != ""


def test_defer_profit_exit_is_rate_limited(tmp_path):
    engine = _engine(tmp_path, manual_take_profit=True, notify_minutes=10)
    _open_long(engine, "005930", 10_000.0)
    position = engine.rt.portfolio.get("005930")

    engine._defer_profit_exit("005930", position, 10_200.0, "확정 익절", "label")
    assert len(engine._sent) == 1

    # Same cycle's worth of re-checks (e.g. fast_exit_check_seconds=5) must
    # not spam a second notice.
    position = engine.rt.portfolio.get("005930")
    engine._defer_profit_exit("005930", position, 10_210.0, "확정 익절", "label")
    assert len(engine._sent) == 1


def test_defer_profit_exit_renotifies_after_the_cooldown(tmp_path):
    engine = _engine(tmp_path, manual_take_profit=True, notify_minutes=10)
    _open_long(engine, "005930", 10_000.0)
    position = engine.rt.portfolio.get("005930")

    stale = (datetime.now(KST) - timedelta(minutes=11)).isoformat()
    position.last_manual_exit_notice = stale
    engine.rt.portfolio.update_position(position)

    position = engine.rt.portfolio.get("005930")
    engine._defer_profit_exit("005930", position, 10_200.0, "확정 익절", "label")
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
