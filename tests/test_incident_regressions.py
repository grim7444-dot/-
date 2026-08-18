"""Regressions for the first live run.

On 2026-08-18 the first live session tripped the drawdown kill switch within
seconds of starting and tried to flatten the whole account. Three separate
defects lined up to make that possible, and each gets a test here:

1. ``state.json`` carried the dry-run broker's 10,000,000 KRW peak equity into
   a live account worth 573,390 KRW. The comparison is meaningless across
   modes, but the kill switch read it as a 94% drawdown.
2. The kill switch fired market orders at 18:15 KST, hours after the close.
   There was no order book to reach.
3. Those orders came back with an empty order number, and the bot logged them
   as ``submitted`` anyway -- so the log claimed two positions were closed when
   nothing had been sold.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from market.calendar import KST
from portfolio import Portfolio
from risk.manager import RiskManager


# ---------------------------------------------------------------------------
# 1. equity history does not survive a mode change
# ---------------------------------------------------------------------------


def _paths(tmp_path):
    return dict(
        state_path=tmp_path / "state.json",
        trades_path=tmp_path / "trades.csv",
        daily_path=tmp_path / "daily_pnl.csv",
    )


def test_mode_change_discards_equity_history(tmp_path):
    dry = Portfolio(**_paths(tmp_path), mode_label="DRY-RUN")
    dry.mark_equity(10_000_000.0)
    dry.save()
    assert dry.state.peak_equity == pytest.approx(10_000_000.0)

    live = Portfolio(**_paths(tmp_path), mode_label="LIVE")
    assert live.state.peak_equity == 0.0
    assert live.state.last_equity == 0.0
    assert live.state.day_start_equity == 0.0
    assert live.state.day_start_date == ""


def test_same_mode_keeps_equity_history(tmp_path):
    """The reset must be narrow: a restart in the same mode loses nothing."""
    first = Portfolio(**_paths(tmp_path), mode_label="LIVE")
    first.mark_equity(573_390.0)
    first.save()

    second = Portfolio(**_paths(tmp_path), mode_label="LIVE")
    assert second.state.peak_equity == pytest.approx(573_390.0)


def test_mode_change_does_not_clear_the_stopped_flag(tmp_path):
    """Rule 6 outranks the reset: STOPPED still needs an explicit resume."""
    dry = Portfolio(**_paths(tmp_path), mode_label="DRY-RUN")
    dry.stop("drawdown kill switch: test")

    live = Portfolio(**_paths(tmp_path), mode_label="LIVE")
    assert live.stopped is True


def test_first_live_equity_does_not_read_as_a_drawdown(tmp_path, config):
    """The incident, end to end: dry run then live, no kill switch."""
    dry = Portfolio(**_paths(tmp_path), mode_label="DRY-RUN")
    dry.mark_equity(10_000_000.0)
    dry.save()

    live = Portfolio(**_paths(tmp_path), mode_label="LIVE")
    status = RiskManager(config, live).check_drawdown(573_390.0)
    assert status.breached is False


# ---------------------------------------------------------------------------
# 2. the kill switch does not fire market orders into a closed market
# ---------------------------------------------------------------------------


class _FixedClockCalendar:
    """KrxCalendar stand-in pinned to one moment."""

    def __init__(self, moment: datetime) -> None:
        from market.calendar import KrxCalendar

        self._inner = KrxCalendar()
        self._moment = moment

    def can_place_market_order(self, moment: datetime | None = None):
        return self._inner.can_place_market_order(self._moment)


CLOSED = datetime(2026, 8, 18, 18, 15, tzinfo=KST)      # the incident's clock
OPEN = datetime(2026, 8, 18, 11, 0, tzinfo=KST)         # a Tuesday, mid-session


def _stopped_manager(portfolio, config, broker, calendar):
    manager = RiskManager(config, portfolio, calendar=calendar)
    portfolio.mark_equity(10_000_000.0)
    status = manager.check_drawdown(8_800_000.0)
    assert status.breached is True
    manager.trip_kill_switch(status, broker)
    return manager


def test_kill_switch_does_not_flatten_outside_trading_hours(
    portfolio, recording_broker, config
):
    _stopped_manager(portfolio, config, recording_broker, _FixedClockCalendar(CLOSED))

    # Orders are still cancelled -- that is safe at any hour -- but no sell is
    # sent, because it would not reach a book.
    assert recording_broker.cancelled == 1
    assert recording_broker.closed == 0
    # And the bot stays STOPPED so it cannot re-enter while unprotected.
    assert portfolio.stopped is True


def test_kill_switch_still_flattens_during_continuous_trading(
    portfolio, recording_broker, config
):
    _stopped_manager(portfolio, config, recording_broker, _FixedClockCalendar(OPEN))

    assert recording_broker.cancelled == 1
    assert recording_broker.closed == 1
    assert portfolio.stopped is True


def test_kill_switch_without_a_calendar_still_flattens(
    portfolio, recording_broker, config
):
    """No calendar means no way to know; flattening is the safer default."""
    _stopped_manager(portfolio, config, recording_broker, None)

    assert recording_broker.closed == 1


# ---------------------------------------------------------------------------
# 3. an order with no order number is a rejection, not a submission
# ---------------------------------------------------------------------------


def _broker_returning(payload):
    """A KiwoomBroker with its transport replaced -- no network, no keys."""
    from broker import KiwoomBroker

    broker = object.__new__(KiwoomBroker)
    broker.allowed_codes = frozenset({"002990", "073240"})
    broker._call = lambda *a, **k: payload
    return broker


def test_empty_order_number_raises_instead_of_reporting_a_submission():
    from broker import BrokerError

    broker = _broker_returning({"ord_no": "", "return_msg": "장운영시간이 아닙니다"})
    with pytest.raises(BrokerError, match="not accepted"):
        broker.submit_order("002990", "SHORT", 21)


def test_nonzero_return_code_raises_even_with_an_order_number():
    from broker import BrokerError

    broker = _broker_returning({"ord_no": "12345", "return_code": "3", "return_msg": "거부"})
    with pytest.raises(BrokerError, match=r"return_code=3"):
        broker.submit_order("002990", "SHORT", 21)


def test_accepted_order_is_reported_as_submitted():
    broker = _broker_returning({"ord_no": "0000123", "return_code": "0"})
    result = broker.submit_order("002990", "SHORT", 21)
    assert result.submitted is True
    assert result.order_id == "0000123"


def test_a_rejected_flatten_is_not_counted_as_closed(monkeypatch):
    """close_all_positions must report what was accepted, not what it tried."""
    from broker import Holding

    broker = _broker_returning({"ord_no": "", "return_msg": "rejected"})
    monkeypatch.setattr(
        type(broker),
        "get_holdings",
        lambda self: {"002990": Holding(code="002990", qty=21.0)},
    )
    assert broker.close_all_positions() == 0
