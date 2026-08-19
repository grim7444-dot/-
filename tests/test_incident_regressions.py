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

from datetime import datetime, timedelta

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


# ---------------------------------------------------------------------------
# 4. a simulated broker never shares a mode label with a real one
# ---------------------------------------------------------------------------


def test_dry_run_broker_gets_its_own_mode_label(workdir, monkeypatch):
    """`live --dry-run` must not stamp its 10,000,000 KRW onto LIVE."""
    import main

    captured: dict[str, object] = {}
    real_build = main.build_runtime

    def spy(args, cli_live, force_dry_run=False):
        rt = real_build(args, cli_live, force_dry_run)
        captured["portfolio"] = rt.portfolio
        return rt

    monkeypatch.setattr(main, "build_runtime", spy)
    assert main.main(["paper", "--dry-run"]) == 0

    portfolio = captured["portfolio"]
    assert portfolio.mode_label.endswith("-SIM")


def test_watch_flag_switches_dry_run_from_one_cycle_to_the_loop(workdir, monkeypatch):
    """--dry-run alone is a smoke test; --dry-run --watch observes all day."""
    import main

    calls: list[str] = []
    monkeypatch.setattr(
        main.TradingEngine, "run_once", lambda self: calls.append("once")
    )
    monkeypatch.setattr(
        main.TradingEngine, "run_forever", lambda self: calls.append("forever")
    )

    assert main.main(["paper", "--dry-run"]) == 0
    assert calls == ["once"]

    calls.clear()
    assert main.main(["paper", "--dry-run", "--watch"]) == 0
    assert calls == ["forever"]

    calls.clear()
    # --once always wins: it is the escape hatch that guarantees termination.
    assert main.main(["paper", "--dry-run", "--watch", "--once"]) == 0
    assert calls == ["once"]


# ---------------------------------------------------------------------------
# 5. the account snapshot cannot understate what the account holds
# ---------------------------------------------------------------------------


def _account_broker(payload):
    from broker import KiwoomBroker

    broker = object.__new__(KiwoomBroker)
    broker._call = lambda *a, **k: payload
    return broker


def test_equity_is_never_less_than_holdings_plus_cash():
    """tot_evlt_amt is the stock valuation alone; cash has to be added back."""
    broker = _account_broker({"tot_evlt_amt": "273,960", "entr": "150,000"})
    account = broker.get_account()
    assert account.cash == pytest.approx(150_000)
    assert account.equity == pytest.approx(423_960)


def test_a_missing_cash_field_is_not_read_as_zero_cash():
    """Absent key vs genuine zero: the first candidate that carries a value wins."""
    broker = _account_broker(
        {"tot_evlt_amt": "273,960", "entr": "0", "d2_entra": "150,000"}
    )
    assert broker.get_account().cash == pytest.approx(150_000)


def test_a_genuinely_empty_account_still_reports_zero():
    broker = _account_broker({"tot_evlt_amt": "273,960", "entr": "0"})
    account = broker.get_account()
    assert account.cash == 0.0
    assert account.equity == pytest.approx(273_960)


def test_balance_fields_are_kept_for_diagnosis_without_nested_payloads():
    broker = _account_broker(
        {"tot_evlt_amt": "273,960", "entr": "0", "acnt_evlt_remn_indv_tot": [{"stk_cd": "073240"}]}
    )
    broker.get_account()
    assert broker.last_balance_fields == {"tot_evlt_amt": "273,960", "entr": "0"}


# ---------------------------------------------------------------------------
# 6. a cp949 console cannot abort a command
# ---------------------------------------------------------------------------


def test_sources_hold_no_characters_a_korean_console_cannot_encode():
    """The check died on its own title line: cp949 has no em dash."""
    import pathlib

    offenders: list[str] = []
    root = pathlib.Path(__file__).resolve().parents[1]
    for path in sorted(root.rglob("*.py")):
        if any(part in {".venv", "__pycache__"} for part in path.parts):
            continue
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for char in line:
                if ord(char) < 128:
                    continue
                try:
                    char.encode("cp949")
                except UnicodeEncodeError:
                    offenders.append(
                        f"{path.relative_to(root)}:{lineno} {char!r} (U+{ord(char):04X})"
                    )
    assert not offenders, "characters cp949 cannot encode:\n" + "\n".join(offenders)


def test_make_console_tolerant_survives_a_replaced_stream(monkeypatch):
    """pytest swaps stdout for an object without reconfigure(); do not crash."""
    import io

    from settings import make_console_tolerant

    monkeypatch.setattr("sys.stdout", io.StringIO())
    monkeypatch.setattr("sys.stderr", io.StringIO())
    make_console_tolerant()


def test_make_console_tolerant_sets_a_forgiving_error_handler(monkeypatch, tmp_path):
    from settings import make_console_tolerant

    target = tmp_path / "out.txt"
    with open(target, "w", encoding="cp949") as handle:
        monkeypatch.setattr("sys.stdout", handle)
        monkeypatch.setattr("sys.stderr", handle)
        make_console_tolerant()
        # Built from its code point so this file stays cp949-clean itself.
        print("em dash: " + chr(0x2014) + " done")  # unguarded, this raises

    assert "done" in target.read_text(encoding="cp949")


# ---------------------------------------------------------------------------
# 7. the deposit is derived when the balance TR does not report one
# ---------------------------------------------------------------------------

#: The real response from a live account on 2026-08-19, trimmed to the scalar
#: fields. Note what is absent: there is no deposit field of any name.
LIVE_BALANCE = {
    "prsm_dpst_aset_amt": "000000000550310",
    "tot_evlt_amt": "000000000273960",
    "tot_pur_amt": "000000000299160",
    "tot_evlt_pl": "-00000000025826",
    "tot_prft_rt": "-8.63",
    "tot_loan_amt": "000000000000000",
    "tot_crd_loan_amt": "000000000000000",
    "tot_crd_ls_amt": "000000000000000",
    "return_code": "0",
}


def test_deposit_is_derived_from_total_assets_minus_holdings():
    account = _account_broker(dict(LIVE_BALANCE)).get_account()
    assert account.equity == pytest.approx(550_310)
    assert account.cash == pytest.approx(276_350)
    assert account.buying_power == pytest.approx(276_350)


def test_borrowings_are_not_counted_as_spendable_cash():
    payload = dict(LIVE_BALANCE, tot_loan_amt="000000000100000")
    assert _account_broker(payload).get_account().cash == pytest.approx(176_350)


def test_an_explicit_deposit_field_wins_over_the_derivation():
    payload = dict(LIVE_BALANCE, entr="000000000276350")
    broker = _account_broker(payload)
    assert broker.get_account().cash == pytest.approx(276_350)


def test_a_fully_invested_account_derives_no_cash():
    payload = dict(
        LIVE_BALANCE,
        prsm_dpst_aset_amt="000000000273960",
        tot_evlt_amt="000000000273960",
    )
    assert _account_broker(payload).get_account().cash == 0.0


# ---------------------------------------------------------------------------
# 8. stale bars are not treated as the present
# ---------------------------------------------------------------------------


def _daily_frame(last_day, rows=5):
    import pandas as pd

    index = pd.DatetimeIndex(
        [last_day - timedelta(days=i) for i in reversed(range(rows))], name="timestamp"
    )
    return pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )


def _market_data(tmp_path):
    from data import MarketData
    from market.calendar import KrxCalendar

    return MarketData(calendar=KrxCalendar(), cache_dir=tmp_path / "cache")


def test_daily_bars_ending_yesterday_are_current(tmp_path):
    md = _market_data(tmp_path)
    now = datetime(2026, 8, 19, 11, 0, tzinfo=KST)          # a Wednesday
    frame = _daily_frame(datetime(2026, 8, 18))             # Tuesday's close
    assert md.is_current(frame, "1Day", now) is True


def test_daily_bars_three_sessions_old_are_not_current(tmp_path):
    md = _market_data(tmp_path)
    now = datetime(2026, 8, 19, 11, 0, tzinfo=KST)
    # This is what a real run produced: a cache that stopped on the 14th while
    # the strategy went on deciding as though it were the 19th.
    frame = _daily_frame(datetime(2026, 8, 14))
    assert md.is_current(frame, "1Day", now) is False


def test_a_weekend_does_not_make_friday_stale(tmp_path):
    md = _market_data(tmp_path)
    monday = datetime(2026, 8, 17, 10, 0, tzinfo=KST)
    friday = _daily_frame(datetime(2026, 8, 14))
    assert md.is_current(friday, "1Day", monday) is True


def test_intraday_bars_get_a_two_interval_grace(tmp_path):
    import pandas as pd

    md = _market_data(tmp_path)
    now = datetime(2026, 8, 19, 11, 30, tzinfo=KST)
    index = pd.DatetimeIndex(
        [datetime(2026, 8, 19, h, tzinfo=KST) for h in (9, 10, 11)], name="timestamp"
    )
    fresh = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=index
    )
    assert md.is_current(fresh, "60Min", now) is True

    stale = fresh.copy()
    stale.index = pd.DatetimeIndex(
        [datetime(2026, 8, 18, h, tzinfo=KST) for h in (13, 14, 15)], name="timestamp"
    )
    assert md.is_current(stale, "60Min", now) is False


def test_an_empty_frame_is_never_current(tmp_path):
    from data import empty_frame

    assert _market_data(tmp_path).is_current(empty_frame(), "1Day") is False
