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

import logging
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


def test_daily_bars_ending_yesterday_are_stale_during_the_session(tmp_path):
    """The loop prices stops off the last close, so it has to be today's."""
    md = _market_data(tmp_path)
    trading = datetime(2026, 8, 19, 11, 0, tzinfo=KST)      # Wednesday, open
    assert md.is_current(_daily_frame(datetime(2026, 8, 18)), "1Day", trading) is False
    assert md.is_current(_daily_frame(datetime(2026, 8, 19)), "1Day", trading) is True


def test_daily_bars_ending_yesterday_are_current_before_the_open(tmp_path):
    """Today's bar does not exist yet at 08:40; demanding it blocks the run."""
    md = _market_data(tmp_path)
    pre_open = datetime(2026, 8, 19, 8, 40, tzinfo=KST)
    assert md.is_current(_daily_frame(datetime(2026, 8, 18)), "1Day", pre_open) is True


def test_daily_bars_three_sessions_old_are_not_current(tmp_path):
    md = _market_data(tmp_path)
    now = datetime(2026, 8, 19, 11, 0, tzinfo=KST)
    # This is what a real run produced: a cache that stopped on the 14th while
    # the strategy went on deciding as though it were the 19th.
    frame = _daily_frame(datetime(2026, 8, 14))
    assert md.is_current(frame, "1Day", now) is False


def test_a_weekend_does_not_make_friday_stale(tmp_path):
    """Saturday and Sunday are not missed sessions."""
    md = _market_data(tmp_path)
    friday = _daily_frame(datetime(2026, 8, 14))
    saturday = datetime(2026, 8, 15, 12, 0, tzinfo=KST)
    monday_pre_open = datetime(2026, 8, 17, 8, 0, tzinfo=KST)
    assert md.is_current(friday, "1Day", saturday) is True
    assert md.is_current(friday, "1Day", monday_pre_open) is True


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


# ---------------------------------------------------------------------------
# 9. names and markets resolve from the broker when KRX will not answer
# ---------------------------------------------------------------------------


class _StubProvider:
    def __init__(self, info):
        self._info = info

    def get_chart(self, code, timeframe, start, end):
        return None

    def get_stock_info(self, code):
        return self._info


def _market_data_with(provider, tmp_path):
    from data import MarketData
    from market.calendar import KrxCalendar

    return MarketData(
        calendar=KrxCalendar(), cache_dir=tmp_path / "cache", intraday_provider=provider
    )


def test_market_tag_falls_back_to_the_broker(tmp_path, monkeypatch):
    """A wrong market tag mis-prices the transaction tax, so it must be checked."""
    import data as data_module
    from broker import StockInfo
    from market.rules import KOSDAQ

    # pykrx unreachable, exactly as KRX has been answering.
    monkeypatch.setattr(
        data_module, "_import_pykrx_stock", lambda: (_ for _ in ()).throw(RuntimeError("KRX down"))
    )
    provider = _StubProvider(
        StockInfo(code="460930", name="현대힘스", market="10", market_raw="10")
    )
    md = _market_data_with(provider, tmp_path)

    assert md.get_name("460930") == "현대힘스"
    assert md.get_market("460930") == KOSDAQ


def test_an_unrecognised_market_label_resolves_to_nothing(tmp_path, monkeypatch):
    """Guessing here would mis-price tax; None just leaves the config alone."""
    import data as data_module
    from broker import StockInfo

    monkeypatch.setattr(
        data_module, "_import_pykrx_stock", lambda: (_ for _ in ()).throw(RuntimeError("KRX down"))
    )
    provider = _StubProvider(
        StockInfo(code="460930", name="", market="KONEX?", market_raw="KONEX?")
    )
    md = _market_data_with(provider, tmp_path)

    assert md.get_market("460930") is None
    assert md.get_name("460930") is None


def test_no_broker_means_no_resolution_rather_than_an_error(tmp_path, monkeypatch):
    import data as data_module

    monkeypatch.setattr(
        data_module, "_import_pykrx_stock", lambda: (_ for _ in ()).throw(RuntimeError("KRX down"))
    )
    md = _market_data_with(None, tmp_path)
    assert md.get_name("460930") is None
    assert md.get_market("460930") is None


def test_a_missing_market_field_does_not_confirm_itself(tmp_path, monkeypatch):
    """StockInfo.market defaults to KOSPI, so verification must read market_raw.

    Reading `market` flagged all four KOSDAQ stocks as "listing says KOSPI"
    when Kiwoom had in fact said nothing at all.
    """
    import data as data_module
    from broker import StockInfo

    monkeypatch.setattr(
        data_module, "_import_pykrx_stock", lambda: (_ for _ in ()).throw(RuntimeError("KRX down"))
    )
    info = StockInfo(code="460930", name="현대힘스")   # no market field returned
    assert info.market == "KOSPI"                      # the defaulted value
    assert info.market_raw == ""                       # what was actually said

    md = _market_data_with(_StubProvider(info), tmp_path)
    assert md.get_market("460930") is None


# ---------------------------------------------------------------------------
# 10. a full cycle actually runs
# ---------------------------------------------------------------------------


class _ErrorCollector(logging.Handler):
    """Collects ERROR records from the bot's own logger.

    Two things defeat the obvious approaches. `setup_logging` sets
    propagate = False on the `bot` logger, so pytest's caplog -- which hangs
    its handler off root -- sees nothing; and it calls handlers.clear(), so a
    handler attached before the run is gone by the time anything is logged.
    Earlier versions of this test using each approach passed with the bug
    still present.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def attach_after_setup(self, module, monkeypatch) -> None:
        """Re-attach once the module under test has configured logging."""
        real_setup = module.setup_logging

        def setup_then_attach(*args, **kwargs):
            real_setup(*args, **kwargs)
            logging.getLogger("bot").addHandler(self)

        monkeypatch.setattr(module, "setup_logging", setup_then_attach)


def test_one_full_cycle_raises_nothing_for_any_stock(workdir, monkeypatch):
    """The loop swallows per-stock exceptions, so only a test can see them.

    A live run skipped all nine stocks with `NameError: name 'timeframe' is
    not defined` -- caught by `except Exception`, logged, and stepped over.
    Every unit test passed, because none of them ran a cycle.
    """
    import main

    collector = _ErrorCollector()
    collector.attach_after_setup(main, monkeypatch)
    try:
        assert main.main(["paper", "--once"]) == 0
    finally:
        logging.getLogger("bot").removeHandler(collector)

    problems = [
        f"{r.name}: {r.getMessage()}"
        for r in collector.records
        if "unexpected error" in r.getMessage() or r.exc_info is not None
    ]
    assert not problems, "a full cycle logged errors:\n" + "\n".join(problems)


def test_every_configured_stock_is_reached_in_a_cycle(workdir, monkeypatch):
    """A stock silently missing from the cycle is a stock with no stop."""
    import main

    seen: list[str] = []
    original = main.TradingEngine._process_code

    def spy(self, code, *args, **kwargs):
        seen.append(code)
        return original(self, code, *args, **kwargs)

    monkeypatch.setattr(main.TradingEngine, "_process_code", spy)
    assert main.main(["paper", "--once"]) == 0

    from settings import load_config

    configured = {
        str(c)
        for c, cfg in (load_config(None).get("universe") or {}).items()
        if cfg.get("enabled", True)
    }
    assert set(seen) == configured


# ---------------------------------------------------------------------------
# 11. an open position is checked every tick, whatever its timeframe
# ---------------------------------------------------------------------------


def _engine(workdir):
    import argparse

    import main

    args = argparse.Namespace(config=None, once=True, dry_run=True, live=False, watch=False)
    return main.TradingEngine(main.build_runtime(args, cli_live=False, force_dry_run=True))


#: A stock configured on daily bars. Picked by timeframe rather than hardcoded,
#: so reassigning strategies in config.yaml cannot quietly turn these into
#: tests of something else.
def _a_daily_code(engine) -> str:
    for code in engine.rt.strategies:
        if engine.rt.universe.get(code, {}).get("timeframe") == "1Day":
            return code
    raise AssertionError("no daily stock configured")


def test_a_daily_stock_is_due_once_a_day_when_flat(workdir):
    engine = _engine(workdir)
    code = _a_daily_code(engine)
    now = 1000.0
    assert code in engine.due_codes(now)      # first look
    engine.mark_ran(code, now)
    # A daily stock's signal cadence is 24h; an hour later it is not due.
    assert code not in engine.due_codes(now + 3600)


def test_a_daily_stock_with_a_position_is_due_every_tick(workdir):
    """The hard stop lives in this loop -- Kiwoom holds no resting stop."""
    from portfolio import LONG, Position

    engine = _engine(workdir)
    code = _a_daily_code(engine)
    now = 1000.0
    engine.mark_ran(code, now)
    assert code not in engine.due_codes(now + 3600)

    engine.rt.portfolio.open_position(
        Position(
            symbol=code,
            side=LONG,
            qty=1,
            entry_price=34_600.0,
            stop_price=31_400.0,
            stop_distance=3_200.0,
        )
    )
    assert code in engine.due_codes(now + 30)
    assert code in engine.due_codes(now + 3600)
