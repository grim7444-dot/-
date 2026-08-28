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
from dataclasses import dataclass
from datetime import datetime, time, timedelta

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

    def spy(args, cli_live, force_dry_run=False, run_screener=False):
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


#: Mid-morning: inside the session, outside every configured entry window, so
#: the cadence is the strategy's own rather than the entry-window override.
QUIET = datetime(2026, 8, 19, 11, 0, tzinfo=KST)


def test_a_daily_stock_is_due_once_a_day_when_flat(workdir):
    engine = _engine(workdir)
    code = _a_daily_code(engine)
    now = 1000.0
    assert code in engine.due_codes(now, moment=QUIET)      # first look
    engine.mark_ran(code, now)
    # A daily stock's signal cadence is 24h; an hour later it is not due.
    assert code not in engine.due_codes(now + 3600, moment=QUIET)


def test_a_stock_inside_its_entry_window_is_due_every_minute(workdir):
    """A four-minute window on a daily cadence would never be looked at."""
    engine = _engine(workdir)
    # A stock whose window does not contain QUIET, so the two cadences differ.
    code = next(
        c
        for c in engine.rt.strategies
        if engine.session_rules(c).entry_from
        and not engine.session_rules(c).entry_allowed(QUIET)[0]
    )
    rules = engine.session_rules(code)
    inside = QUIET.replace(hour=rules.entry_from.hour, minute=rules.entry_from.minute)
    now = 1000.0
    engine.mark_ran(code, now)
    assert code not in engine.due_codes(now + 90, moment=QUIET)
    assert code in engine.due_codes(now + 90, moment=inside)


def test_a_daily_stock_with_a_position_is_due_every_tick(workdir):
    """The hard stop lives in this loop -- Kiwoom holds no resting stop."""
    from portfolio import LONG, Position

    engine = _engine(workdir)
    code = _a_daily_code(engine)
    now = 1000.0
    engine.mark_ran(code, now)
    assert code not in engine.due_codes(now + 3600, moment=QUIET)

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
    assert code in engine.due_codes(now + 30, moment=QUIET)
    assert code in engine.due_codes(now + 3600, moment=QUIET)


# ---------------------------------------------------------------------------
# 12. observation reads real data and cannot order
# ---------------------------------------------------------------------------


def _credentials(label: str, with_keys: bool):
    from settings import Credentials, Secret

    blank = Secret("")
    key = Secret("k" * 43) if with_keys else blank
    secret = Secret("s" * 43) if with_keys else blank
    return Credentials(
        app_key=key,
        secret_key=secret,
        account_no=blank,
        telegram_token=blank,
        telegram_chat_id=blank,
        loaded_for=label,
    )


def test_observation_uses_the_real_broker_when_credentials_exist(monkeypatch, workdir):
    """Watching a synthetic random walk all day is worse than not watching."""
    import broker as broker_module
    from broker import DryRunBroker, ReadOnlyBroker
    from settings import resolve_mode

    monkeypatch.setattr(broker_module, "KiwoomBroker", lambda *a, **k: object())
    decision = resolve_mode({}, cli_live=False)
    creds = _credentials(decision.label, with_keys=True)
    built = broker_module.build_broker(decision, creds, {}, force_dry_run=True)
    assert isinstance(built, ReadOnlyBroker)
    assert not isinstance(built, DryRunBroker)
    assert built.dry_run is True


def test_observation_falls_back_to_simulation_without_credentials(workdir):
    from broker import DryRunBroker, build_broker
    from settings import resolve_mode

    decision = resolve_mode({}, cli_live=False)
    built = build_broker(decision, _credentials(decision.label, with_keys=False), {}, force_dry_run=True)
    assert isinstance(built, DryRunBroker)


class _RecordingInner:
    dry_run = False

    def __init__(self):
        self.orders = 0

    def submit_order(self, *a, **k):
        self.orders += 1
        raise AssertionError("a read-only broker must not reach the order path")

    def cancel_all_orders(self):
        raise AssertionError("a read-only broker must not cancel")

    def close_all_positions(self):
        raise AssertionError("a read-only broker must not flatten")

    def get_chart(self, code, timeframe, start, end, max_rows=None):
        return "real bars"


def test_a_read_only_broker_refuses_every_write_path():
    from broker import ReadOnlyBroker

    inner = _RecordingInner()
    observer = ReadOnlyBroker(inner)

    result = observer.submit_order("002990", "LONG", 10)
    assert result.submitted is False
    assert observer.cancel_all_orders() == 0
    assert observer.close_all_positions() == 0
    assert inner.orders == 0


def test_a_read_only_broker_still_serves_real_bars():
    from broker import ReadOnlyBroker

    observer = ReadOnlyBroker(_RecordingInner())
    assert observer.get_chart("002990", "30Min", None, None) == "real bars"


def test_observation_gets_its_own_mode_label():
    """Its positions diverge from the account's, so its equity is its own."""
    from broker import DryRunBroker, ReadOnlyBroker

    assert ReadOnlyBroker(_RecordingInner()).mode_suffix == "-OBSERVE"
    assert DryRunBroker().mode_suffix == "-SIM"


# ---------------------------------------------------------------------------
# 13. realtime volume-surge/VI candidates stand in when pykrx's snapshot is
#    empty (2026-08-27: pykrx's "today" trading-value data isn't populated
#    until well after the open, so the screener legitimately sees 0
#    candidates for a while every morning)
# ---------------------------------------------------------------------------


class _FakeRealtimeBroker:
    """A broker double exposing only get_volume_surge/get_vi_triggered."""

    def __init__(self, surges_by_market=None, vis_by_market=None):
        self._surges = surges_by_market or {}
        self._vis = vis_by_market or {}

    def get_volume_surge(self, market="000"):
        return self._surges.get(market, [])

    def get_vi_triggered(self, market="000"):
        return self._vis.get(market, [])


class _FakeRT:
    def __init__(self, config, broker):
        self.config = config
        self.broker = broker


def _make_realtime_engine(config, broker):
    from main import TradingEngine

    engine = TradingEngine.__new__(TradingEngine)
    engine.rt = _FakeRT(config, broker)
    return engine


def test_realtime_candidates_tags_each_pick_with_its_real_market():
    """A pick's market must come from which mrkt_tp call found it.

    _realtime_candidates() used to leave "market" unset entirely, which
    rt.market_of() silently defaults to KOSPI -- wrong tick size and sell-tax
    rate for a KOSDAQ pick, both of which differ by market in market/rules.py.
    """
    from broker import VolumeSurgeCandidate
    from market.rules import KOSDAQ, KOSPI

    broker = _FakeRealtimeBroker(
        surges_by_market={
            "001": [VolumeSurgeCandidate(code="005930", name="삼성전자", price=72_500.0, surge_rate=50.0)],
            "101": [
                VolumeSurgeCandidate(code="082800", name="비보존제약", price=3_200.0, surge_rate=80.0),
                VolumeSurgeCandidate(code="000009", name="너무싸다", price=500.0, surge_rate=99.0),
                VolumeSurgeCandidate(code="000010", name="너무비싸다", price=999_999.0, surge_rate=98.0),
            ],
        }
    )
    config = {
        "screener": {"min_price": 2000, "max_price": 100_000, "n_stocks": 5},
        "universe": {},
    }
    engine = _make_realtime_engine(config, broker)
    results = dict(engine._realtime_candidates())

    assert "000009" not in results  # below min_price
    assert "000010" not in results  # above max_price
    assert results["082800"]["market"] == KOSDAQ
    assert results["005930"]["market"] == KOSPI
    # ranked by surge_rate desc, both markets pooled together
    assert list(results) == ["082800", "005930"]


def test_realtime_candidates_excludes_existing_universe_and_caps_n_stocks():
    from broker import VolumeSurgeCandidate

    surges = [
        VolumeSurgeCandidate(code=f"00000{i}", name=f"s{i}", price=5_000.0, surge_rate=float(10 - i))
        for i in range(6)
    ]
    broker = _FakeRealtimeBroker(surges_by_market={"001": surges})
    config = {
        "screener": {"min_price": 2000, "max_price": 0, "n_stocks": 3},
        "universe": {"000002": {"name": "already tracked"}},
    }
    engine = _make_realtime_engine(config, broker)
    results = engine._realtime_candidates()

    codes = [c for c, _ in results]
    assert "000002" not in codes  # already in the universe -- not a new pick
    assert codes == ["000000", "000001", "000003"]  # next-highest surge_rate, capped at n_stocks


def test_realtime_candidates_alternates_orb_and_pullback_with_independent_params():
    from broker import VolumeSurgeCandidate

    surges = [
        VolumeSurgeCandidate(code=f"10000{i}", name=f"s{i}", price=5_000.0, surge_rate=float(10 - i))
        for i in range(4)
    ]
    broker = _FakeRealtimeBroker(surges_by_market={"001": surges})
    config = {"screener": {"min_price": 2000, "max_price": 0, "n_stocks": 4}, "universe": {}}
    engine = _make_realtime_engine(config, broker)
    results = engine._realtime_candidates()

    assert [cfg["strategy"] for _, cfg in results] == [
        "orb", "pullback_bounce", "orb", "pullback_bounce",
    ]
    # each candidate must own its params dict -- not share the template's.
    results[0][1]["params"]["stop_pct"] = 0.999
    assert results[2][1]["params"]["stop_pct"] != 0.999


def test_realtime_candidates_falls_back_to_vi_when_no_volume_surge_picks():
    from broker import ViTriggeredStock

    vis = [
        ViTriggeredStock(code="082800", name="비보존제약", trigger_price=3_200.0, trigger_count_today=2),
        ViTriggeredStock(code="032820", name="우리기술", trigger_price=2_500.0, trigger_count_today=1),
    ]
    broker = _FakeRealtimeBroker(vis_by_market={"101": vis})
    config = {"screener": {"min_price": 2000, "max_price": 0, "n_stocks": 5}, "universe": {}}
    engine = _make_realtime_engine(config, broker)
    results = dict(engine._realtime_candidates())

    assert list(results) == ["082800", "032820"]  # ranked by trigger_count_today desc


def test_realtime_candidates_prefers_volume_surge_over_vi_when_both_present():
    """ka10023 was the user's explicit primary pick, ka10054 secondary."""
    from broker import ViTriggeredStock, VolumeSurgeCandidate

    broker = _FakeRealtimeBroker(
        surges_by_market={"001": [VolumeSurgeCandidate(code="005930", name="s", price=5000.0, surge_rate=1.0)]},
        vis_by_market={"101": [ViTriggeredStock(code="082800", name="v", trigger_price=5000.0, trigger_count_today=99)]},
    )
    config = {"screener": {"min_price": 2000, "max_price": 0, "n_stocks": 5}, "universe": {}}
    engine = _make_realtime_engine(config, broker)
    results = engine._realtime_candidates()

    assert [c for c, _ in results] == ["005930", "082800"]


def test_realtime_candidates_tolerates_one_market_failing():
    """Kiwoom erroring on one of the two market queries must not zero out the other."""
    from broker import VolumeSurgeCandidate

    class _FlakyBroker:
        def get_volume_surge(self, market="000"):
            if market == "001":
                raise RuntimeError("kiwoom 500")
            return [VolumeSurgeCandidate(code="082800", name="비보존제약", price=3_200.0, surge_rate=10.0)]

        def get_vi_triggered(self, market="000"):
            return []

    config = {"screener": {"min_price": 2000, "max_price": 0, "n_stocks": 5}, "universe": {}}
    engine = _make_realtime_engine(config, _FlakyBroker())
    results = engine._realtime_candidates()

    assert [c for c, _ in results] == ["082800"]


def test_refresh_dynamic_universe_uses_realtime_candidates_when_pykrx_scan_is_empty(monkeypatch):
    """The exact scenario this feature exists for: pykrx scan returns 0, mid-open."""
    import screener as screener_module
    from broker import VolumeSurgeCandidate
    from market.rules import KOSDAQ
    from main import TradingEngine

    class _EmptyScanScreener:
        last_scan_failed = False

        def __init__(self, config):
            pass

        def scan(self):
            return []

    monkeypatch.setattr(screener_module, "DailyScreener", _EmptyScanScreener)

    broker = _FakeRealtimeBroker(
        surges_by_market={
            "101": [VolumeSurgeCandidate(code="082800", name="비보존제약", price=3_200.0, surge_rate=50.0)],
        }
    )
    config = {
        "screener": {"enabled": True, "min_price": 2000, "max_price": 0, "n_stocks": 5},
        "universe": {},
        "risk": {},
    }

    class _FakePortfolio:
        def positions(self):
            return []

    class _FakeNotifier:
        def send(self, *a, **k):
            pass

    engine = TradingEngine.__new__(TradingEngine)
    engine.rt = _FakeRT(config, broker)
    engine.rt.portfolio = _FakePortfolio()
    engine.rt.strategies = {}
    engine._session_rules = {}
    engine._tg_notifier = _FakeNotifier()

    engine._refresh_dynamic_universe()

    universe = engine.rt.config["universe"]
    assert "082800" in universe
    assert universe["082800"]["_realtime"] is True
    assert universe["082800"]["market"] == KOSDAQ
    assert "082800" in engine.rt.strategies
    assert "082800" in engine._session_rules


def test_refresh_dynamic_universe_prefers_a_working_pykrx_scan_over_realtime(monkeypatch):
    """Realtime candidates are a stand-in for an empty scan, not a replacement for a working one."""
    import screener as screener_module
    from main import TradingEngine

    pykrx_pick_cfg = {
        "enabled": True, "strategy": "orb", "timeframe": "1Min", "market": "KOSPI",
        "params": {}, "_screener": True,
    }

    class _WorkingScreener:
        last_scan_failed = False

        def __init__(self, config):
            pass

        def scan(self):
            return [("005930", pykrx_pick_cfg)]

    monkeypatch.setattr(screener_module, "DailyScreener", _WorkingScreener)

    calls: list[str] = []

    class _NoisyBroker(_FakeRealtimeBroker):
        def get_volume_surge(self, market="000"):
            calls.append(market)
            return super().get_volume_surge(market)

    config = {
        "screener": {"enabled": True, "min_price": 2000, "max_price": 0, "n_stocks": 5},
        "universe": {},
        "risk": {},
    }

    class _FakePortfolio:
        def positions(self):
            return []

    class _FakeNotifier:
        def send(self, *a, **k):
            pass

    engine = TradingEngine.__new__(TradingEngine)
    engine.rt = _FakeRT(config, _NoisyBroker())
    engine.rt.portfolio = _FakePortfolio()
    engine.rt.strategies = {}
    engine._session_rules = {}
    engine._tg_notifier = _FakeNotifier()

    engine._refresh_dynamic_universe()

    assert "005930" in engine.rt.config["universe"]
    assert not calls  # _realtime_candidates() must not even be consulted


# ---------------------------------------------------------------------------
# 14. late-session opportunistic take-profit (2026-08-27, user request):
#    a stalled position in the last 20 minutes before its force exit takes a
#    smaller profit now rather than riding to the forced exit
# ---------------------------------------------------------------------------


FORCE_EXIT_1510 = time(15, 10)


def _late_exit_position(entry_price: float, highest_price: float) -> "Position":
    from portfolio import LONG, Position

    return Position(
        symbol="005930", side=LONG, qty=1,
        entry_price=entry_price, stop_price=entry_price * 0.98, stop_distance=entry_price * 0.02,
        highest_price=highest_price,
    )


def _late_exit_rules(force_exit_at=FORCE_EXIT_1510, hold_overnight=False):
    from market.session_rules import SessionRules

    return SessionRules(force_exit_at=force_exit_at, hold_overnight=hold_overnight)


def test_late_exit_fires_when_stalled_and_still_profitable_near_the_close():
    from main import _late_exit_reason

    # Peaked at +2.5% (armed but never reached the 2% lock's usual bar in
    # this scenario -- entry 10,000, peak 10,250), now pulled back to 10,080
    # (-1.66% off peak) but still +0.8% over entry.
    position = _late_exit_position(entry_price=10_000.0, highest_price=10_250.0)
    now = datetime(2026, 8, 27, 14, 55, tzinfo=KST)  # 15 minutes before 15:10
    reason = _late_exit_reason(position, 10_080.0, now, _late_exit_rules(), {})
    assert reason is not None
    assert "조기 익절" in reason


def test_late_exit_does_nothing_outside_the_20_minute_window():
    from main import _late_exit_reason

    position = _late_exit_position(entry_price=10_000.0, highest_price=10_250.0)
    now = datetime(2026, 8, 27, 14, 49, tzinfo=KST)  # 21 minutes before 15:10
    assert _late_exit_reason(position, 10_080.0, now, _late_exit_rules(), {}) is None


def test_late_exit_boundary_is_inclusive_at_exactly_20_minutes():
    from main import _late_exit_reason

    position = _late_exit_position(entry_price=10_000.0, highest_price=10_250.0)
    now = datetime(2026, 8, 27, 14, 50, tzinfo=KST)  # exactly 20 minutes before 15:10
    assert _late_exit_reason(position, 10_080.0, now, _late_exit_rules(), {}) is not None


def test_late_exit_does_nothing_once_the_stall_pullback_is_too_small():
    from main import _late_exit_reason

    # Only 1% off peak (10,147.5), short of the 1.5% stall threshold.
    position = _late_exit_position(entry_price=10_000.0, highest_price=10_250.0)
    now = datetime(2026, 8, 27, 14, 55, tzinfo=KST)
    assert _late_exit_reason(position, 10_147.5, now, _late_exit_rules(), {}) is None


def test_late_exit_does_nothing_when_profit_is_too_thin():
    from main import _late_exit_reason

    # Stalled 1.6% off peak, but current gain is only +0.3%, below the 0.5% floor.
    position = _late_exit_position(entry_price=10_000.0, highest_price=10_250.0)
    now = datetime(2026, 8, 27, 14, 55, tzinfo=KST)
    assert _late_exit_reason(position, 10_030.0, now, _late_exit_rules(), {}) is None


def test_late_exit_exempts_close_auction_overnight_holds():
    """These are meant to be held past the close -- a 'stall' here is just the plan."""
    from main import _late_exit_reason

    position = _late_exit_position(entry_price=10_000.0, highest_price=10_250.0)
    now = datetime(2026, 8, 27, 14, 55, tzinfo=KST)
    rules = _late_exit_rules(force_exit_at=time(9, 5), hold_overnight=True)
    assert _late_exit_reason(position, 10_080.0, now, rules, {}) is None


def test_late_exit_does_nothing_without_a_force_exit_configured():
    from main import _late_exit_reason

    position = _late_exit_position(entry_price=10_000.0, highest_price=10_250.0)
    now = datetime(2026, 8, 27, 14, 55, tzinfo=KST)
    assert _late_exit_reason(position, 10_080.0, now, _late_exit_rules(force_exit_at=None), {}) is None


def test_late_exit_thresholds_are_configurable_via_risk_cfg():
    from main import _late_exit_reason

    position = _late_exit_position(entry_price=10_000.0, highest_price=10_250.0)
    now = datetime(2026, 8, 27, 14, 55, tzinfo=KST)
    # Default thresholds would fire (see the first test above); a stricter
    # config (5% stall required) must suppress it.
    strict_cfg = {"late_exit_stall_pct": 0.05}
    assert _late_exit_reason(position, 10_080.0, now, _late_exit_rules(), strict_cfg) is None
    # A narrower window (5 minutes) must suppress it at the 15-minutes-out mark.
    narrow_window_cfg = {"late_exit_minutes": 5}
    assert _late_exit_reason(position, 10_080.0, now, _late_exit_rules(), narrow_window_cfg) is None


# ---------------------------------------------------------------------------
# 15. the daily 복기 report fires once at the close, and the weekly rollup
#    fires alongside it every Friday (2026-08-27, user request)
# ---------------------------------------------------------------------------


class _FakeCalendarForReport:
    def __init__(self, business_day: bool = True):
        self._business_day = business_day

    def is_business_day(self, day):
        return self._business_day


class _FakeDecisionForReport:
    label = "PAPER"


class _CapturingNotifier:
    def __init__(self):
        self.sent: list[str] = []

    def send(self, text, *a, **k):
        self.sent.append(text)


def _report_engine(portfolio, config):
    from main import TradingEngine

    class _FakeRTForReport:
        pass

    engine = TradingEngine.__new__(TradingEngine)
    rt = _FakeRTForReport()
    rt.config = config
    rt.portfolio = portfolio
    rt.risk = RiskManager(config, portfolio)
    rt.decision = _FakeDecisionForReport()
    rt.calendar = _FakeCalendarForReport()
    engine.rt = rt
    engine._daily_report_date = None
    engine._tg_notifier = _CapturingNotifier()
    return engine


def _freeze_main_clock(monkeypatch, moment: datetime) -> None:
    import main

    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return moment

    monkeypatch.setattr(main, "datetime", _Frozen)


def test_daily_report_fires_once_after_the_close(portfolio, config, monkeypatch):
    engine = _report_engine(portfolio, config)
    _freeze_main_clock(monkeypatch, datetime(2026, 8, 26, 15, 31, tzinfo=KST))  # Wed, just after 15:30

    engine._maybe_send_daily_report()
    assert len(engine._tg_notifier.sent) == 1
    assert "Evening report" in engine._tg_notifier.sent[0]
    assert "복기" in engine._tg_notifier.sent[0]

    # A second call the same day (e.g. next loop tick while still closed) must not resend.
    engine._maybe_send_daily_report()
    assert len(engine._tg_notifier.sent) == 1


def test_daily_report_does_not_fire_before_the_close(portfolio, config, monkeypatch):
    engine = _report_engine(portfolio, config)
    _freeze_main_clock(monkeypatch, datetime(2026, 8, 26, 8, 0, tzinfo=KST))  # pre-open "closed" window

    engine._maybe_send_daily_report()
    assert engine._tg_notifier.sent == []


def test_daily_report_skips_a_non_business_day(portfolio, config, monkeypatch):
    engine = _report_engine(portfolio, config)
    engine.rt.calendar = _FakeCalendarForReport(business_day=False)
    _freeze_main_clock(monkeypatch, datetime(2026, 8, 22, 16, 0, tzinfo=KST))  # a Saturday

    engine._maybe_send_daily_report()
    assert engine._tg_notifier.sent == []


def test_friday_also_sends_the_weekly_report(portfolio, config, monkeypatch):
    engine = _report_engine(portfolio, config)
    _freeze_main_clock(monkeypatch, datetime(2026, 8, 28, 15, 31, tzinfo=KST))  # a Friday

    engine._maybe_send_daily_report()
    assert len(engine._tg_notifier.sent) == 2
    assert "Evening report" in engine._tg_notifier.sent[0]
    assert "Weekly report" in engine._tg_notifier.sent[1]


def test_a_non_friday_does_not_send_the_weekly_report(portfolio, config, monkeypatch):
    engine = _report_engine(portfolio, config)
    _freeze_main_clock(monkeypatch, datetime(2026, 8, 26, 15, 31, tzinfo=KST))  # Wednesday

    engine._maybe_send_daily_report()
    assert len(engine._tg_notifier.sent) == 1


# ---------------------------------------------------------------------------
# 16. 상한가 (upper-limit) candidates hold overnight instead of selling into
#    the close (2026-08-27, user request)
# ---------------------------------------------------------------------------


def _near_limit_position(entry_price: float, entry_moment: datetime, near_limit_hold: bool = False) -> "Position":
    from portfolio import LONG, Position

    return Position(
        symbol="005930", side=LONG, qty=1,
        entry_price=entry_price, stop_price=entry_price * 0.98, stop_distance=entry_price * 0.02,
        entry_time=entry_moment.isoformat(),
        near_limit_hold=near_limit_hold,
    )


def test_near_limit_flags_a_position_the_first_time_it_crosses_the_threshold():
    from main import _near_limit_decision

    entered = datetime(2026, 8, 27, 9, 30, tzinfo=KST)
    position = _near_limit_position(10_000.0, entered)
    now = datetime(2026, 8, 27, 13, 0, tzinfo=KST)  # same day, well before any exit
    should_flag, reason = _near_limit_decision(position, 12_700.0, now, {})  # +27%
    assert should_flag is True
    assert reason is None  # not exited yet -- same-day, holds through the close


def test_near_limit_does_not_flag_below_the_threshold():
    from main import _near_limit_decision

    entered = datetime(2026, 8, 27, 9, 30, tzinfo=KST)
    position = _near_limit_position(10_000.0, entered)
    now = datetime(2026, 8, 27, 13, 0, tzinfo=KST)
    should_flag, reason = _near_limit_decision(position, 12_400.0, now, {})  # +24%, short of 26%
    assert should_flag is False
    assert reason is None


def test_near_limit_does_not_exit_same_day_even_once_flagged():
    from main import _near_limit_decision

    entered = datetime(2026, 8, 27, 9, 30, tzinfo=KST)
    position = _near_limit_position(10_000.0, entered, near_limit_hold=True)
    now = datetime(2026, 8, 27, 15, 25, tzinfo=KST)  # late the same day
    should_flag, reason = _near_limit_decision(position, 13_000.0, now, {})
    assert should_flag is False  # already flagged -- nothing new to persist
    assert reason is None  # still same day -- must not sell yet


def test_near_limit_exits_the_next_morning_at_the_configured_time():
    from main import _near_limit_decision

    entered = datetime(2026, 8, 27, 9, 30, tzinfo=KST)
    position = _near_limit_position(10_000.0, entered, near_limit_hold=True)
    now = datetime(2026, 8, 28, 9, 5, tzinfo=KST)  # next morning, exactly 09:05
    should_flag, reason = _near_limit_decision(position, 13_000.0, now, {})
    assert should_flag is False
    assert reason is not None
    assert "익일 오전" in reason


def test_near_limit_waits_for_the_configured_exit_time_the_next_morning():
    from main import _near_limit_decision

    entered = datetime(2026, 8, 27, 9, 30, tzinfo=KST)
    position = _near_limit_position(10_000.0, entered, near_limit_hold=True)
    now = datetime(2026, 8, 28, 9, 2, tzinfo=KST)  # next morning, before 09:05
    should_flag, reason = _near_limit_decision(position, 13_000.0, now, {})
    assert reason is None  # continuous trading barely open -- wait a few more minutes


def test_near_limit_threshold_and_exit_time_are_configurable():
    from main import _near_limit_decision

    entered = datetime(2026, 8, 27, 9, 30, tzinfo=KST)
    position = _near_limit_position(10_000.0, entered)
    now = datetime(2026, 8, 27, 13, 0, tzinfo=KST)
    # +27% would flag with the default 26% threshold, but not with a 30% one.
    should_flag, _ = _near_limit_decision(position, 12_700.0, now, {"near_limit_hold_pct": 0.30})
    assert should_flag is False

    flagged = _near_limit_position(10_000.0, entered, near_limit_hold=True)
    later = datetime(2026, 8, 28, 9, 6, tzinfo=KST)
    _, reason = _near_limit_decision(flagged, 13_000.0, later, {"near_limit_exit_time": "09:30"})
    assert reason is None  # 09:06 is still before the configured 09:30


# ---------------------------------------------------------------------------
# 17. a position that dipped hard and bounced takes a smaller profit than a
#    smooth winner would (2026-08-27, user request)
# ---------------------------------------------------------------------------


def _dip_position(entry_price: float, lowest_price: float | None) -> "Position":
    from portfolio import LONG, Position

    return Position(
        symbol="005930", side=LONG, qty=1,
        entry_price=entry_price, stop_price=entry_price * 0.98, stop_distance=entry_price * 0.02,
        lowest_price=lowest_price,
    )


def test_dip_recovery_fires_after_an_18pct_dip_bounces_to_15pct():
    from main import _dip_recovery_reason

    position = _dip_position(10_000.0, lowest_price=9_810.0)  # -1.9% at its lowest
    reason = _dip_recovery_reason(position, 10_150.0, {})  # now +1.5%
    assert reason is not None
    assert "조기 익절" in reason


def test_dip_recovery_does_nothing_for_a_position_that_never_dipped():
    """A smooth winner must be untouched -- this rule is only for the dip pattern."""
    from main import _dip_recovery_reason

    position = _dip_position(10_000.0, lowest_price=9_950.0)  # only -0.5% at its lowest
    reason = _dip_recovery_reason(position, 10_150.0, {})  # +1.5%, same gain as above
    assert reason is None


def test_dip_recovery_waits_for_the_recovery_target_even_after_a_deep_dip():
    from main import _dip_recovery_reason

    position = _dip_position(10_000.0, lowest_price=9_800.0)  # -2.0% dip
    reason = _dip_recovery_reason(position, 10_100.0, {})  # only +1.0%, short of 1.5%
    assert reason is None


def test_dip_recovery_does_nothing_with_no_recorded_lowest_price():
    from main import _dip_recovery_reason

    position = _dip_position(10_000.0, lowest_price=None)
    assert _dip_recovery_reason(position, 10_150.0, {}) is None


def test_dip_recovery_thresholds_are_configurable():
    from main import _dip_recovery_reason

    position = _dip_position(10_000.0, lowest_price=9_810.0)  # -1.9% dip
    # A stricter dip requirement (3%) must suppress it even though the
    # default 1.8% would have fired (see the first test above).
    strict_cfg = {"dip_recovery_dip_pct": 0.03}
    assert _dip_recovery_reason(position, 10_150.0, strict_cfg) is None
    # A higher recovery bar (2.5%) must also suppress a mere +1.5%.
    higher_bar_cfg = {"dip_recovery_profit_pct": 0.025}
    assert _dip_recovery_reason(position, 10_150.0, higher_bar_cfg) is None


# ---------------------------------------------------------------------------
# 18. daily profit lock: once today's gain clears the target, no new entry
#    (2026-08-27, user request: 하루 최소 2%이상은 남겨야 되)
# ---------------------------------------------------------------------------


def test_daily_lock_blocks_a_new_entry_once_the_target_is_reached():
    from main import _daily_profit_lock_reason

    reason = _daily_profit_lock_reason(equity=459_000.0, day_start_equity=450_000.0, lock_pct=0.02)
    assert reason is not None
    assert "잠금" in reason


def test_daily_lock_allows_entries_below_the_target():
    from main import _daily_profit_lock_reason

    # +1% -- short of the 2% target.
    assert _daily_profit_lock_reason(450_000.0 * 1.01, 450_000.0, 0.02) is None


def test_daily_lock_allows_entries_on_a_losing_day():
    from main import _daily_profit_lock_reason

    assert _daily_profit_lock_reason(440_000.0, 450_000.0, 0.02) is None


def test_daily_lock_disabled_at_zero_never_blocks():
    from main import _daily_profit_lock_reason

    assert _daily_profit_lock_reason(999_000.0, 450_000.0, 0.0) is None


def test_daily_lock_does_nothing_without_a_known_day_start_equity():
    """day_start_equity == 0 means mark_equity() hasn't run yet for today -- nothing to compare against."""
    from main import _daily_profit_lock_reason

    assert _daily_profit_lock_reason(459_000.0, 0.0, 0.02) is None


def test_daily_lock_threshold_is_configurable():
    from main import _daily_profit_lock_reason

    # +2% would lock under the default 2% target, but not under a 5% one.
    assert _daily_profit_lock_reason(459_000.0, 450_000.0, 0.05) is None


# ---------------------------------------------------------------------------
# 19. orderbook confirmation flipped to ask-side dominance (2026-08-27, user
#    request): 매도잔량이 매수잔량보다 훨씬 많아야 상승 신호로 본다
# ---------------------------------------------------------------------------


@dataclass
class _FakeQtyOrderBook:
    total_bid_qty: float = 0.0
    total_ask_qty: float = 0.0


def test_ask_bid_ratio_is_high_when_asks_dominate():
    from main import _ask_bid_ratio

    book = _FakeQtyOrderBook(total_bid_qty=1_000.0, total_ask_qty=3_000.0)
    assert _ask_bid_ratio(book) == pytest.approx(3.0)


def test_ask_bid_ratio_is_low_when_bids_dominate():
    """A thick buy wall must NOT read as bullish under the new convention."""
    from main import _ask_bid_ratio

    book = _FakeQtyOrderBook(total_bid_qty=3_000.0, total_ask_qty=1_000.0)
    assert _ask_bid_ratio(book) == pytest.approx(1 / 3)


def test_ask_bid_ratio_handles_zero_bid_quantity():
    from main import _ask_bid_ratio

    assert _ask_bid_ratio(_FakeQtyOrderBook(total_bid_qty=0.0, total_ask_qty=500.0)) == float("inf")
    assert _ask_bid_ratio(_FakeQtyOrderBook(total_bid_qty=0.0, total_ask_qty=0.0)) == 1.0


# ---------------------------------------------------------------------------
# 20. volume-surge/VI codes strip Kiwoom's combined-market venue suffix
#    (2026-08-28: a live run added 12 codes like "108860_AL" to the
#    universe -- every chart/order lookup for them failed from then on,
#    since every other broker call expects a bare 6-digit code)
# ---------------------------------------------------------------------------


def test_clean_stock_code_strips_a_trailing_venue_suffix():
    from broker import _clean_stock_code

    assert _clean_stock_code("108860_AL") == "108860"


def test_clean_stock_code_strips_a_leading_market_prefix():
    from broker import _clean_stock_code

    assert _clean_stock_code("A005930") == "005930"


def test_clean_stock_code_passes_through_a_bare_code():
    from broker import _clean_stock_code

    assert _clean_stock_code("005930") == "005930"


def test_clean_stock_code_returns_empty_for_junk():
    from broker import _clean_stock_code

    assert _clean_stock_code("") == ""
    assert _clean_stock_code(None) == ""
    assert _clean_stock_code("AL") == ""


def test_volume_surge_parses_a_suffixed_code_clean():
    broker = _broker_returning({
        "trde_qty_sdnin": [
            {"stk_cd": "108860_AL", "stk_nm": "셀바스AI", "cur_prc": "5000",
             "flu_rt": "3.5", "now_trde_qty": "10000", "sdnin_qty": "5000", "sdnin_rt": "80"},
        ],
    })
    out = broker.get_volume_surge()
    assert len(out) == 1
    assert out[0].code == "108860"


def test_vi_triggered_parses_a_suffixed_code_clean():
    broker = _broker_returning({
        "motn_stk": [
            {"stk_cd": "376900_AL", "stk_nm": "로킷헬스케어", "motn_pric": "10000",
             "open_pric_pre_flu_rt": "5.0", "vimotn_cnt": "1", "viaplc_tp": "1"},
        ],
    })
    out = broker.get_vi_triggered()
    assert len(out) == 1
    assert out[0].code == "376900"


# ---------------------------------------------------------------------------
# 21. ETN/ETF names are filtered out of the realtime volume-surge/VI scan
#    (2026-08-28: a live run added leveraged ETNs and brand-name ETFs to
#    the day-trading universe alongside real stocks)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [
    "하나 K방산TOP10 ETN",
    "한투 레버리지코스닥150선물 ETN",
    "삼성 코스닥 150 TR ETN",
    "WON 미국빌리어네어",
    "KODEX 미국나스닥AI테크액티브",
    "PLUS K리츠",
])
def test_is_fund_like_flags_every_instrument_seen_live(name):
    from main import _is_fund_like

    assert _is_fund_like(name) is True


@pytest.mark.parametrize("name", ["삼성전자", "비보존 제약", "씨피시스템", "엔켐"])
def test_is_fund_like_leaves_ordinary_stock_names_alone(name):
    from main import _is_fund_like

    assert _is_fund_like(name) is False


def test_realtime_candidates_excludes_fund_like_names():
    from broker import VolumeSurgeCandidate

    surges = [
        VolumeSurgeCandidate(code="005930", name="삼성전자", price=70_000.0, surge_rate=10.0),
        VolumeSurgeCandidate(code="411420", name="KODEX 미국나스닥AI테크액티브", price=15_000.0, surge_rate=90.0),
    ]
    broker = _FakeRealtimeBroker(surges_by_market={"001": surges})
    config = {"screener": {"min_price": 2000, "max_price": 0, "n_stocks": 5}, "universe": {}}
    engine = _make_realtime_engine(config, broker)
    results = dict(engine._realtime_candidates())

    assert "005930" in results
    assert "411420" not in results
