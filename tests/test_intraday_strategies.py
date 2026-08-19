"""Day trading (단타) and close-based trading (종가매매).

Both strategies are defined as much by *when* they act as by what they see, so
the time rules get as much attention here as the signals do. The two failure
modes that matter are a day trade that survives the close -- which makes it an
unhedged overnight position, since no stop exists while the bot is off -- and
an overnight position that is sold seconds after it is opened because its exit
time already passed today.
"""

from __future__ import annotations

from datetime import date, datetime, time

import pandas as pd
import pytest

from market.calendar import KST
from market.session_rules import SessionRules, parse_clock
from portfolio import LONG, Position
from strategies.base import Action
from strategies.close_auction import CloseAuction
from strategies.scalping import Scalping, session_vwap


# ---------------------------------------------------------------------------
# Session rules
# ---------------------------------------------------------------------------


DAY_TRADE = SessionRules(
    entry_from=time(9, 15),
    entry_until=time(14, 30),
    force_exit_at=time(15, 10),
)
OVERNIGHT = SessionRules(
    entry_from=time(15, 15),
    entry_until=time(15, 19),
    force_exit_at=time(9, 5),
    hold_overnight=True,
)

WEDNESDAY = date(2026, 8, 19)


def _at(hour: int, minute: int, day: date = WEDNESDAY) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=KST)


def test_parse_clock_rejects_anything_that_is_not_a_time():
    assert parse_clock("15:15", "x") == time(15, 15)
    with pytest.raises(ValueError, match="HH:MM"):
        parse_clock("quarter past three", "x")


@pytest.mark.parametrize(
    "moment, allowed",
    [
        ((9, 0), False),    # opening minutes: widest spread, no channel yet
        ((9, 14), False),
        ((9, 15), True),
        ((12, 0), True),
        ((14, 30), True),
        ((14, 31), False),  # too late to be worth opening before the flat-out
        ((15, 9), False),
    ],
)
def test_day_trade_entry_window(moment, allowed):
    assert DAY_TRADE.entry_allowed(_at(*moment))[0] is allowed


def test_day_trade_is_flattened_at_its_time_whatever_the_signal():
    assert DAY_TRADE.exit_due(_at(15, 9), WEDNESDAY)[0] is False
    due, why = DAY_TRADE.exit_due(_at(15, 10), WEDNESDAY)
    assert due is True
    assert "15:10" in why


def test_a_day_trade_rule_may_not_flatten_inside_the_closing_auction():
    """15:20 onward has no continuous book; the position would carry overnight."""
    with pytest.raises(ValueError, match="closing auction"):
        SessionRules(force_exit_at=time(15, 25)).validate()


def test_a_day_trade_may_not_enter_after_its_own_flat_out():
    with pytest.raises(ValueError, match="flattened"):
        SessionRules(
            entry_from=time(9, 15), entry_until=time(15, 15), force_exit_at=time(15, 10)
        ).validate()


def test_an_entry_window_that_is_never_open_is_rejected():
    with pytest.raises(ValueError, match="never open"):
        SessionRules(entry_from=time(15, 0), entry_until=time(9, 0)).validate()


def test_an_overnight_exit_before_the_open_is_rejected():
    with pytest.raises(ValueError, match="before the open"):
        SessionRules(force_exit_at=time(8, 40), hold_overnight=True).validate()


def test_an_overnight_position_is_not_sold_the_moment_it_is_opened():
    """The regression this exists for: 15:16 is past 09:05, but only today."""
    opened_today = WEDNESDAY
    assert OVERNIGHT.exit_due(_at(15, 16), opened_today)[0] is False
    assert OVERNIGHT.exit_due(_at(15, 30), opened_today)[0] is False


def test_an_overnight_position_is_sold_the_next_morning():
    thursday = date(2026, 8, 20)
    assert OVERNIGHT.exit_due(_at(9, 4, thursday), WEDNESDAY)[0] is False
    due, why = OVERNIGHT.exit_due(_at(9, 5, thursday), WEDNESDAY)
    assert due is True
    assert "2026-08-19" in why


def test_an_overnight_position_with_no_entry_date_is_not_force_sold():
    """Unknown entry day: hold and let the stop work, rather than guess."""
    assert OVERNIGHT.exit_due(_at(9, 30), None)[0] is False


def test_rules_come_out_of_config():
    rules = SessionRules.from_config(
        {"entry_window": ["15:15", "15:19"], "force_exit_at": "09:05", "hold_overnight": True}
    )
    assert rules == OVERNIGHT


def test_config_without_rules_constrains_nothing():
    rules = SessionRules.from_config({})
    assert rules.entry_allowed(_at(9, 0))[0] is True
    assert rules.exit_due(_at(15, 59), WEDNESDAY)[0] is False


def test_every_configured_stock_has_valid_rules():
    """A malformed window must fail at start-up, not at 15:19 one day."""
    from settings import load_config

    for code, cfg in (load_config(None).get("universe") or {}).items():
        SessionRules.from_config(cfg)   # raises if the config is wrong


# ---------------------------------------------------------------------------
# Scalping
# ---------------------------------------------------------------------------


def _intraday(closes, volumes=None, day=WEDNESDAY, start_hour=9):
    n = len(closes)
    index = pd.DatetimeIndex(
        [
            datetime(day.year, day.month, day.day, start_hour, 0, tzinfo=KST)
            + pd.Timedelta(minutes=3 * i)
            for i in range(n)
        ],
        name="timestamp",
    )
    closes = [float(c) for c in closes]
    opens = [closes[0]] + closes[:-1]
    return pd.DataFrame(
        {
            "open": opens,
            "high": [max(o, c) * 1.002 for o, c in zip(opens, closes)],
            "low": [min(o, c) * 0.998 for o, c in zip(opens, closes)],
            "close": closes,
            "volume": volumes if volumes is not None else [1000.0] * n,
        },
        index=index,
    )


def test_session_vwap_restarts_each_day():
    """Cumulating across days would anchor today to last week's prices."""
    first = _intraday([100] * 5, day=date(2026, 8, 18))
    second = _intraday([200] * 5, day=WEDNESDAY)
    both = pd.concat([first, second])
    vwap = session_vwap(both)
    assert vwap.iloc[-1] == pytest.approx(200, rel=0.01)
    assert vwap.iloc[4] == pytest.approx(100, rel=0.01)


def _scalper(**kw):
    """A scalper with the cost gate open unless a test is about the cost gate.

    The gate is checked first and is decisive, so leaving it armed would make
    every structural test report "too quiet" instead of the condition it means
    to exercise.
    """
    kw.setdefault("min_edge_mult", 0.0)
    return Scalping(symbol="002990", timeframe="3Min", period=5, **kw)


def test_scalping_enters_on_a_volume_break_above_vwap():
    closes = [100] * 20 + [101, 102, 103, 108]
    volumes = [1000.0] * 23 + [5000.0]
    signal = _scalper().evaluate(_intraday(closes, volumes))
    assert signal.action is Action.ENTER_LONG
    assert "VWAP" in signal.reason
    assert signal.stop_distance > 0


def test_scalping_refuses_a_break_on_ordinary_volume():
    closes = [100] * 20 + [101, 102, 103, 108]
    signal = _scalper().evaluate(_intraday(closes, [1000.0] * 24))
    assert signal.action is Action.HOLD
    assert "volume" in signal.reason


def test_scalping_refuses_a_break_below_vwap():
    """A high made inside a decline is a bounce, not a breakout."""
    closes = list(range(200, 176, -1))          # falling all session
    closes[-1] = closes[-2] + 3                 # a pop off the low
    signal = _scalper().evaluate(_intraday(closes, [1000.0] * 23 + [9000.0]))
    assert signal.action is Action.HOLD
    assert "VWAP" in signal.reason


def test_scalping_refuses_a_stock_too_quiet_to_pay_for_itself():
    """A round trip costs ~0.38%; a 0.05% ATR cannot repay it."""
    closes = [100 + 0.01 * i for i in range(24)]
    scalper = _scalper(min_edge_mult=3.0, round_trip_cost_pct=0.0038)
    signal = scalper.evaluate(_intraday(closes, [1000.0] * 23 + [9000.0]))
    assert signal.action is Action.HOLD
    assert "pay for itself" in signal.reason


def test_scalping_exits_when_it_loses_vwap():
    closes = [100] * 20 + [108, 106, 103, 96]
    position = Position(
        symbol="002990", side=LONG, qty=10, entry_price=108,
        stop_price=104, stop_distance=4,
    )
    signal = _scalper().evaluate(_intraday(closes), position)
    assert signal.action is Action.EXIT
    assert "VWAP" in signal.reason


# ---------------------------------------------------------------------------
# Close-based trading
# ---------------------------------------------------------------------------


def _daily(closes, highs=None, lows=None, volumes=None):
    n = len(closes)
    index = pd.DatetimeIndex(
        [datetime(2026, 1, 1, tzinfo=KST) + pd.Timedelta(days=i) for i in range(n)],
        name="timestamp",
    )
    closes = [float(c) for c in closes]
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs if highs is not None else [c * 1.02 for c in closes],
            "low": lows if lows is not None else [c * 0.98 for c in closes],
            "close": closes,
            "volume": volumes if volumes is not None else [1000.0] * n,
        },
        index=index,
    )


def _rising(n=40):
    return [100 + i for i in range(n)]


def test_close_auction_buys_a_strong_close_in_an_uptrend_on_volume():
    closes = _rising()
    highs = [c * 1.02 for c in closes]
    lows = [c * 0.98 for c in closes]
    highs[-1] = closes[-1] * 1.001          # finishing on its high
    volumes = [1000.0] * (len(closes) - 1) + [3000.0]
    signal = CloseAuction(symbol="460930").evaluate(_daily(closes, highs, lows, volumes))
    assert signal.action is Action.ENTER_LONG
    assert "strong close" in signal.reason


def test_close_auction_refuses_a_close_off_the_low():
    closes = _rising()
    highs = [c * 1.02 for c in closes]
    lows = [c * 0.98 for c in closes]
    lows[-1] = closes[-1] * 0.999           # finishing on its low
    volumes = [1000.0] * (len(closes) - 1) + [3000.0]
    signal = CloseAuction(symbol="460930").evaluate(_daily(closes, highs, lows, volumes))
    assert signal.action is Action.HOLD
    assert "range" in signal.reason


def test_close_auction_refuses_a_strong_close_in_a_downtrend():
    closes = [200 - i for i in range(40)]
    highs = [c * 1.02 for c in closes]
    lows = [c * 0.98 for c in closes]
    highs[-1] = closes[-1] * 1.001
    volumes = [1000.0] * 39 + [3000.0]
    signal = CloseAuction(symbol="460930").evaluate(_daily(closes, highs, lows, volumes))
    assert signal.action is Action.HOLD
    assert "EMA" in signal.reason


def test_close_auction_refuses_a_quiet_close():
    closes = _rising()
    highs = [c * 1.02 for c in closes]
    lows = [c * 0.98 for c in closes]
    highs[-1] = closes[-1] * 1.001
    signal = CloseAuction(symbol="460930").evaluate(_daily(closes, highs, lows))
    assert signal.action is Action.HOLD
    assert "volume" in signal.reason


def test_close_auction_holds_an_open_position_for_the_morning_exit():
    position = Position(
        symbol="460930", side=LONG, qty=10, entry_price=140,
        stop_price=136, stop_distance=4,
    )
    signal = CloseAuction(symbol="460930").evaluate(_daily(_rising()), position)
    assert signal.action is Action.HOLD
    assert "morning exit" in signal.reason


def test_close_auction_warms_up_in_far_less_than_a_year():
    """The reason a three-month-old listing can run this at all."""
    assert CloseAuction(symbol="439960").warmup <= 25


# ---------------------------------------------------------------------------
# The engine honours the clock
# ---------------------------------------------------------------------------


def _engine(workdir):
    import argparse

    import main

    args = argparse.Namespace(config=None, once=True, dry_run=True, live=False, watch=False)
    return main.TradingEngine(main.build_runtime(args, cli_live=False, force_dry_run=True))


def test_the_engine_loads_a_rule_for_every_stock(workdir):
    engine = _engine(workdir)
    for code in engine.rt.strategies:
        rules = engine.session_rules(code)
        assert rules.force_exit_at is not None, f"{code} has no flat-out time"


def test_day_traders_flatten_before_the_closing_auction(workdir):
    """Every day-trade rule must be flat while a continuous book still exists."""
    from market.calendar import CLOSING_AUCTION_START

    engine = _engine(workdir)
    for code in engine.rt.strategies:
        rules = engine.session_rules(code)
        if rules.hold_overnight:
            continue
        assert rules.force_exit_at < CLOSING_AUCTION_START, code


def test_overnight_stocks_enter_while_the_book_is_still_continuous(workdir):
    """15:20 onward is the closing auction: a market order has no book there."""
    from market.calendar import CLOSING_AUCTION_START

    engine = _engine(workdir)
    overnight = [
        c for c in engine.rt.strategies if engine.session_rules(c).hold_overnight
    ]
    assert overnight, "expected some close-based stocks"
    for code in overnight:
        rules = engine.session_rules(code)
        assert rules.entry_until < CLOSING_AUCTION_START, code


def test_a_malformed_window_fails_at_start_up_not_at_1519(workdir, monkeypatch):
    import main
    from settings import load_config as real_load_config

    def broken(path):
        config = real_load_config(path)
        first = next(iter(config["universe"]))
        config["universe"][first]["entry_window"] = ["15:30", "09:00"]
        return config

    monkeypatch.setattr(main, "load_config", broken)
    with pytest.raises(ValueError, match="never open"):
        _engine(workdir)


def test_the_cost_gate_admits_the_stocks_it_is_configured_for(workdir):
    """The five day-trade names move 9-16% a day; none may be barred by costs.

    An earlier version compared costs to a single ATR rather than to the move
    a winning trade keeps, and its threshold sat above the 3-minute ATR of
    four of the five -- a filter that would have silently traded nothing.
    """
    import math

    from settings import load_config
    from strategies import build_strategy

    config = load_config(None)
    risk_cfg = config.get("risk") or {}
    bars_per_session = (6.5 * 60) / 3.0
    measured_daily_atr_pct = {          # from `profile --live`, 2026-08-19
        "009830": 0.0915,
        "002990": 0.1644,
        "093370": 0.1084,
        "006340": 0.1124,
        "073240": 0.0972,
    }

    for code, cfg in (config.get("universe") or {}).items():
        if cfg.get("strategy") != "scalping":
            continue
        strategy = build_strategy(str(code), cfg, risk_cfg)
        daily = measured_daily_atr_pct[str(code)]
        # Volatility scales with the square root of time, so a day's range
        # spread over the session's bars is roughly this per bar.
        per_bar = daily / math.sqrt(bars_per_session)
        assert per_bar > strategy.min_atr_pct, (
            f"{code}: 3-min ATR ~{per_bar:.2%} would be blocked by the "
            f"{strategy.min_atr_pct:.2%} cost gate"
        )


# ---------------------------------------------------------------------------
# The backtest models what the live bot does
# ---------------------------------------------------------------------------


def _flat_day(day: date, closes, volumes=None):
    """One session of 3-minute bars, 09:00 onward."""
    return _intraday(closes, volumes, day=day)


def test_backtest_flattens_a_day_trade_at_its_time():
    """A backtest that holds through the close measures a different strategy."""
    from backtest import Backtester
    from data import BarSet
    from strategies.scalping import Scalping

    # A session long enough to reach 15:10: 09:00 + 3 min x 125 = 15:15.
    n = 130
    closes = [100.0] * 20 + [100 + 0.6 * i for i in range(n - 20)]
    volumes = [1000.0] * 23 + [9000.0] + [1000.0] * (n - 24)
    bars = _flat_day(WEDNESDAY, closes, volumes)

    universe = {
        "002990": {
            "name": "test",
            "market": "KOSPI",
            "min_qty": 1,
            "entry_window": ["09:15", "14:30"],
            "force_exit_at": "15:10",
        }
    }
    config = {
        "risk": {"per_trade_pct": 0.01, "atr_period": 14, "long_only": True},
        "account": {"starting_equity": 10_000_000},
        "backtest": {"fill_delay_bars": 1},
        "universe": universe,
    }
    strategy = Scalping(symbol="002990", timeframe="3Min", period=5, min_edge_mult=0.0)
    result = Backtester(config, apply_drawdown_guard=False).run(
        {"002990": BarSet("002990", "3Min", bars, "synthetic")},
        {"002990": strategy},
        universe,
    )

    assert result.trades, "expected the day trade to open and close"
    for trade in result.trades:
        assert str(trade.exit_time)[11:16] <= "15:20", (
            f"still open into the closing auction at {trade.exit_time}"
        )


def test_backtest_refuses_entries_outside_the_window():
    from backtest import Backtester
    from data import BarSet
    from strategies.scalping import Scalping

    n = 40
    closes = [100.0] * 20 + [100 + 2.0 * i for i in range(n - 20)]
    volumes = [1000.0] * 23 + [9000.0] * (n - 23)
    bars = _flat_day(WEDNESDAY, closes, volumes)      # 09:00 - 10:57

    universe = {
        "002990": {
            "name": "test",
            "market": "KOSPI",
            "min_qty": 1,
            # A window this bot's bars never reach.
            "entry_window": ["13:00", "14:30"],
            "force_exit_at": "15:10",
        }
    }
    config = {
        "risk": {"per_trade_pct": 0.01, "atr_period": 14, "long_only": True},
        "account": {"starting_equity": 10_000_000},
        "backtest": {"fill_delay_bars": 1},
        "universe": universe,
    }
    result = Backtester(config, apply_drawdown_guard=False).run(
        {"002990": BarSet("002990", "3Min", bars, "synthetic")},
        {"002990": Scalping(symbol="002990", timeframe="3Min", period=5, min_edge_mult=0.0)},
        universe,
    )
    assert not result.trades
    assert any("entries open at" in s for s in result.skipped_orders)


def test_a_close_trade_is_bought_at_the_close_and_sold_at_the_next_open():
    """Filling the entry at the next open would fill it at the exit price."""
    from backtest import Backtester
    from data import BarSet
    from strategies.close_auction import CloseAuction

    closes = _rising(45)
    highs = [c * 1.02 for c in closes]
    lows = [c * 0.98 for c in closes]
    volumes = [1000.0] * len(closes)
    # One qualifying day: finishes on its high, on volume.
    highs[40] = closes[40] * 1.001
    volumes[40] = 4000.0
    bars = _daily(closes, highs, lows, volumes)

    universe = {
        "460930": {
            "name": "test",
            "market": "KOSDAQ",
            "min_qty": 1,
            "entry_window": ["15:15", "15:19"],
            "force_exit_at": "09:05",
            "hold_overnight": True,
        }
    }
    config = {
        "risk": {"per_trade_pct": 0.01, "atr_period": 14, "long_only": True},
        "account": {"starting_equity": 10_000_000},
        "backtest": {"fill_delay_bars": 1},
        "universe": universe,
    }
    result = Backtester(config, apply_drawdown_guard=False).run(
        {"460930": BarSet("460930", "1Day", bars, "synthetic")},
        {"460930": CloseAuction(symbol="460930")},
        universe,
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    # Bought on day 40's close, before slippage; sold on day 41's open.
    assert trade.entry_price == pytest.approx(closes[40], rel=0.01)
    assert trade.exit_price == pytest.approx(bars["open"].iloc[41], rel=0.01)
    assert trade.entry_time.date() < trade.exit_time.date()
