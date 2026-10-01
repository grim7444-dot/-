"""BounceReversal (반등매매) -- buy a confirmed technical bounce off a sharp
intraday crash, from a shared blog article's own spec (2026-09-16, user
request): down >= crash_pct from the previous close, already
bounce_confirm_pct off its own post-crash low, on volume_mult volume,
closing strong, above a short support EMA. Exit is a flat take_profit_pct
target -- the hard stop is enforced by the engine itself, not here.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from market.calendar import KST
from portfolio import LONG, Position
from strategies.base import Action
from strategies.bounce import BounceReversal

DAY = date(2026, 8, 19)
PREV_DAY = DAY - timedelta(days=1)


def _session(closes, volumes=None, day=DAY, start_hour=9, bar_minutes=5):
    n = len(closes)
    index = pd.DatetimeIndex(
        [
            datetime(day.year, day.month, day.day, start_hour, 0, tzinfo=KST)
            + pd.Timedelta(minutes=bar_minutes * i)
            for i in range(n)
        ],
        name="timestamp",
    )
    closes = [float(c) for c in closes]
    opens = [closes[0]] + closes[:-1]
    return pd.DataFrame(
        {
            "open": opens,
            "high": [max(o, c) * 1.001 for o, c in zip(opens, closes)],
            "low": [min(o, c) * 0.999 for o, c in zip(opens, closes)],
            "close": closes,
            "volume": volumes if volumes is not None else [1000.0] * n,
        },
        index=index,
    )


def _window(today_closes, today_volumes=None, prev_close=10_000.0):
    """A prior session (a few flat bars ending at prev_close) plus today."""
    prior = _session([prev_close] * 3, day=PREV_DAY)
    today = _session(today_closes, volumes=today_volumes, day=DAY)
    return pd.concat([prior, today])


#: 16 bars declining 9,500 -> 9,100 -- the crash_pct check reads the CURRENT
#: (already-bounced) price against the prior close, not the low itself, so
#: the low has to leave enough room for a confirmed bounce (>=1.5% off it)
#: to still land at or past a 7% drop from the 10,000 prior close (<= 9,300).
_DECLINE_END = 9_100.0
_DECLINE = [9_500.0 - i * ((9_500.0 - _DECLINE_END) / 15.0) for i in range(16)]


def _crash_then_bounce(final_volume=3000.0, final_close=9_260.0, n_bounce=6):
    """The decline above, then *n_bounce* bars rising to *final_close* on a
    volume spike on the last bar only."""
    bounce = [
        _DECLINE_END + i * ((final_close - _DECLINE_END) / n_bounce)
        for i in range(1, n_bounce + 1)
    ]
    closes = _DECLINE + bounce
    volumes = [1000.0] * (len(closes) - 1) + [final_volume]
    return _window(closes, today_volumes=volumes)


def _bouncer(**kw):
    return BounceReversal(symbol="TEST", **kw)


def test_enters_on_a_confirmed_bounce_with_volume_and_strength():
    signal = _bouncer().evaluate(_crash_then_bounce())
    assert signal.action is Action.ENTER_LONG
    assert "급락 반등" in signal.reason


def test_refuses_a_mild_decline_short_of_the_crash_threshold():
    """Only a ~3% dip -- nowhere near the 7% crash bar."""
    closes = [9_900.0 - i * 5.0 for i in range(16)] + [9_800.0 + i * 5.0 for i in range(1, 7)]
    window = _window(closes, today_volumes=[1000.0] * 21 + [3000.0])
    signal = _bouncer().evaluate(window)
    assert signal.action is Action.HOLD
    assert "급락 기준" in signal.reason


def test_refuses_a_crash_still_at_its_low_with_no_bounce_yet():
    """Down 7%+, but the last bar is still the low itself -- nothing to
    confirm a turn yet (this is exactly the "falling knife" case)."""
    decline = [9_500.0 - i * 20.0 for i in range(22)]  # keeps falling to the end
    window = _window(decline, today_volumes=[1000.0] * 22)
    signal = _bouncer().evaluate(window)
    assert signal.action is Action.HOLD
    assert "반등 확인 기준" in signal.reason


def test_refuses_a_bounce_without_a_volume_spike():
    window = _crash_then_bounce(final_volume=1000.0)  # no spike, same as every other bar
    signal = _bouncer().evaluate(window)
    assert signal.action is Action.HOLD
    assert "거래량 부족" in signal.reason


def test_refuses_a_bounce_bar_that_closes_weak():
    """5 bounce bars up to 9,260, then a 6th bar that opens at 9,260 and
    closes lower at 9,230 -- still clears crash_pct/bounce_pct/support_ema
    (verified below) and has its volume spike, but sellers controlled this
    bar, not the buyers a bounce needs."""
    bounce5 = [
        _DECLINE_END + i * ((9_260.0 - _DECLINE_END) / 5.0) for i in range(1, 6)
    ]
    closes = _DECLINE + bounce5 + [9_230.0]
    volumes = [1000.0] * (len(closes) - 1) + [3000.0]
    window = _window(closes, today_volumes=volumes)
    signal = _bouncer().evaluate(window)
    assert signal.action is Action.HOLD
    assert "약세 마감" in signal.reason


def _crash_spike_then_fade():
    """A sharp single-bar spike off the crash low (clears bounce_confirm_pct
    comfortably) followed by a multi-bar fade -- the fade keeps the lagging
    5-bar EMA elevated above where price has since settled, so this isolates
    the support_ema filter specifically (bounce_pct and crash_pct both still
    pass on their own)."""
    spike = [9_400.0]
    fade = [9_370.0, 9_340.0, 9_310.0, 9_280.0, 9_260.0, 9_240.0]
    closes = _DECLINE + spike + fade
    volumes = [1000.0] * (len(closes) - 1) + [3000.0]
    return _window(closes, today_volumes=volumes)


def test_refuses_a_bounce_below_its_own_support_ema():
    signal = _bouncer().evaluate(_crash_spike_then_fade())
    assert signal.action is Action.HOLD
    assert "이평선" in signal.reason


def test_support_ema_filter_off_ignores_it():
    """The fade's own final bar is mildly bearish (part of what keeps the
    EMA elevated above it), so min_bar_strength is also relaxed here --
    this test isolates the support_ema toggle, not bar strength."""
    signal = _bouncer(support_ema=0, min_bar_strength=0).evaluate(_crash_spike_then_fade())
    assert signal.action is Action.ENTER_LONG


def test_holds_an_open_position_short_of_its_target():
    position = Position(
        symbol="TEST", side=LONG, qty=10, entry_price=9_260.0,
        stop_price=9_075.0, stop_distance=185.0, take_profit=9_630.0,
    )
    signal = _bouncer().evaluate(_crash_then_bounce(), position)
    assert signal.action is Action.HOLD
    assert "반등 대기 중" in signal.reason


def test_exits_once_the_fixed_target_is_reached():
    position = Position(
        symbol="TEST", side=LONG, qty=10, entry_price=9_260.0,
        stop_price=9_075.0, stop_distance=185.0, take_profit=9_400.0,
    )
    window = _crash_then_bounce(final_close=9_450.0)  # past the 9,400 target
    signal = _bouncer().evaluate(window, position)
    assert signal.action is Action.EXIT
    assert "반등 목표" in signal.reason


def test_market_down_does_not_block_entry():
    """Unlike close_auction (no stop exists overnight), a bounce trade has
    an active stop the whole time it is held, so a broad down day is not a
    reason to skip it -- market_down is accepted for interface parity and
    deliberately not acted on here."""
    signal = _bouncer().evaluate(_crash_then_bounce(), market_down=True)
    assert signal.action is Action.ENTER_LONG


def test_registered_under_its_own_name():
    from strategies import STRATEGY_CLASSES, BounceReversal as Exported

    assert STRATEGY_CLASSES["bounce"] is Exported
