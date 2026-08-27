"""The tiered profit-lock exit shared by ORB and PullbackBounce.

Both strategies manage an open position the same way (see the "identical
tiering" comment in strategies/orb.py): a fixed percent stop, an "armed"
stage that sets no floor yet, and a "locked" stage once the peak gain
clears lock_pct, where the floor never drops below entry+lock_pct and
otherwise trails the peak by peak_trail_pct.

2026-08-27 (user request): a position that has run further -- past
big_win_pct -- widens that trail to big_win_trail_pct instead of the
tighter peak_trail_pct, so a real trend is not stopped out by a small
pullback that peak_trail_pct alone would have caught.
"""

from __future__ import annotations

import pytest

from portfolio import LONG, Position
from strategies.base import Action
from strategies.orb import ORB
from strategies.pullback import PullbackBounce
from tests.conftest import make_bars


def _window(last_close: float):
    """A warm-enough synthetic window with a specific final close."""
    window = make_bars(n=100, start_price=10_000.0, seed=3, freq="1min", volatility=0.003).copy()
    window.iloc[-1, window.columns.get_loc("close")] = last_close
    window.iloc[-1, window.columns.get_loc("high")] = max(window.iloc[-1]["high"], last_close)
    window.iloc[-1, window.columns.get_loc("low")] = min(window.iloc[-1]["low"], last_close)
    return window


def _position(entry_price: float, highest_price: float) -> Position:
    return Position(
        symbol="TEST", side=LONG, qty=1,
        entry_price=entry_price, stop_price=entry_price * 0.98, stop_distance=entry_price * 0.02,
        highest_price=highest_price,
    )


STRATEGIES = [
    pytest.param(lambda: ORB(symbol="TEST", timeframe="1Min"), id="orb"),
    pytest.param(lambda: PullbackBounce(symbol="TEST", timeframe="1Min"), id="pullback_bounce"),
]


@pytest.mark.parametrize("make_strategy", STRATEGIES)
def test_a_big_win_widens_the_trail_past_where_the_normal_trail_would_exit(make_strategy):
    """The whole point of the feature: this pullback must NOT be an exit."""
    strategy = make_strategy()
    entry = 10_000.0
    peak = entry * (1 + strategy.big_win_pct + 0.01)  # comfortably past big_win_pct
    floor_default_trail = peak * (1.0 - strategy.peak_trail_pct)
    floor_big_win_trail = peak * (1.0 - strategy.big_win_trail_pct)
    assert floor_big_win_trail < floor_default_trail  # sanity: the feature only matters if this holds

    # A price between the two floors would exit under the narrow trail but
    # must hold under the widened one.
    price = (floor_default_trail + floor_big_win_trail) / 2
    signal = strategy.evaluate(_window(price), _position(entry, peak))
    assert signal.action is Action.HOLD, signal.reason


@pytest.mark.parametrize("make_strategy", STRATEGIES)
def test_a_big_win_still_exits_once_it_falls_through_the_wider_floor(make_strategy):
    strategy = make_strategy()
    entry = 10_000.0
    peak = entry * (1 + strategy.big_win_pct + 0.01)
    floor_big_win_trail = max(
        entry * (1 + strategy.lock_pct), peak * (1.0 - strategy.big_win_trail_pct)
    )
    price = floor_big_win_trail - 1.0
    signal = strategy.evaluate(_window(price), _position(entry, peak))
    assert signal.action is Action.EXIT
    assert "익절" in signal.reason


@pytest.mark.parametrize("make_strategy", STRATEGIES)
def test_a_moderate_win_below_big_win_pct_keeps_the_narrow_trail(make_strategy):
    """Below big_win_pct, behavior must be unchanged from before this feature."""
    strategy = make_strategy()
    entry = 10_000.0
    # Past lock_pct but short of big_win_pct.
    peak = entry * (1 + max(strategy.lock_pct + 0.002, strategy.big_win_pct - 0.01))
    assert peak - entry < entry * strategy.big_win_pct  # stays below the big-win threshold
    floor = max(entry * (1 + strategy.lock_pct), peak * (1.0 - strategy.peak_trail_pct))
    signal = strategy.evaluate(_window(floor - 1.0), _position(entry, peak))
    assert signal.action is Action.EXIT
    signal = strategy.evaluate(_window(floor + 1.0), _position(entry, peak))
    assert signal.action is Action.HOLD
