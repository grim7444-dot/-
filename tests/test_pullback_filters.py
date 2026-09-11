"""VWAP support and Fibonacci-retracement entry filters for PullbackBounce.

User-requested (2026-08-27, after reviewing a list of common Korean retail
day-trading techniques): 눌림목에 VWAP 지지선 추가, 눌림목에 피보나치
되돌림 추가.

* VWAP: a bounce priced below the session's volume-weighted average price
  hasn't recovered the "average buyer's" cost basis yet, so it's held back.
* Fibonacci: pullback_depth (existing) measures the dip as a fraction of
  swing_high alone; retracement_pct measures it as a fraction of the whole
  prior swing (swing_low to swing_high) -- the standard Fibonacci
  definition -- and requires it to land in [fib_min, fib_max] (default
  38.2%-50%). Too shallow suggests the trend hasn't really paused; too deep
  suggests it broke.

Both default enabled but gate independently of every other filter
(RSI/MACD/resistance/BB), so each test disables the others to isolate the
one under test.
"""

from __future__ import annotations

import pandas as pd

from market.calendar import KST
from strategies.base import Action
from strategies.pullback import PullbackBounce


def _bars(closes: list[float], volumes: list[float]) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range("2026-08-27 09:00", periods=n, freq="1min", tz=KST)
    closes = [float(c) for c in closes]
    highs = [c * 1.002 for c in closes]
    lows = [c * 0.998 for c in closes]
    opens = [closes[0]] + closes[:-1]
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


#: A clean uptrend into a swing high, a 2-bar pullback that both clears
#: VWAP and lands in the Fibonacci 38.2%-50% zone, then a bounce bar that
#: reclaims the prior bar's high -- every base condition for ENTER_LONG.
_WARM = [10_000 + i * 20 for i in range(15)]
_SWING_HIGH_BAR = [10_500.0]
_GOOD_PULLBACK = [10_370.0, 10_360.0]
_DEEP_PULLBACK = [10_350.0, 10_300.0]  # retraces past the 50% Fibonacci line
_BOUNCE = [10_400.0]

LIGHT_VOLUME = [5_000.0] * 15 + [30_000.0] + [5_000.0, 5_000.0] + [5_000.0]
HEAVY_SWING_VOLUME = [5_000.0] * 15 + [800_000.0] + [5_000.0, 5_000.0] + [5_000.0]


def _strategy(**overrides) -> PullbackBounce:
    params = dict(
        symbol="TEST", timeframe="1Min",
        trend_ema=5, swing_lookback=10, pullback_bars=2, pullback_min_pct=0.005,
        min_bar_strength=0.0,
        use_rsi_filter=False, use_macd_filter=False,
        use_resistance_filter=False, use_bb_filter=False,
        use_vwap_filter=False, use_fib_filter=False,
    )
    params.update(overrides)
    return PullbackBounce(**params)


def test_vwap_filter_blocks_a_bounce_priced_below_vwap():
    strategy = _strategy(use_vwap_filter=True)
    window = _bars(_WARM + _SWING_HIGH_BAR + _GOOD_PULLBACK + _BOUNCE, HEAVY_SWING_VOLUME)
    signal = strategy.evaluate(window, None)
    assert signal.action is Action.HOLD
    assert "VWAP" in signal.reason


def test_vwap_filter_allows_a_bounce_priced_above_vwap():
    strategy = _strategy(use_vwap_filter=True)
    window = _bars(_WARM + _SWING_HIGH_BAR + _GOOD_PULLBACK + _BOUNCE, LIGHT_VOLUME)
    signal = strategy.evaluate(window, None)
    assert signal.action is Action.ENTER_LONG
    assert "VWAP" in signal.reason


def test_vwap_filter_off_ignores_vwap_entirely():
    strategy = _strategy(use_vwap_filter=False)
    window = _bars(_WARM + _SWING_HIGH_BAR + _GOOD_PULLBACK + _BOUNCE, HEAVY_SWING_VOLUME)
    signal = strategy.evaluate(window, None)
    assert signal.action is Action.ENTER_LONG  # same heavy-volume bars that blocked it above


def test_fib_filter_blocks_a_retracement_past_the_configured_zone():
    strategy = _strategy(use_fib_filter=True, fib_min=0.382, fib_max=0.5)
    window = _bars(_WARM + _SWING_HIGH_BAR + _DEEP_PULLBACK + _BOUNCE, LIGHT_VOLUME)
    signal = strategy.evaluate(window, None)
    assert signal.action is Action.HOLD
    assert "피보나치" in signal.reason


def test_fib_filter_allows_a_retracement_inside_the_configured_zone():
    strategy = _strategy(use_fib_filter=True, fib_min=0.382, fib_max=0.5)
    window = _bars(_WARM + _SWING_HIGH_BAR + _GOOD_PULLBACK + _BOUNCE, LIGHT_VOLUME)
    signal = strategy.evaluate(window, None)
    assert signal.action is Action.ENTER_LONG
    assert "피보" in signal.reason


def test_fib_filter_off_ignores_retracement_depth_entirely():
    strategy = _strategy(use_fib_filter=False)
    window = _bars(_WARM + _SWING_HIGH_BAR + _DEEP_PULLBACK + _BOUNCE, LIGHT_VOLUME)
    signal = strategy.evaluate(window, None)
    assert signal.action is Action.ENTER_LONG  # same deep retracement that blocked it above


def test_both_filters_pass_together_on_the_qualifying_pattern():
    strategy = _strategy(use_vwap_filter=True, use_fib_filter=True, fib_min=0.382, fib_max=0.5)
    window = _bars(_WARM + _SWING_HIGH_BAR + _GOOD_PULLBACK + _BOUNCE, LIGHT_VOLUME)
    signal = strategy.evaluate(window, None)
    assert signal.action is Action.ENTER_LONG
    assert "VWAP" in signal.reason and "피보" in signal.reason
