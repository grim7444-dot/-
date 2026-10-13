"""confirm_bars: ORB/PullbackBounce require the breakout/bounce level to hold
for an extra bar before entering (2026-09-17, user request).

trades.csv audit (194 live trades): 54% of trades ended in a stop-out, and
40% of those stop-outs closed within 5 minutes of entry -- a false-breakout
signature, not a mis-sized stop (stop distances themselves clustered right
around the configured 1.3-2% tiers). confirm_bars requires the trigger level
(ORB's range high, PullbackBounce's prior-bar high) to still hold on the next
bar too before entering, filtering the one-bar fakeouts that immediately
reverse.
"""

from __future__ import annotations

import pandas as pd

from market.calendar import KST
from strategies.base import Action
from strategies.orb import ORB
from strategies.pullback import PullbackBounce


# ---------------------------------------------------------------------------
# ORB
# ---------------------------------------------------------------------------


def _orb(**overrides) -> ORB:
    params = dict(
        symbol="TEST", timeframe="1Min",
        range_minutes=5, session_open_hour=9, session_open_minute=0,
        volume_lookback=5, volume_mult=1.2,
        trend_ema=5, min_bar_strength=0.35,
        use_bb_filter=False,
        stop_pct=0.013, early_stop_pct=0.02, early_stop_until="09:30",
        arm_pct=0.012, lock_pct=0.025,
    )
    params.update(overrides)
    return ORB(**params)


def _range_then_breakout(n_breakout_bars: int) -> pd.DataFrame:
    """09:00-09:05 opening range (high 10,010), a few quiet bars still
    inside it (enough margin for warmup at confirm_bars up to 2), then
    n_breakout_bars consecutive bars closing well above the range high on
    good volume and bar strength."""
    rows = [dict(open=10_000, high=10_010, low=9_990, close=10_000, volume=1_000) for _ in range(5)]
    rows.extend(
        dict(open=10_000, high=10_008, low=9_995, close=10_005, volume=1_000) for _ in range(4)
    )
    for i in range(n_breakout_bars):
        px = 10_050 + i * 20
        rows.append(dict(open=px - 30, high=px + 10, low=px - 35, close=px, volume=2_000))
    idx = pd.date_range("2026-08-31 09:00", periods=len(rows), freq="1min", tz="Asia/Seoul")
    return pd.DataFrame(rows, index=idx)


def test_confirm_bars_defaults_to_1():
    assert ORB(symbol="TEST").confirm_bars == 1


def test_default_confirm_bars_blocks_a_single_bar_breakout():
    strategy = _orb()  # confirm_bars=1 by default
    window = _range_then_breakout(1)
    signal = strategy.evaluate(window, None)
    assert signal.action is Action.HOLD
    assert "확인 대기" in signal.reason


def test_default_confirm_bars_enters_once_the_breakout_holds_a_second_bar():
    strategy = _orb()
    window = _range_then_breakout(2)
    signal = strategy.evaluate(window, None)
    assert signal.action is Action.ENTER_LONG, signal.reason


def test_confirm_bars_zero_restores_immediate_entry():
    strategy = _orb(confirm_bars=0)
    window = _range_then_breakout(1)
    signal = strategy.evaluate(window, None)
    assert signal.action is Action.ENTER_LONG, signal.reason


def test_confirm_bars_two_needs_three_consecutive_bars_above_range_high():
    strategy = _orb(confirm_bars=2)
    window = _range_then_breakout(2)
    signal = strategy.evaluate(window, None)
    assert signal.action is Action.HOLD
    assert "확인 대기" in signal.reason

    window = _range_then_breakout(3)
    signal = strategy.evaluate(window, None)
    assert signal.action is Action.ENTER_LONG, signal.reason


# ---------------------------------------------------------------------------
# PullbackBounce
# ---------------------------------------------------------------------------


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


def _pullback(**overrides) -> PullbackBounce:
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


_WARM = [10_000 + i * 20 for i in range(15)]
_SWING_HIGH_BAR = [10_500.0]
_GOOD_PULLBACK = [10_370.0, 10_360.0]


def _swing_then_bounce(n_bounce_bars: int) -> pd.DataFrame:
    """A clean uptrend into a swing high, a 2-bar pullback, then
    n_bounce_bars consecutive bars each reclaiming the prior bar's high --
    same fixture PullbackBounce's other entry tests already rely on, just
    with the final reclaim stretched over more than one bar."""
    px = _GOOD_PULLBACK[-1]
    bounce = []
    for _ in range(n_bounce_bars):
        px = px * 1.01  # +1%/bar, comfortably above the 0.2% high-vs-close buffer
        bounce.append(px)
    closes = _WARM + _SWING_HIGH_BAR + _GOOD_PULLBACK + bounce
    volumes = [5_000.0] * len(closes)
    return _bars(closes, volumes)


def test_pullback_confirm_bars_defaults_to_1():
    assert PullbackBounce(symbol="TEST").confirm_bars == 1


def test_default_confirm_bars_blocks_a_single_bar_bounce():
    strategy = _pullback()  # confirm_bars=1 by default
    window = _swing_then_bounce(1)
    signal = strategy.evaluate(window, None)
    assert signal.action is Action.HOLD
    assert "확인 대기" in signal.reason


def test_default_confirm_bars_enters_once_the_bounce_holds_a_second_bar():
    strategy = _pullback()
    window = _swing_then_bounce(2)
    signal = strategy.evaluate(window, None)
    assert signal.action is Action.ENTER_LONG, signal.reason


def test_pullback_confirm_bars_zero_restores_immediate_entry():
    strategy = _pullback(confirm_bars=0)
    window = _swing_then_bounce(1)
    signal = strategy.evaluate(window, None)
    assert signal.action is Action.ENTER_LONG, signal.reason
