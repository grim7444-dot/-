"""ORB's entry conditions -- range breakout, trend, volume, bar strength, and
the optional VWAP confirmation.

2026-08-31 (user request): "돌파매매도 느리고" -- a real breakout often clears
the opening range while the session's cumulative VWAP, pulled up by an
earlier high-volume spike, is still above the current price for a while.
Requiring price to also clear VWAP was delaying entries on genuine breakouts,
not just filtering fake ones, so it is now off by default (use_vwap_filter).
Re-enabling it must still behave exactly as before.
"""

from __future__ import annotations

import pandas as pd

from strategies.base import Action
from strategies.orb import ORB


def _orb(**overrides) -> ORB:
    params = dict(
        symbol="TEST", timeframe="1Min",
        range_minutes=5, session_open_hour=9, session_open_minute=0,
        volume_lookback=5, volume_mult=1.2,
        trend_ema=5, min_bar_strength=0.35,
        use_bb_filter=False,  # isolates the VWAP filter under test in this file
        stop_pct=0.013, early_stop_pct=0.02, early_stop_until="09:30",
        arm_pct=0.012, lock_pct=0.025,
    )
    params.update(overrides)
    return ORB(**params)


def _breakout_window(vwap_spike: bool) -> pd.DataFrame:
    """09:00-09:05 opening range (high 10,010), then an uptrend, then a
    breakout bar closing at 10,070 -- comfortably above the range high, on
    1.5x volume, closing near its own high. When vwap_spike is True, an
    early high-volume bar at a much higher price pulls the session's
    cumulative VWAP up above 10,070; otherwise VWAP settles well below it.
    Verified against indicators/session_vwap directly before use.
    """
    idx = pd.date_range("2026-08-31 09:00", periods=13, freq="1min", tz="Asia/Seoul")
    rows = [dict(open=10_000, high=10_010, low=9_990, close=10_000, volume=1_000) for _ in range(5)]
    if vwap_spike:
        rows.append(dict(open=10_005, high=10_300, low=10_000, close=10_250, volume=20_000))
    else:
        rows.append(dict(open=10_005, high=10_015, low=10_000, close=10_010, volume=1_000))
    rows.extend(
        dict(open=10_010 + i * 5, high=10_020 + i * 5, low=10_005 + i * 5, close=10_015 + i * 5, volume=1_000)
        for i in range(5)
    )
    rows.append(dict(open=10_035, high=10_045, low=10_030, close=10_040, volume=1_000))
    rows.append(dict(open=10_040, high=10_080, low=10_035, close=10_070, volume=1_500))
    return pd.DataFrame(rows, index=idx[: len(rows)])


def test_vwap_filter_is_off_by_default():
    assert ORB(symbol="TEST").use_vwap_filter is False


def test_disabled_vwap_filter_enters_on_a_breakout_below_vwap():
    """The whole point of the change: this must no longer be held back."""
    strategy = _orb()  # use_vwap_filter defaults to False
    window = _breakout_window(vwap_spike=True)
    signal = strategy.evaluate(window, None)
    assert signal.action is Action.ENTER_LONG, signal.reason
    assert "VWAP" not in signal.reason


def test_enabled_vwap_filter_still_blocks_a_breakout_below_vwap():
    """Re-enabling it must restore the exact old behavior."""
    strategy = _orb(use_vwap_filter=True)
    window = _breakout_window(vwap_spike=True)
    signal = strategy.evaluate(window, None)
    assert signal.action is Action.HOLD
    assert "VWAP" in signal.reason


def test_enabled_vwap_filter_allows_a_breakout_above_vwap():
    strategy = _orb(use_vwap_filter=True)
    window = _breakout_window(vwap_spike=False)
    signal = strategy.evaluate(window, None)
    assert signal.action is Action.ENTER_LONG, signal.reason
    assert "VWAP" in signal.reason  # mentioned in the entry note when it was checked


# ---------------------------------------------------------------------------
# Bollinger over-extension filter (2026-08-31, user request): "너무 고점에서
# 매수를 하는것도 문제인듯 ... 불린져밴드수치 확인후 빠르게 진입". A range
# breakout clearing the upper band is a normal breakout signal (PullbackBounce's
# blanket "above the upper band -> block" would reject genuine breakouts), so
# this only rejects a breakout bar that is already far beyond the band it had
# *before* that bar -- shift(1) is essential: measuring against a band that
# includes the breakout bar itself is self-referential and caps the ratio at
# ~1.18x no matter how extreme the bar is (verified by hand before writing
# this), which would make the filter a silent no-op.
# ---------------------------------------------------------------------------


def _quiet_then_breakout_window(breakout_close: float) -> pd.DataFrame:
    """09:00-09:05 opening range (high 10,010), 15 quiet bars settling a tight
    Bollinger band around 10,000, a small 2-bar lead-in, then a breakout bar
    at `breakout_close`. Verified against indicators/bollinger_bands directly:
    the prior (shift(1)) upper band here is ~10,005.6, so +0.9% (10,100)
    stays under the default 3% cutoff and +3.9% (10,400) clears it.
    """
    rows = [dict(open=10_000, high=10_010, low=9_990, close=10_000, volume=1_000) for _ in range(5)]
    px = 10_000
    for n in (1, -1, 2, -2, 1, -1, 2, -2, 1, -1, 2, -2, 1, -1, 2):
        px = 10_000 + n
        rows.append(dict(open=px, high=px + 2, low=px - 2, close=px, volume=1_000))
    rows.append(dict(open=px, high=px + 5, low=px - 2, close=px + 3, volume=1_000))
    rows.append(dict(open=px + 3, high=px + 8, low=px, close=px + 6, volume=1_000))
    last_close = rows[-1]["close"]
    rows.append(dict(
        open=last_close, high=breakout_close + 10, low=last_close - 5,
        close=breakout_close, volume=1_500,
    ))
    idx = pd.date_range("2026-08-31 09:00", periods=len(rows), freq="1min", tz="Asia/Seoul")
    return pd.DataFrame(rows, index=idx)


def test_bb_filter_is_on_by_default():
    assert ORB(symbol="TEST").use_bb_filter is True


def test_enabled_bb_filter_allows_a_modest_breakout():
    strategy = _orb(use_bb_filter=True)
    window = _quiet_then_breakout_window(10_100.0)  # ~+0.9% past the prior upper band
    signal = strategy.evaluate(window, None)
    assert signal.action is Action.ENTER_LONG, signal.reason


def test_enabled_bb_filter_blocks_a_breakout_far_past_the_prior_band():
    strategy = _orb(use_bb_filter=True)
    window = _quiet_then_breakout_window(10_400.0)  # ~+3.9% past the prior upper band
    signal = strategy.evaluate(window, None)
    assert signal.action is Action.HOLD
    assert "과도확장" in signal.reason


def test_disabled_bb_filter_allows_the_same_extended_breakout_through():
    strategy = _orb(use_bb_filter=False)
    window = _quiet_then_breakout_window(10_400.0)
    signal = strategy.evaluate(window, None)
    assert signal.action is Action.ENTER_LONG, signal.reason
