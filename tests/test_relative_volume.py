"""relative_volume (RVOL) and its use_rvol_filter wiring in ORB/PullbackBounce
(2026-09-23, user request: research day-trading techniques and add a solid
one). A trailing rolling_mean_volume treats every time of day the same, so
an ordinary volume tick during a naturally quiet stretch (lunch, etc.) can
still clear it -- RVOL instead compares a bar to the SAME time-of-day bar on
prior sessions, which several external sources tie to a measurable win-rate
gain over trailing-average volume confirmation alone.
"""

from __future__ import annotations

import pandas as pd

from indicators import relative_volume
from market.calendar import KST
from strategies.base import Action
from strategies.orb import ORB
from strategies.pullback import PullbackBounce


# ---------------------------------------------------------------------------
# indicators.relative_volume
# ---------------------------------------------------------------------------


def test_relative_volume_is_nan_on_the_first_session_present():
    idx = pd.date_range("2026-09-23 09:00", periods=5, freq="1min", tz=KST)
    df = pd.DataFrame(
        {"open": 1, "high": 1, "low": 1, "close": 1, "volume": [100, 200, 300, 400, 500]},
        index=idx,
    )
    result = relative_volume(df)
    assert result.isna().all()


def test_relative_volume_compares_against_the_same_time_of_day_prior_session():
    idx1 = pd.date_range("2026-09-22 09:00", periods=3, freq="1min", tz=KST)
    idx2 = pd.date_range("2026-09-23 09:00", periods=3, freq="1min", tz=KST)
    df = pd.DataFrame(
        {"open": 1, "high": 1, "low": 1, "close": 1, "volume": [100, 200, 300, 150, 999, 450]},
        index=idx1.append(idx2),
    )
    result = relative_volume(df)
    assert result.iloc[:3].isna().all()
    assert result.iloc[3] == 150 / 100
    assert result.iloc[4] == 999 / 200
    assert result.iloc[5] == 450 / 300


# ---------------------------------------------------------------------------
# ORB
# ---------------------------------------------------------------------------


def _orb(**overrides) -> ORB:
    params = dict(
        symbol="TEST", timeframe="1Min",
        range_minutes=5, session_open_hour=9, session_open_minute=0,
        volume_lookback=5, volume_mult=1.2,
        trend_ema=5, min_bar_strength=0.0,
        use_bb_filter=False, confirm_bars=0,
        stop_pct=0.013,
    )
    params.update(overrides)
    return ORB(**params)


def _orb_two_session_breakout(day1_final_volume: float, day2_final_volume: float) -> pd.DataFrame:
    """Two identically-shaped 10-bar sessions (so each bar lines up with
    the same time-of-day bar on the other day) -- day 2 breaks the opening
    range on its final bar. day1_final_volume sets the RVOL baseline for
    that exact time slot; day2_final_volume is what actually breaks out.
    Both sessions' first 9 bars carry volume=1,000, so the trailing
    volume_mult check (avg of the preceding 5 bars = 1,000) only depends on
    day2_final_volume, isolating RVOL as the thing under test.
    """
    day1_rows = [
        dict(open=10_000, high=10_010, low=9_990, close=10_000, volume=1_000) for _ in range(9)
    ]
    day1_rows.append(dict(open=10_000, high=10_010, low=9_995, close=10_005, volume=day1_final_volume))
    idx1 = pd.date_range("2026-09-22 09:00", periods=10, freq="1min", tz="Asia/Seoul")
    day1 = pd.DataFrame(day1_rows, index=idx1)

    day2_rows = [
        dict(open=10_000, high=10_010, low=9_990, close=10_000, volume=1_000) for _ in range(9)
    ]
    day2_rows.append(dict(open=10_020, high=10_060, low=10_015, close=10_050, volume=day2_final_volume))
    idx2 = pd.date_range("2026-09-23 09:00", periods=10, freq="1min", tz="Asia/Seoul")
    day2 = pd.DataFrame(day2_rows, index=idx2)
    return pd.concat([day1, day2])


def test_orb_rvol_filter_defaults_on():
    assert ORB(symbol="TEST").use_rvol_filter is True


def test_orb_rvol_blocks_a_breakout_that_clears_trailing_volume_but_not_rvol():
    strategy = _orb()
    # day2 final bar: 1,500/1,000 = 1.5x trailing (clears volume_mult=1.2x),
    # but day1 had 2,000 at this exact time slot -> RVOL = 1,500/2,000 = 0.75x.
    window = _orb_two_session_breakout(day1_final_volume=2_000.0, day2_final_volume=1_500.0)
    signal = strategy.evaluate(window, None)
    assert signal.action is Action.HOLD
    assert "상대거래량" in signal.reason


def test_orb_rvol_allows_a_breakout_once_both_checks_pass():
    strategy = _orb()
    window = _orb_two_session_breakout(day1_final_volume=500.0, day2_final_volume=1_500.0)
    signal = strategy.evaluate(window, None)
    assert signal.action is Action.ENTER_LONG, signal.reason


def test_orb_rvol_filter_off_ignores_it():
    strategy = _orb(use_rvol_filter=False)
    window = _orb_two_session_breakout(day1_final_volume=2_000.0, day2_final_volume=1_500.0)
    signal = strategy.evaluate(window, None)
    assert signal.action is Action.ENTER_LONG, signal.reason


def test_orb_rvol_with_no_prior_session_fails_open():
    """A single-session window has no RVOL baseline (NaN) -- must not block
    an otherwise-valid breakout just because history is thin."""
    strategy = _orb()
    idx = pd.date_range("2026-09-23 09:00", periods=10, freq="1min", tz="Asia/Seoul")
    rows = [
        dict(open=10_000, high=10_010, low=9_990, close=10_000, volume=1_000) for _ in range(9)
    ]
    rows.append(dict(open=10_020, high=10_060, low=10_015, close=10_050, volume=1_500))
    window = pd.DataFrame(rows, index=idx)
    signal = strategy.evaluate(window, None)
    assert signal.action is Action.ENTER_LONG, signal.reason


# ---------------------------------------------------------------------------
# PullbackBounce
# ---------------------------------------------------------------------------


def _pullback(**overrides) -> PullbackBounce:
    params = dict(
        symbol="TEST", timeframe="1Min",
        trend_ema=5, swing_lookback=10, pullback_bars=2, pullback_min_pct=0.005,
        min_bar_strength=0.0,
        use_rsi_filter=False, use_macd_filter=False,
        use_resistance_filter=False, use_bb_filter=False,
        use_vwap_filter=False, use_fib_filter=False,
        confirm_bars=0,
    )
    params.update(overrides)
    return PullbackBounce(**params)


def _bars(closes: list[float], volumes: list[float], day: str) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range(f"{day} 09:00", periods=n, freq="1min", tz=KST)
    closes = [float(c) for c in closes]
    highs = [c * 1.002 for c in closes]
    lows = [c * 0.998 for c in closes]
    opens = [closes[0]] + closes[:-1]
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


#: Same swing/pullback/bounce shape proven elsewhere (test_trend_buffer.py) to
#: reach ENTER_LONG on its own merits, replayed identically on two sessions.
_WARM = [10_000 + i * 80 for i in range(15)]
_SWING_HIGH_BAR = [10_800.0]
_PULLBACK = [10_748.16, 10_713.6]
_BOUNCE = [10_772.6]


def _pullback_two_session_bounce(day1_final_volume: float, day2_final_volume: float) -> pd.DataFrame:
    closes = _WARM + _SWING_HIGH_BAR + _PULLBACK + _BOUNCE
    n = len(closes)
    day1 = _bars(closes, [5_000.0] * (n - 1) + [day1_final_volume], day="2026-09-22")
    day2 = _bars(closes, [5_000.0] * (n - 1) + [day2_final_volume], day="2026-09-23")
    return pd.concat([day1, day2])


def test_pullback_rvol_filter_defaults_on():
    assert PullbackBounce(symbol="TEST").use_rvol_filter is True


def test_pullback_rvol_blocks_a_bounce_with_weak_relative_volume():
    strategy = _pullback()
    window = _pullback_two_session_bounce(day1_final_volume=20_000.0, day2_final_volume=8_000.0)
    signal = strategy.evaluate(window, None)
    assert signal.action is Action.HOLD
    assert "상대거래량" in signal.reason


def test_pullback_rvol_allows_a_bounce_with_strong_relative_volume():
    strategy = _pullback()
    window = _pullback_two_session_bounce(day1_final_volume=2_000.0, day2_final_volume=8_000.0)
    signal = strategy.evaluate(window, None)
    assert signal.action is Action.ENTER_LONG, signal.reason


def test_pullback_rvol_filter_off_ignores_it():
    strategy = _pullback(use_rvol_filter=False)
    window = _pullback_two_session_bounce(day1_final_volume=20_000.0, day2_final_volume=8_000.0)
    signal = strategy.evaluate(window, None)
    assert signal.action is Action.ENTER_LONG, signal.reason
