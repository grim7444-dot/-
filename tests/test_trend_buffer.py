"""trend_buffer_pct: ORB/PullbackBounce allow entry a small buffer below the
EMA trend line instead of requiring price to clear it outright (2026-09-18,
user request: "장이 좋은데 너무 매매를 안하네").

Live log audit (2026-09-18 14:29-14:39): almost every HOLD was "EMA 아래 --
상승 추세 아님", and the misses were consistently tiny (0.1-0.9%) -- price
sitting right under a short, noisy EMA rather than in a clear downtrend.
trend_buffer_pct softens just that boundary; every other entry condition is
unchanged.
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
        volume_lookback=5, volume_mult=0.0,
        trend_ema=5, min_bar_strength=0.0,
        use_bb_filter=False, confirm_bars=1,
        stop_pct=0.013,
    )
    params.update(overrides)
    return ORB(**params)


def _orb_window(final_close: float) -> pd.DataFrame:
    """09:00-09:04 opening range (high 10,010), a run-up to 10,080 (pulls
    EMA(5) up above the range), then a breakout bar at 10,060 and a final
    bar at *final_close* -- both above range_high (10,010), satisfying
    confirm_bars=1, with the EMA(5) sitting just above final_close."""
    closes = [10_000] * 5 + [10_050, 10_080, 10_060, final_close]
    rows = [
        dict(open=c, high=c + 10, low=c - 10, close=c, volume=10_000)
        for c in closes
    ]
    idx = pd.date_range("2026-08-31 09:00", periods=len(rows), freq="1min", tz="Asia/Seoul")
    return pd.DataFrame(rows, index=idx)


def test_orb_trend_buffer_defaults_to_0_3_pct():
    assert ORB(symbol="TEST").trend_buffer_pct == 0.003


def test_orb_default_buffer_lets_through_a_close_just_under_the_ema():
    # EMA(5) here lands at ~10,043.46; 10,040 is ~0.03% under it -- within
    # the default 0.3% buffer.
    strategy = _orb()
    signal = strategy.evaluate(_orb_window(10_040), None)
    assert signal.action is Action.ENTER_LONG, signal.reason


def test_orb_still_blocks_a_close_well_under_the_ema():
    # Same setup, but the final bar drops to 9,950 -- ~0.6% under the EMA,
    # outside even the widened buffer.
    strategy = _orb()
    signal = strategy.evaluate(_orb_window(9_950), None)
    assert signal.action is Action.HOLD
    assert "상승 추세 아님" in signal.reason


def test_orb_trend_buffer_zero_restores_the_strict_check():
    strategy = _orb(trend_buffer_pct=0.0)
    signal = strategy.evaluate(_orb_window(10_040), None)
    assert signal.action is Action.HOLD
    assert "상승 추세 아님" in signal.reason


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
        confirm_bars=1,
    )
    params.update(overrides)
    return PullbackBounce(**params)


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


def test_pullback_trend_buffer_defaults_to_0_3_pct():
    assert PullbackBounce(symbol="TEST").trend_buffer_pct == 0.003


#: A steeper uptrend into a 10,800 swing high, an 0.8%-deep pullback, and a
#: two-bar reclaim -- b1 = 10,745.74. The final close is tuned per-test;
#: 10,772.6 verified (via evaluate() directly) to sit ~0.09% under the
#: resulting EMA(5) of ~10,782 -- inside the default 0.3% buffer but not
#: clearing it outright.
_WARM = [10_000 + i * 80 for i in range(15)]
_SWING_HIGH_BAR = [10_800.0]
_PULLBACK = [10_748.16, 10_713.6]
_B1 = 10_745.7408


def _pullback_window(second_bounce_close: float) -> pd.DataFrame:
    closes = _WARM + _SWING_HIGH_BAR + _PULLBACK + [_B1, second_bounce_close]
    volumes = [5_000.0] * len(closes)
    return _bars(closes, volumes)


def test_pullback_default_buffer_lets_through_a_close_just_under_the_ema():
    strategy = _pullback()
    signal = strategy.evaluate(_pullback_window(10_772.6), None)
    assert signal.action is Action.ENTER_LONG, signal.reason


def test_pullback_still_blocks_a_close_well_under_the_ema():
    strategy = _pullback()
    signal = strategy.evaluate(_pullback_window(10_600.0), None)
    assert signal.action is Action.HOLD
    assert "상승 추세 아님" in signal.reason


def test_pullback_trend_buffer_zero_restores_the_strict_check():
    strategy = _pullback(trend_buffer_pct=0.0)
    signal = strategy.evaluate(_pullback_window(10_772.6), None)
    assert signal.action is Action.HOLD
    assert "상승 추세 아님" in signal.reason
