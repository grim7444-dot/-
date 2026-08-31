"""The tiered profit-lock exit shared by ORB and PullbackBounce.

Both strategies manage an open position the same way (see the "identical
tiering" comment in strategies/orb.py): a fixed percent stop, an "armed"
stage that sets no floor yet, and a "locked" stage once the peak gain
clears lock_pct, where the floor never drops below entry+lock_pct and
otherwise trails the peak by peak_trail_pct.

2026-08-27 (user request): a position that has run further -- past
big_win_pct -- used to widen that trail to a looser peak-relative one, so a
real trend was not stopped out by a small pullback peak_trail_pct alone
would have caught.

2026-08-31 (user request): "3%이상 오르면 2%로 아래 떨어질때까지 기다리다
팔기로" -- past big_win_pct the floor is now a FIXED entry-relative
big_win_floor_pct rather than any kind of trail behind the peak. However
much higher the peak climbs afterward, the floor does not follow it up;
only a genuine fall back to that fixed floor triggers the exit.
"""

from __future__ import annotations

import pandas as pd
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


def _window_at(n_bars: int, last_close: float):
    """A warm-enough window whose LAST bar sits at 09:00 + (n_bars-1) minutes.

    make_bars() (tests/conftest.py) starts every series at 09:00 KST on
    1-minute bars, so this is how test_early_stop_pct controls which side
    of early_stop_until the bar being evaluated falls on.
    """
    window = make_bars(n=n_bars, start_price=10_000.0, seed=3, freq="1min", volatility=0.003).copy()
    window.iloc[-1, window.columns.get_loc("close")] = last_close
    window.iloc[-1, window.columns.get_loc("high")] = max(window.iloc[-1]["high"], last_close)
    window.iloc[-1, window.columns.get_loc("low")] = min(window.iloc[-1]["low"], last_close)
    return window


STRATEGIES = [
    pytest.param(lambda: ORB(symbol="TEST", timeframe="1Min"), id="orb"),
    pytest.param(lambda: PullbackBounce(symbol="TEST", timeframe="1Min"), id="pullback_bounce"),
]


@pytest.mark.parametrize("make_strategy", STRATEGIES)
def test_a_big_win_uses_a_fixed_entry_relative_floor(make_strategy):
    """Once past big_win_pct, the floor is entry + big_win_floor_pct -- not
    a trail behind the peak at all."""
    strategy = make_strategy()
    entry = 10_000.0
    peak = entry * (1 + strategy.big_win_pct + 0.01)  # comfortably past big_win_pct
    floor = max(
        entry * (1 + strategy.lock_pct), entry * (1 + strategy.big_win_floor_pct)
    )

    signal = strategy.evaluate(_window(floor + 1.0), _position(entry, peak))
    assert signal.action is Action.HOLD, signal.reason

    signal = strategy.evaluate(_window(floor - 1.0), _position(entry, peak))
    assert signal.action is Action.EXIT
    assert "익절" in signal.reason


@pytest.mark.parametrize("make_strategy", STRATEGIES)
def test_the_fixed_floor_does_not_rise_as_the_peak_climbs_further(make_strategy):
    """The whole point of the 2026-08-31 change: a peak far past big_win_pct
    must not raise the floor -- unlike the old peak-relative trail, the
    position rides all the way back down to the same fixed floor."""
    strategy = make_strategy()
    entry = 10_000.0
    high_peak = entry * (1 + strategy.big_win_pct + 0.05)  # well past the threshold
    floor = max(
        entry * (1 + strategy.lock_pct), entry * (1 + strategy.big_win_floor_pct)
    )

    # Far below the peak, but still above the fixed floor -- must hold.
    signal = strategy.evaluate(_window(floor + 1.0), _position(entry, high_peak))
    assert signal.action is Action.HOLD, signal.reason

    signal = strategy.evaluate(_window(floor - 1.0), _position(entry, high_peak))
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


# ---------------------------------------------------------------------------
# early_stop_pct: a wider hard stop before early_stop_until (2026-08-28, user
# request: 오전 9시장은 낙폭이 커서 정상 stop_pct로는 노이즈에도 자주 걸림)
# ---------------------------------------------------------------------------


#: Minimal warm-up so a window can land inside the 09:00-09:30 early window
#: on 1-minute bars (PullbackBounce's default RSI/MACD/BB filters alone need
#: 35+ bars, which would already be past 09:30 -- irrelevant here anyway
#: since these tests only exercise position management, not entry filters).
EARLY_STOP_STRATEGIES = [
    pytest.param(
        lambda: ORB(
            symbol="TEST", timeframe="1Min", trend_ema=5, volume_lookback=5,
            stop_pct=0.013, early_stop_pct=0.02, early_stop_until="09:30",
        ),
        id="orb",
    ),
    pytest.param(
        lambda: PullbackBounce(
            symbol="TEST", timeframe="1Min", trend_ema=5, swing_lookback=5,
            use_rsi_filter=False, use_macd_filter=False,
            use_resistance_filter=False, use_bb_filter=False,
            stop_pct=0.013, early_stop_pct=0.02, early_stop_until="09:30",
        ),
        id="pullback_bounce",
    ),
]

ENTRY = 10_000.0
TIGHT_STOP_PRICE = ENTRY * (1 - 0.013)  # 9,870
WIDE_STOP_PRICE = ENTRY * (1 - 0.02)    # 9,800
BETWEEN = (TIGHT_STOP_PRICE + WIDE_STOP_PRICE) / 2  # 9,835


@pytest.mark.parametrize("make_strategy", EARLY_STOP_STRATEGIES)
def test_early_window_uses_the_wider_stop_and_holds_past_the_normal_one(make_strategy):
    """A drop that would stop out the normal 1.3% must NOT exit before 09:30."""
    strategy = make_strategy()
    window = _window_at(25, BETWEEN)  # last bar at 09:24 -- before early_stop_until
    signal = strategy.evaluate(window, _position(ENTRY, ENTRY))
    assert signal.action is Action.HOLD, signal.reason


@pytest.mark.parametrize("make_strategy", EARLY_STOP_STRATEGIES)
def test_early_window_still_exits_once_it_clears_the_wider_stop(make_strategy):
    strategy = make_strategy()
    window = _window_at(25, WIDE_STOP_PRICE - 1.0)
    signal = strategy.evaluate(window, _position(ENTRY, ENTRY))
    assert signal.action is Action.EXIT
    assert "2.0%" in signal.reason


@pytest.mark.parametrize("make_strategy", EARLY_STOP_STRATEGIES)
def test_after_early_stop_until_the_normal_tighter_stop_applies(make_strategy):
    """The same drop that held above must exit once the clock passes 09:30."""
    strategy = make_strategy()
    window = _window_at(35, BETWEEN)  # last bar at 09:34 -- past early_stop_until
    signal = strategy.evaluate(window, _position(ENTRY, ENTRY))
    assert signal.action is Action.EXIT
    assert "1.3%" in signal.reason


# ---------------------------------------------------------------------------
# early_stop_pct also widens whenever price is still above its own trend EMA,
# independent of time of day (2026-08-28, user request: 상승곡선에서는
# 1.3이아니라 2%까지도 주는건 어때 -- a still-intact uptrend deserves the same
# breathing room the early morning gets, and tightens back up the moment the
# trend actually breaks).
# ---------------------------------------------------------------------------


def _window_trend_intact(n_bars: int, flat_level: float, last_close: float):
    """Warm-up padding, then 6 flat bars at `flat_level` (settles a 5-period
    EMA almost exactly there), then a final bar at `last_close`. Lets a test
    control price-vs-trend-EMA precisely instead of relying on make_bars'
    own noise -- verified against indicators.ema() before use."""
    window = make_bars(n=n_bars, start_price=10_000.0, seed=3, freq="1min", volatility=0.003).copy()
    close_col = window.columns.get_loc("close")
    high_col = window.columns.get_loc("high")
    low_col = window.columns.get_loc("low")
    for i in range(1, 7):
        window.iloc[-i, close_col] = flat_level
        window.iloc[-i, high_col] = max(window.iloc[-i]["high"], flat_level)
        window.iloc[-i, low_col] = min(window.iloc[-i]["low"], flat_level)
    window.iloc[-1, close_col] = last_close
    window.iloc[-1, high_col] = max(window.iloc[-1]["high"], last_close)
    window.iloc[-1, low_col] = min(window.iloc[-1]["low"], last_close)
    return window


@pytest.mark.parametrize("make_strategy", EARLY_STOP_STRATEGIES)
def test_trend_intact_past_early_stop_until_still_uses_the_wider_stop(make_strategy):
    """Past 09:30, but price (9,835) is still above its own settled trend
    EMA (9,700) -- the uptrend never broke, so the wide 2% stop must still
    apply and this must HOLD rather than exit on the normal 1.3%."""
    strategy = make_strategy()
    window = _window_trend_intact(35, flat_level=9_700.0, last_close=BETWEEN)
    assert pd.Timestamp(window.index[-1]).time() >= strategy.early_stop_until  # sanity: past the window
    signal = strategy.evaluate(window, _position(ENTRY, ENTRY))
    assert signal.action is Action.HOLD, signal.reason


@pytest.mark.parametrize("make_strategy", EARLY_STOP_STRATEGIES)
def test_trend_broken_past_early_stop_until_still_exits_on_the_tighter_stop(make_strategy):
    """Same clock, but price (9,835) has fallen below its own settled trend
    EMA (10,000) -- the uptrend broke, so this must exit on the normal 1.3%
    (this is the pre-existing behavior; still true with the new OR clause)."""
    strategy = make_strategy()
    window = _window_trend_intact(35, flat_level=10_000.0, last_close=BETWEEN)
    signal = strategy.evaluate(window, _position(ENTRY, ENTRY))
    assert signal.action is Action.EXIT
    assert "1.3%" in signal.reason


# ---------------------------------------------------------------------------
# midday (11:00-15:00) symmetric 1.5%/1.5% override, decided once at entry
# and held for the position's whole life (2026-08-28, user request: 너무
# 거래가 없네. 익절1.5% 손절1.5%로 가자 11시부터3시까지-- faster turnover
# in the middle of the day).
# ---------------------------------------------------------------------------

MIDDAY_STRATEGIES = [
    pytest.param(
        lambda: ORB(
            symbol="TEST", timeframe="1Min", trend_ema=5, volume_lookback=5,
            stop_pct=0.013, arm_pct=0.012, lock_pct=0.025,
            midday_stop_pct=0.015, midday_lock_pct=0.015,
            midday_window_start="11:00", midday_window_end="15:00",
        ),
        id="orb",
    ),
    pytest.param(
        lambda: PullbackBounce(
            symbol="TEST", timeframe="1Min", trend_ema=5, swing_lookback=5,
            use_rsi_filter=False, use_macd_filter=False,
            use_resistance_filter=False, use_bb_filter=False,
            stop_pct=0.013, arm_pct=0.012, lock_pct=0.025,
            midday_stop_pct=0.015, midday_lock_pct=0.015,
            midday_window_start="11:00", midday_window_end="15:00",
        ),
        id="pullback_bounce",
    ),
]

MIDDAY_ENTRY = "2026-08-28T03:00:00+00:00"   # 12:00 KST -- inside the window
MORNING_ENTRY = "2026-08-28T00:40:00+00:00"  # 09:40 KST -- before the window


def _position_entered_at(entry_price: float, highest_price: float, entry_time: str) -> Position:
    return Position(
        symbol="TEST", side=LONG, qty=1,
        entry_price=entry_price, stop_price=entry_price * 0.98, stop_distance=entry_price * 0.02,
        highest_price=highest_price, entry_time=entry_time,
    )


@pytest.mark.parametrize("make_strategy", MIDDAY_STRATEGIES)
def test_midday_entry_holds_past_the_normal_1_3pct_stop(make_strategy):
    """-1.4% is past the normal 1.3% stop but inside midday's wider 1.5%."""
    strategy = make_strategy()
    position = _position_entered_at(ENTRY, ENTRY, MIDDAY_ENTRY)
    signal = strategy.evaluate(_window(ENTRY * (1 - 0.014)), position)
    assert signal.action is Action.HOLD, signal.reason


@pytest.mark.parametrize("make_strategy", MIDDAY_STRATEGIES)
def test_midday_entry_exits_once_it_clears_the_1_5pct_stop(make_strategy):
    strategy = make_strategy()
    position = _position_entered_at(ENTRY, ENTRY, MIDDAY_ENTRY)
    signal = strategy.evaluate(_window(ENTRY * (1 - 0.016)), position)
    assert signal.action is Action.EXIT
    assert "1.5%" in signal.reason


@pytest.mark.parametrize("make_strategy", MIDDAY_STRATEGIES)
def test_a_morning_entry_still_uses_the_normal_1_3pct_stop(make_strategy):
    """Same -1.4% drop, but this position was opened before 11:00 -- the
    wall clock moving into the midday window later must not retroactively
    widen a stop that was never meant to apply to it. Uses a window with a
    settled flat trend EMA (10,000) above the current price, so trend_intact
    is False here and the plain 1.3% stop is the only thing in play."""
    strategy = make_strategy()
    position = _position_entered_at(ENTRY, ENTRY, MORNING_ENTRY)
    window = _window_trend_intact(35, flat_level=10_000.0, last_close=ENTRY * (1 - 0.014))
    signal = strategy.evaluate(window, position)
    assert signal.action is Action.EXIT
    assert "1.3%" in signal.reason


@pytest.mark.parametrize("make_strategy", MIDDAY_STRATEGIES)
def test_midday_entry_locks_in_at_1_5pct_instead_of_the_normal_2_5pct(make_strategy):
    """A 1.6% peak is past midday's 1.5% lock (floor set, trailing begins)
    but still short of the normal 2.5% lock (would just be "armed")."""
    strategy = make_strategy()
    peak = ENTRY * 1.016
    position = _position_entered_at(ENTRY, peak, MIDDAY_ENTRY)
    floor = ENTRY * 1.015  # midday lock floor dominates the loose 0.5% trail here

    below_floor = strategy.evaluate(_window(floor - 1.0), position)
    assert below_floor.action is Action.EXIT
    assert "확정 익절" in below_floor.reason

    above_floor = strategy.evaluate(_window(floor + 1.0), position)
    assert above_floor.action is Action.HOLD
    assert "익절 대기" in above_floor.reason


@pytest.mark.parametrize("make_strategy", MIDDAY_STRATEGIES)
def test_a_morning_entry_is_only_armed_not_locked_at_the_same_1_6pct_peak(make_strategy):
    strategy = make_strategy()
    peak = ENTRY * 1.016
    position = _position_entered_at(ENTRY, peak, MORNING_ENTRY)
    signal = strategy.evaluate(_window(peak), position)
    assert signal.action is Action.HOLD
    assert "무장" in signal.reason
