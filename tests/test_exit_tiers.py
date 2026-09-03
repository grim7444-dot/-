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

2026-08-31 (user request): tried a fixed entry-relative floor past
big_win_pct instead of any trail ("3%이상 오르면 2%로 아래 떨어질때까지
기다리다 팔기로"), then corrected to "안되, 고점 대비 하락" -- back to a
peak-relative trail as originally, just wider (big_win_pct 3%->4%,
big_win_trail_pct 1%->2%).
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
def test_the_trail_widens_further_once_a_higher_peak_is_reached(make_strategy):
    """The floor is still peak-relative -- a much higher peak must raise it,
    unlike the fixed-floor approach tried and reverted on 2026-08-31."""
    strategy = make_strategy()
    entry = 10_000.0
    high_peak = entry * (1 + strategy.big_win_pct + 0.05)  # well past the threshold
    floor_at_low_peak = max(
        entry * (1 + strategy.lock_pct),
        (entry * (1 + strategy.big_win_pct + 0.01)) * (1.0 - strategy.big_win_trail_pct),
    )
    floor_at_high_peak = max(
        entry * (1 + strategy.lock_pct), high_peak * (1.0 - strategy.big_win_trail_pct)
    )
    assert floor_at_high_peak > floor_at_low_peak  # the trail follows the higher peak up

    signal = strategy.evaluate(_window(floor_at_high_peak - 1.0), _position(entry, high_peak))
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
# ORB-only: a wider stop for a position that was already up hot_move_pct+
# from today's open at entry, as long as the trend stays intact (2026-09-02,
# user request: "10%이상 급등하고 있는 종목을 추격할때는 3%정도 떨어져도
# 조금더 기다려서 현재 강한종목이라면 조금 더 기다리다가 손절"). Unlike
# early_stop_pct/trend_intact, this is keyed off how far price had already
# run at entry, not the time of day.
# ---------------------------------------------------------------------------


def _hot_orb() -> ORB:
    return ORB(
        symbol="TEST", timeframe="1Min", trend_ema=5, volume_lookback=5,
        stop_pct=0.013, hot_move_pct=0.10, hot_stop_pct=0.03,
    )


HOT_SESSION_OPEN = 10_000.0
HOT_ENTRY = 11_200.0  # +12% off the session open -- clears hot_move_pct (10%)


def test_hot_mover_entry_holds_a_pullback_that_would_stop_out_normally():
    """3% off entry (10,864) is within the wider hot band, but already past
    where the normal 1.3% stop (11,054) would have exited."""
    strategy = _hot_orb()
    window = _window_trend_intact(35, flat_level=10_500.0, last_close=10_870.0)
    position = _position(entry_price=HOT_ENTRY, highest_price=HOT_ENTRY)
    signal = strategy.evaluate(window, position)
    assert signal.action is Action.HOLD, signal.reason


def test_hot_mover_still_exits_once_the_wider_band_itself_is_cleared():
    strategy = _hot_orb()
    stop_price = HOT_ENTRY * (1 - strategy.hot_stop_pct)  # 10,864
    window = _window_trend_intact(35, flat_level=10_500.0, last_close=stop_price - 1.0)
    position = _position(entry_price=HOT_ENTRY, highest_price=HOT_ENTRY)
    signal = strategy.evaluate(window, position)
    assert signal.action is Action.EXIT
    assert "3.0%" in signal.reason


def test_hot_mover_band_stops_applying_once_the_trend_breaks():
    """Same entry and same price as the HOLD case above, but the trend EMA
    has caught up to (settled above) price -- trend_intact is False, so this
    must fall back to the normal tighter stop and exit."""
    strategy = _hot_orb()
    window = _window_trend_intact(35, flat_level=11_200.0, last_close=10_870.0)
    position = _position(entry_price=HOT_ENTRY, highest_price=HOT_ENTRY)
    signal = strategy.evaluate(window, position)
    assert signal.action is Action.EXIT
    assert "1.3%" in signal.reason


def test_an_entry_below_hot_move_pct_never_gets_the_wider_band():
    """Only +5% off the open -- short of the 10% hot_move_pct threshold."""
    strategy = _hot_orb()
    entry = 10_500.0
    stop_price = entry * (1 - strategy.hot_stop_pct)  # would HOLD if hot
    # flat_level well above the close (not just barely) so the trend EMA is
    # unambiguously broken -- see test_trend_broken_past_early_stop_until_*
    # above for the same margin convention.
    window = _window_trend_intact(35, flat_level=11_000.0, last_close=stop_price + 1.0)
    position = _position(entry_price=entry, highest_price=entry)
    signal = strategy.evaluate(window, position)
    # Normal stop_pct (1.3%) is tighter than the hot band and should have
    # already fired at this price.
    assert signal.action is Action.EXIT
    assert "1.3%" in signal.reason


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
