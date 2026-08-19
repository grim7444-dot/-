"""The drop-then-bounce study.

The point of this module is to answer a belief with a measurement, so the
thing that matters most is that it cannot flatter the belief: no event may be
counted whose outcome has not happened yet, costs come off every result, and a
sample too small to mean anything has to say so.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from market.calendar import KST
from study import (
    LIMIT_DOWN_RETURN,
    BounceEvent,
    BounceStats,
    find_bounce_events,
    format_bounce_study,
)


def _daily(closes, opens=None):
    n = len(closes)
    index = pd.DatetimeIndex(
        [datetime(2026, 1, 1, tzinfo=KST) + pd.Timedelta(days=i) for i in range(n)],
        name="timestamp",
    )
    closes = [float(c) for c in closes]
    opens = [float(o) for o in (opens if opens is not None else closes)]
    return pd.DataFrame(
        {
            "open": opens,
            "high": [max(o, c) for o, c in zip(opens, closes)],
            "low": [min(o, c) for o, c in zip(opens, closes)],
            "close": closes,
            "volume": [1000.0] * n,
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# Finding the pattern
# ---------------------------------------------------------------------------


def test_two_down_days_of_the_required_size_are_found():
    # 100 -> 90 (-10%) -> 80 (-11%), combined -20%
    bars = _daily([100, 100, 90, 80, 88])
    events = find_bounce_events("X", bars, down_days=2, drop_pct=0.15)
    assert len(events) == 1
    event = events[0]
    assert event.entry_close == 80
    assert event.next_close == 88
    assert event.to_close_pct == pytest.approx(0.10)


def test_a_drop_that_is_too_shallow_is_not_an_event():
    bars = _daily([100, 100, 98, 96, 99])          # -4% over two days
    assert find_bounce_events("X", bars, down_days=2, drop_pct=0.15) == []


def test_a_drop_interrupted_by_an_up_day_is_not_an_event():
    """Two down days means consecutive, not two down days in a week."""
    bars = _daily([100, 80, 85, 70, 75])
    assert find_bounce_events("X", bars, down_days=2, drop_pct=0.15) == []


def test_the_final_bar_never_produces_an_event():
    """Its outcome has not happened yet, so counting it would be lookahead."""
    bars = _daily([100, 100, 90, 80])              # the drop ends on the last bar
    assert find_bounce_events("X", bars, down_days=2, drop_pct=0.15) == []


def test_limit_down_mode_requires_every_day_at_the_limit():
    ordinary = _daily([100, 100, 88, 78, 85])      # -12%, -11%: big, not limits
    assert find_bounce_events("X", ordinary, down_days=2, limit_down=True) == []

    limits = _daily([100, 100, 70, 49, 60])        # -30%, -30%
    events = find_bounce_events("X", limits, down_days=2, limit_down=True)
    assert len(events) == 1
    assert events[0].drop_pct == pytest.approx(-0.51)


def test_the_limit_threshold_admits_a_close_a_tick_off_the_limit():
    assert LIMIT_DOWN_RETURN > -0.30
    off_by_a_tick = _daily([100, 100, 71, 50.4, 60])
    assert find_bounce_events("X", off_by_a_tick, down_days=2, limit_down=True)


def test_three_day_runs_can_be_asked_for():
    bars = _daily([100, 100, 92, 84, 76, 84])
    assert find_bounce_events("X", bars, down_days=3, drop_pct=0.20)
    assert not find_bounce_events("X", bars, down_days=3, drop_pct=0.30)


def test_a_history_shorter_than_the_pattern_yields_nothing():
    assert find_bounce_events("X", _daily([100, 90]), down_days=2) == []


# ---------------------------------------------------------------------------
# Costs and reporting
# ---------------------------------------------------------------------------


def _stats_with(returns_to_open, cost=0.0038):
    events = [
        BounceEvent(
            code="X",
            trigger_date=pd.Timestamp("2026-01-01"),
            drop_pct=-0.2,
            entry_close=100.0,
            next_open=100.0 * (1 + r),
            next_close=100.0 * (1 + r),
        )
        for r in returns_to_open
    ]
    return BounceStats(code="X", name="test", sessions=500, events=events)


def test_costs_are_subtracted_from_every_result():
    stats = _stats_with([0.01, 0.01])
    summary = stats.to_open(cost_pct=0.0038)
    assert summary["mean"] == pytest.approx(0.01 - 0.0038)


def test_a_bounce_smaller_than_costs_counts_as_a_loss():
    """+0.3% on a 0.38% round trip is a losing trade, not a winning observation."""
    stats = _stats_with([0.003] * 10)
    summary = stats.to_open(cost_pct=0.0038)
    assert summary["win_rate"] == 0.0
    assert summary["mean"] < 0


def test_a_mean_inside_its_own_error_bar_is_reported_as_noise():
    """Three good outcomes are not evidence, however good they are."""
    report = format_bounce_study(
        [_stats_with([0.05, 0.001, 0.09])],
        down_days=2,
        drop_pct=0.15,
        cost_pct=0.0038,
        limit_down=False,
    )
    assert "inside the range chance produces" in report


def test_a_positive_mean_over_a_negative_median_is_called_a_lottery_ticket():
    """The failure mode this study exists for: outliers carrying the average."""
    returns = [-0.02] * 9 + [0.40]        # nine small losses, one huge win
    report = format_bounce_study(
        [_stats_with(returns)],
        down_days=2,
        drop_pct=0.15,
        cost_pct=0.0038,
        limit_down=False,
    )
    assert "lottery ticket, not an edge" in report
    assert "more than half" in report


def test_a_sub_fifty_win_rate_is_stated_plainly():
    returns = [-0.02] * 6 + [0.20] * 4
    report = format_bounce_study(
        [_stats_with(returns)],
        down_days=2,
        drop_pct=0.15,
        cost_pct=0.0038,
        limit_down=False,
    )
    assert "Fewer than half the trades" in report


def test_the_report_says_so_when_the_pattern_loses_after_costs():
    report = format_bounce_study(
        [_stats_with([0.001] * 30)],
        down_days=2,
        drop_pct=0.15,
        cost_pct=0.0038,
        limit_down=False,
    )
    assert "loses money on both exits" in report
    assert "too few" not in report


def test_the_report_says_when_the_gain_needs_a_full_extra_day():
    """Buy-close-sell-open is a different trade from buy-close-sell-close."""
    events = [
        BounceEvent(
            code="X",
            trigger_date=pd.Timestamp("2026-01-01"),
            drop_pct=-0.2,
            entry_close=100.0,
            next_open=100.5,      # the gap barely pays
            next_close=104.0,     # the session does
        )
        for _ in range(30)
    ]
    report = format_bounce_study(
        [BounceStats(code="X", name="t", sessions=500, events=events)],
        down_days=2,
        drop_pct=0.15,
        cost_pct=0.0038,
        limit_down=False,
    )
    assert "rather than the close-to-open trade it was" in report


def test_limit_down_mode_warns_that_the_entry_may_not_fill():
    report = format_bounce_study(
        [], down_days=2, drop_pct=0.15, cost_pct=0.0038, limit_down=True
    )
    assert "no bid to hit" in report


def test_no_events_is_reported_as_no_answer_rather_than_as_a_negative():
    report = format_bounce_study(
        [BounceStats(code="X", name="t", sessions=500)],
        down_days=2,
        drop_pct=0.15,
        cost_pct=0.0038,
        limit_down=False,
    )
    assert "cannot answer it" in report


def test_synthetic_bars_are_excluded_from_the_study(tmp_path, monkeypatch):
    """Inventing the evidence would be worse than having none."""
    import study
    from data import BarSet

    class _Stub:
        def get_bars(self, code, timeframe, **kw):
            return BarSet(code, timeframe, _daily([100, 100, 90, 80, 95]), "synthetic")

    stats = study.run_bounce_study({"X": {"name": "t"}}, _Stub(), months=6)
    assert stats[0].count == 0
