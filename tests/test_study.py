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


def test_the_report_shows_the_result_without_thinly_sampled_stocks():
    """One stock with one lucky event must not carry a pooled average."""
    deep = BounceStats(
        code="DEEP", name="deep", sessions=500,
        events=_stats_with([0.002] * 20).events,
    )
    lucky = BounceStats(
        code="THIN", name="thin", sessions=500,
        events=_stats_with([0.40]).events,
    )
    report = format_bounce_study(
        [deep, lucky], down_days=3, drop_pct=0.2, cost_pct=0.0038, limit_down=False
    )
    assert "Dropping the 1 stock(s)" in report
    assert "1 of 21 events" in report


def test_the_report_warns_about_searching_for_a_good_parameter_set():
    report = format_bounce_study(
        [_stats_with([0.03] * 40)],
        down_days=3, drop_pct=0.2, cost_pct=0.0038, limit_down=False,
    )
    assert "Every parameter set tried raises the bar" in report


# ---------------------------------------------------------------------------
# Out-of-sample code lists
# ---------------------------------------------------------------------------


def test_codes_parse_from_however_they_were_pasted():
    from study import parse_codes

    dotted = "096770.058470.001210.399720"
    assert parse_codes(dotted) == ["096770", "058470", "001210", "399720"]
    assert parse_codes("096770, 058470") == ["096770", "058470"]
    assert parse_codes("096770\n058470  001210") == ["096770", "058470", "001210"]


def test_a_leading_zero_lost_by_a_spreadsheet_is_restored():
    from study import parse_codes

    assert parse_codes("5930,660") == ["005930", "000660"]


def test_codes_run_together_with_no_separator_are_split_by_six():
    from study import parse_codes

    assert parse_codes("096770058470") == ["096770", "058470"]


def test_duplicates_are_dropped_so_one_stock_cannot_vote_twice():
    from study import parse_codes

    assert parse_codes("096770.058470.096770") == ["096770", "058470"]


def test_empty_input_yields_no_codes():
    from study import parse_codes

    assert parse_codes("") == []
    assert parse_codes("   ") == []


# ---------------------------------------------------------------------------
# Independence: same-day events are one observation
# ---------------------------------------------------------------------------


def _event_on(day, ret):
    return BounceEvent(
        code="X",
        trigger_date=pd.Timestamp(day),
        drop_pct=-0.25,
        entry_close=100.0,
        next_open=100.0 * (1 + ret),
        next_close=100.0 * (1 + ret),
    )


def test_events_sharing_a_date_are_counted_once():
    """A market-wide fall is one observation however many stocks it hits."""
    # Twelve stocks, all triggered on the same three crash days.
    stats = [
        BounceStats(
            code=f"S{i}",
            name=f"s{i}",
            sessions=700,
            events=[
                _event_on("2026-01-05", 0.09),
                _event_on("2026-04-07", 0.08),
                _event_on("2026-06-11", -0.02),
            ],
        )
        for i in range(12)
    ]
    report = format_bounce_study(
        stats, down_days=3, drop_pct=0.2, cost_pct=0.0038, limit_down=False
    )
    assert "happened on 3 distinct dates" in report
    assert "largest single day: 12 stocks at once" in report


def test_a_per_event_t_above_two_is_overruled_by_the_clustered_one():
    """Thirty stocks reacting to three market days are three observations.

    Per event this clears t = 2 comfortably. Per date it does not come close,
    and the per-date figure is the honest one -- the thirty stocks did not
    each independently confirm anything.
    """
    stats = [
        BounceStats(
            code=f"S{i}",
            name=f"s{i}",
            sessions=700,
            events=[
                _event_on("2026-01-05", 0.09),
                _event_on("2026-04-07", 0.08),
                _event_on("2026-06-11", -0.10),
            ],
        )
        for i in range(30)
    ]
    report = format_bounce_study(
        stats, down_days=3, drop_pct=0.2, cost_pct=0.0038, limit_down=False
    )
    assert "treating one market-wide fall as many independent" in report
    assert "this is not" in report


def test_a_result_spread_across_many_dates_keeps_its_significance():
    """Some clustering is normal; what matters is that it survives it."""
    import random

    random.seed(7)
    events = []
    for i in range(60):
        # Forty distinct dates, a few of them shared by two stocks.
        day = 1 + (i % 40)
        events.append(
            _event_on(
                f"2026-{1 + day // 29:02d}-{1 + day % 29:02d}",
                0.02 + random.uniform(-0.005, 0.005),
            )
        )
    stats = [BounceStats(code="S", name="s", sessions=700, events=events)]
    report = format_bounce_study(
        stats, down_days=3, drop_pct=0.2, cost_pct=0.0038, limit_down=False
    )
    assert "distinct dates" in report
    assert "It survives date clustering too" in report


def test_a_market_scan_says_its_result_is_an_upper_bound():
    """A ticker list cannot fully undo survivorship bias; say so."""
    report = format_bounce_study(
        [_stats_with([0.02] * 30)],
        down_days=3, drop_pct=0.2, cost_pct=0.0038, limit_down=False,
        survivorship_note=True,
    )
    assert "upper bound" in report


def test_a_large_scan_summarises_instead_of_listing_every_stock():
    stats = [
        BounceStats(code=f"{i:06d}", name="", sessions=700, events=_stats_with([0.02]).events)
        for i in range(200)
    ]
    report = format_bounce_study(
        stats, down_days=3, drop_pct=0.2, cost_pct=0.0038, limit_down=False,
        per_stock=False,
    )
    assert "200 stocks scanned, 200 produced at least one event." in report
    assert "000199" not in report          # no 200-row table


def test_an_unknown_market_name_is_refused():
    from study import market_codes

    with pytest.raises(ValueError, match="unknown market"):
        market_codes("konex")


# ---------------------------------------------------------------------------
# Getting a ticker list when KRX's listing endpoint is down
# ---------------------------------------------------------------------------


class _StockModule:
    """Stand-in for pykrx.stock with selectable failures."""

    def __init__(self, working: set[str], codes, trading_days=None):
        self.working = working
        self.codes = codes
        self.trading_days = trading_days
        self.calls: list[str] = []

    def _answer(self, name, stamp):
        self.calls.append(f"{name}@{stamp}")
        if name not in self.working:
            raise RuntimeError("Expecting value: line 1 column 1 (char 0)")
        if self.trading_days is not None and stamp not in self.trading_days:
            return []
        return list(self.codes)

    def get_market_ticker_list(self, stamp, market=None):
        return self._answer("get_market_ticker_list", stamp)

    def get_market_ohlcv_by_ticker(self, stamp, market=None):
        return pd.DataFrame(
            index=pd.Index(self._answer("get_market_ohlcv_by_ticker", stamp))
        )


def _patched(monkeypatch, module):
    import data as data_module

    monkeypatch.setattr(data_module, "_import_pykrx_stock", lambda: module)


def test_the_ticker_list_falls_back_when_the_listing_endpoint_is_down(monkeypatch):
    """The exact failure seen live: listing 500s, OHLCV works."""
    from datetime import date

    from study import market_codes

    module = _StockModule(working={"get_market_ohlcv_by_ticker"}, codes=["005930", "000660"])
    _patched(monkeypatch, module)

    assert market_codes("kospi", as_of=date(2026, 8, 19))[0] == ["005930", "000660"]
    assert any("get_market_ticker_list" in c for c in module.calls)


def test_a_non_trading_day_makes_it_step_back(monkeypatch):
    from datetime import date

    from study import market_codes

    module = _StockModule(
        working={"get_market_ticker_list"},
        codes=["005930"],
        trading_days={"20260814"},          # the 17th-19th return nothing
    )
    _patched(monkeypatch, module)
    assert market_codes("kospi", as_of=date(2026, 8, 19))[0] == ["005930"]


def test_everything_failing_yields_an_empty_list_not_an_exception(monkeypatch):
    from datetime import date

    from study import market_codes

    _patched(monkeypatch, _StockModule(working=set(), codes=["005930"]))
    assert market_codes("kosdaq", as_of=date(2026, 8, 19))[0] == []


def test_junk_entries_never_reach_the_study(monkeypatch):
    from datetime import date

    from study import market_codes

    module = _StockModule(
        working={"get_market_ticker_list"},
        codes=["005930", "A000660", "", "12345", "005930", "035720"],
    )
    _patched(monkeypatch, module)
    assert market_codes("kospi", as_of=date(2026, 8, 19))[0] == ["005930", "035720"]


def test_the_limit_is_applied_after_deduplication(monkeypatch):
    from datetime import date

    from study import market_codes

    module = _StockModule(
        working={"get_market_ticker_list"}, codes=["005930", "005930", "000660", "035720"]
    )
    _patched(monkeypatch, module)
    assert market_codes("kospi", as_of=date(2026, 8, 19), limit=2)[0] == ["005930", "000660"]


def test_the_broker_supplies_the_list_when_every_krx_endpoint_is_down(monkeypatch):
    """What actually happened: pykrx cannot answer at all, Kiwoom can."""
    from datetime import date

    from study import market_codes

    _patched(monkeypatch, _StockModule(working=set(), codes=[]))

    class _Broker:
        def get_market_codes(self, market):
            assert market == "kosdaq"
            return ["035720", "247540"]

    codes, source = market_codes("kosdaq", as_of=date(2026, 8, 19), broker=_Broker())
    assert codes == ["035720", "247540"]
    assert "CURRENT" in source          # the survivorship caveat gets stronger


def test_the_broker_is_not_consulted_when_pykrx_answers(monkeypatch):
    from datetime import date

    from study import market_codes

    _patched(monkeypatch, _StockModule(working={"get_market_ticker_list"}, codes=["005930"]))

    class _Broker:
        def get_market_codes(self, market):
            raise AssertionError("should not be asked")

    codes, source = market_codes("kospi", as_of=date(2026, 8, 19), broker=_Broker())
    assert codes == ["005930"]
    assert "pykrx" in source


# ---------------------------------------------------------------------------
# The strong-close pattern, held to the same standard
# ---------------------------------------------------------------------------


def test_a_strong_close_in_an_uptrend_on_volume_is_an_event():
    from study import find_strength_events

    closes = [100 + i for i in range(45)]
    highs = [c * 1.03 for c in closes]
    lows = [c * 0.97 for c in closes]
    volumes = [1000.0] * 45
    highs[40] = closes[40] * 1.001          # finishes on its high
    volumes[40] = 4000.0
    bars = _daily(closes)
    bars["high"] = highs
    bars["low"] = lows
    bars["volume"] = volumes

    events = find_strength_events("X", bars)
    assert [str(e.trigger_date)[:10] for e in events] == [str(bars.index[40])[:10]]


def test_a_strong_close_in_a_downtrend_is_not_an_event():
    from study import find_strength_events

    closes = [200 - i for i in range(45)]
    bars = _daily(closes)
    bars["high"] = [c * 1.001 for c in closes]     # every day closes on its high
    bars["low"] = [c * 0.97 for c in closes]
    bars["volume"] = [4000.0] * 45
    assert find_strength_events("X", bars) == []


def test_a_strong_close_on_ordinary_volume_is_not_an_event():
    from study import find_strength_events

    closes = [100 + i for i in range(45)]
    bars = _daily(closes)
    bars["high"] = [c * 1.001 for c in closes]
    bars["low"] = [c * 0.97 for c in closes]
    bars["volume"] = [1000.0] * 45                 # never above its own average
    assert find_strength_events("X", bars) == []


def test_the_strong_close_report_is_titled_for_what_it_measured():
    report = format_bounce_study(
        [_stats_with([0.01] * 30)],
        down_days=3, drop_pct=0.2, cost_pct=0.0038, limit_down=False,
        pattern="strong-close",
    )
    assert "STRONG CLOSE, NEXT SESSION" in report
    assert "top 30% of the day's range" in report


# ---------------------------------------------------------------------------
# The other patterns
# ---------------------------------------------------------------------------


def _with(bars, **columns):
    for name, values in columns.items():
        bars[name] = values
    return bars


def test_a_volume_spike_is_found_regardless_of_direction():
    from study import find_volume_spike_events

    closes = [100.0] * 30
    bars = _with(_daily(closes), volume=[1000.0] * 28 + [9000.0, 1000.0])
    events = find_volume_spike_events("X", bars, volume_mult=5.0)
    assert [str(e.trigger_date)[:10] for e in events] == [str(bars.index[28])[:10]]


def test_a_volume_spike_below_the_multiple_is_not_an_event():
    from study import find_volume_spike_events

    bars = _with(_daily([100.0] * 30), volume=[1000.0] * 28 + [3000.0, 1000.0])
    assert find_volume_spike_events("X", bars, volume_mult=5.0) == []


def test_the_ma_cross_fires_on_the_crossing_session_only():
    from study import find_ma_cross_events

    # Falling long enough for 5 < 20, then rising through it and staying above.
    closes = [200 - i * 2 for i in range(40)] + [125 + i * 4 for i in range(20)]
    events = find_ma_cross_events("X", _daily(closes), fast=5, slow=20)
    assert len(events) == 1, [str(e.trigger_date)[:10] for e in events]


def test_a_new_high_needs_to_beat_the_prior_window_not_itself():
    from study import find_new_high_events

    flat_then_break = [100.0] * 30 + [101.0] + [100.0] * 3
    events = find_new_high_events("X", _daily(flat_then_break), period=20)
    assert len(events) == 1


def test_every_pattern_is_reachable_and_titled():
    from study import PATTERNS

    for name, (finder, title, description) in PATTERNS.items():
        assert callable(finder), name
        assert title.strip(), name
        assert description.strip().endswith("."), name
        report = format_bounce_study(
            [_stats_with([0.01] * 30)],
            down_days=3, drop_pct=0.2, cost_pct=0.0038, limit_down=False,
            pattern=name,
        )
        if name != "drop":
            assert title in report, name


# ---------------------------------------------------------------------------
# Stock selection, checked out of sample
# ---------------------------------------------------------------------------


def _stock(code, early_returns, late_returns, hold=0.0):
    """A stock whose events straddle a split date.

    *hold* is what simply holding it overnight paid per session, the control
    the pattern has to beat.
    """
    events = []
    for i, r in enumerate(early_returns):
        events.append(_event_on(f"2026-01-{1 + i:02d}", r))
    for i, r in enumerate(late_returns):
        events.append(_event_on(f"2026-02-{1 + i:02d}", r))
    days = [pd.Timestamp(f"2026-02-{1 + i:02d}").date() for i in range(20)]
    baseline = pd.Series([hold] * len(days), index=days)
    return BounceStats(
        code=code, name=code, sessions=700, events=events, baseline=baseline
    )


def test_selection_that_does_not_carry_over_is_called_noise():
    """Last period's winners are next period's average -- say so."""
    from study import format_selection_study

    # Five stocks whose early scores differ wildly and whose late scores do not.
    stats = [
        _stock("000001", [0.10] * 6, [0.00] * 6),
        _stock("000002", [0.08] * 6, [0.00] * 6),
        _stock("000003", [0.00] * 6, [0.00] * 6),
        _stock("000004", [-0.08] * 6, [0.00] * 6),
        _stock("000005", [-0.10] * 6, [0.00] * 6),
    ]
    report = format_selection_study(stats, cost_pct=0.0038, top_n=2, min_events=5)
    assert "The ranking was noise" in report


def test_a_pattern_that_only_matches_the_stocks_drift_is_called_out():
    """Picking stocks that went up is not the same as having an edge."""
    from study import format_selection_study

    stats = [
        _stock("000001", [0.10] * 6, [0.05] * 6, hold=0.05),
        _stock("000002", [0.09] * 6, [0.05] * 6, hold=0.05),
        _stock("000003", [-0.05] * 6, [0.00] * 6, hold=0.0),
        _stock("000004", [-0.08] * 6, [0.00] * 6, hold=0.0),
    ]
    report = format_selection_study(stats, cost_pct=0.0038, top_n=2, min_events=5)
    assert "any rule that bought" in report


def test_selection_that_beats_holding_but_lacks_dates_is_still_refused():
    from study import format_selection_study

    stats = [
        _stock("000001", [0.10] * 6, [0.06, -0.04, 0.09, -0.05, 0.08, 0.01], hold=0.0),
        _stock("000002", [0.09] * 6, [0.05, -0.06, 0.07, -0.03, 0.02, 0.04], hold=0.0),
        _stock("000003", [-0.05] * 6, [-0.05] * 6, hold=0.0),
        _stock("000004", [-0.08] * 6, [-0.06] * 6, hold=0.0),
    ]
    report = format_selection_study(stats, cost_pct=0.0038, top_n=2, min_events=5)
    assert "inside what chance" in report


def test_an_edge_smaller_than_costs_is_not_worth_acting_on():
    from study import format_selection_study

    # The chosen stocks beat both the field and holding, but only just.
    stats = [
        _stock("000001", [0.10] * 6, [0.0060] * 6, hold=0.0),
        _stock("000002", [0.09] * 6, [0.0060] * 6, hold=0.0),
        _stock("000003", [-0.05] * 6, [0.0040] * 6, hold=0.0),
        _stock("000004", [-0.08] * 6, [0.0040] * 6, hold=0.0),
    ]
    report = format_selection_study(stats, cost_pct=0.0038, top_n=2, min_events=5)
    assert "Not worth acting on" in report


def test_stocks_with_too_few_events_are_not_ranked():
    """A stock with two lucky events is not a stock with a record."""
    from study import format_selection_study

    stats = [
        _stock("000001", [0.50] * 2, [0.0] * 2),      # spectacular, and ineligible
        _stock("000002", [0.01] * 6, [0.0] * 6),
        _stock("000003", [0.02] * 6, [0.0] * 6),
    ]
    report = format_selection_study(stats, cost_pct=0.0038, top_n=2, min_events=5)
    assert "Eligible         : 2 stocks" in report
    assert "000001" not in report.split("Top 2")[1].split("Out of sample")[0]


def test_no_eligible_stock_says_so_rather_than_ranking_anyway():
    from study import format_selection_study

    stats = [_stock("000001", [0.1] * 2, [0.0] * 2)]
    report = format_selection_study(stats, cost_pct=0.0038, top_n=5, min_events=5)
    assert "nothing" in report and "to select on" in report


def test_no_events_at_all_is_not_an_error():
    from study import format_selection_study

    report = format_selection_study(
        [BounceStats(code="X", name="x", sessions=700)], cost_pct=0.0038
    )
    assert "nothing to rank" in report
