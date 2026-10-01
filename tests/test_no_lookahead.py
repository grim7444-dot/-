"""Safety rule 10 - a signal bar can never be its own fill bar.

Three independent angles:

1. every signal-driven fill lands on ``signal_index + 1``;
2. the fill price is the next bar's open (adjusted for slippage), and is not
   the signal bar's close;
3. rewriting future bars leaves past signals untouched - the strategies
   genuinely cannot see forward.

Rule 11 is covered at the bottom: costs are charged and reported, and the
Korean sell-side tax shows up in the totals.
"""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from backtest import Backtester
from data import BarSet
from indicators import atr, ema, rolling_max, sma
from market.rules import KOSDAQ, KOSPI, TradingCosts
from strategies.breakout import Breakout
from strategies.mean_reversion import MeanReversion
from strategies.trend_following import TrendFollowing
from tests.conftest import make_bars


@pytest.fixture
def bt_config():
    return {
        "account": {"starting_equity": 10_000_000.0},
        "risk": {
            "per_trade_pct": 0.01,
            "max_total_risk_pct": 0.06,
            "max_open_positions": 6,
            "max_drawdown_pct": 0.10,
            "atr_period": 14,
            "hard_stop_atr_mult": 1.0,
            "long_only": True,
        },
        "costs": {
            "commission_bps": 1.5,
            "sell_tax_bps": {"KOSPI": 15.0, "KOSDAQ": 15.0},
            "slippage_bps": 10.0,
        },
        "backtest": {"fill_delay_bars": 1, "respect_price_limits": False},
    }


@pytest.fixture
def stock_bars():
    return make_bars(n=600, start_price=30_000.0, seed=11, volatility=0.008)


def run_backtest(config, bars, strategy, market=KOSPI, apply_guard=False):
    code = strategy.symbol
    barsets = {code: BarSet(code, strategy.timeframe, bars, "synthetic", market)}
    engine = Backtester(config, apply_drawdown_guard=apply_guard)
    return engine.run(barsets, {code: strategy}, {code: {"min_qty": 1, "market": market}})


# --------------------------------------------------------------------------
# 1. The fill index invariant
# --------------------------------------------------------------------------


def test_backtest_produces_trades_at_all(bt_config, stock_bars):
    """Guard against a vacuous suite: the other tests need real fills."""
    result = run_backtest(
        bt_config, stock_bars, MeanReversion("009830", period=20, entry_sigma=1.5)
    )
    assert len(result.fills) > 0
    assert any(f.signal_driven for f in result.fills)


def test_every_signal_driven_fill_is_on_the_next_bar(bt_config, stock_bars):
    result = run_backtest(
        bt_config, stock_bars, MeanReversion("009830", period=20, entry_sigma=1.5)
    )
    signal_fills = [f for f in result.fills if f.signal_driven]
    assert signal_fills
    for fill in signal_fills:
        assert fill.fill_index == fill.signal_index + 1, (
            f"{fill.code} filled on bar {fill.fill_index} "
            f"from a signal on bar {fill.signal_index}"
        )
    assert result.verify_no_lookahead() is True


def test_fill_time_is_strictly_after_signal_time(bt_config, stock_bars):
    result = run_backtest(
        bt_config, stock_bars, MeanReversion("009830", period=20, entry_sigma=1.5)
    )
    for fill in result.fills:
        if fill.signal_driven and fill.signal_time is not None:
            assert fill.fill_time > fill.signal_time


def test_fill_price_is_the_next_bar_open_not_the_signal_close(bt_config, stock_bars):
    costs = TradingCosts.from_config(bt_config)
    result = run_backtest(
        bt_config, stock_bars, MeanReversion("009830", period=20, entry_sigma=1.5)
    )

    checked = 0
    for fill in result.fills:
        if not fill.signal_driven:
            continue
        next_open = float(stock_bars.iloc[fill.fill_index]["open"])
        buying = (fill.kind == "entry" and fill.side == "LONG") or (
            fill.kind != "entry" and fill.side == "SHORT"
        )
        assert fill.price == pytest.approx(costs.apply_slippage(next_open, buying), rel=1e-9)
        # And it must not have been booked at the signal bar's close.
        assert fill.price != pytest.approx(fill.signal_close, rel=1e-12)
        checked += 1
    assert checked > 0


def test_same_bar_fills_are_rejected_by_construction(bt_config):
    config = copy.deepcopy(bt_config)
    config["backtest"]["fill_delay_bars"] = 0
    with pytest.raises(ValueError, match="look-ahead"):
        Backtester(config)


# --------------------------------------------------------------------------
# 2. Strategies cannot see the future
# --------------------------------------------------------------------------


def test_rewriting_future_bars_leaves_past_signals_unchanged(bt_config, stock_bars):
    """The decisive test: perturb the tail, the head must not move."""
    cut = 400
    original = run_backtest(
        bt_config, stock_bars, MeanReversion("009830", period=20, entry_sigma=1.5)
    )

    tampered_bars = stock_bars.copy()
    price_cols = tampered_bars.columns.get_indexer(["open", "high", "low", "close"])
    tampered_bars.iloc[cut:, price_cols] *= 3.0
    tampered_bars.iloc[cut:, tampered_bars.columns.get_loc("volume")] *= 9.0
    tampered = run_backtest(
        bt_config, tampered_bars, MeanReversion("009830", period=20, entry_sigma=1.5)
    )

    def early(result):
        return [
            (f.signal_index, f.kind, f.side, round(f.price, 8))
            for f in result.fills
            if f.signal_driven and f.fill_index < cut
        ]

    assert early(original), "no early fills to compare"
    assert early(original) == early(tampered)


@pytest.mark.parametrize(
    "indicator",
    [
        lambda df: sma(df["close"], 20),
        lambda df: ema(df["close"], 50),
        lambda df: atr(df, 14),
        lambda df: rolling_max(df["high"], 20),
    ],
)
def test_indicators_do_not_use_future_bars(stock_bars, indicator):
    cut = 300
    full = indicator(stock_bars)
    truncated = indicator(stock_bars.iloc[: cut + 1])
    # The value at bar `cut` must be identical whether or not later bars exist.
    assert truncated.iloc[cut] == pytest.approx(full.iloc[cut], rel=1e-12, nan_ok=True)


@pytest.mark.parametrize(
    "strategy_factory",
    [
        lambda: MeanReversion("009830", period=20, entry_sigma=1.5),
        lambda: Breakout("439960", period=20, volume_mult=1.5),
        lambda: TrendFollowing("002990", fast_ema=20, slow_ema=50),
    ],
)
def test_strategy_signal_depends_only_on_the_window(stock_bars, strategy_factory):
    cut = 350
    from_window = strategy_factory().evaluate(stock_bars.iloc[: cut + 1].copy(), None)
    from_full = strategy_factory().evaluate(stock_bars.iloc[: cut + 1], None)
    assert from_window.action == from_full.action
    assert from_window.reason == from_full.reason
    assert from_window.bar_index == cut


def test_strategy_never_receives_more_than_the_signal_bar(bt_config, stock_bars):
    """Instrument the strategy and assert the window always ends at bar i."""
    seen: list[int] = []

    class Spy(MeanReversion):
        def evaluate(self, window, position=None):
            seen.append(len(window))
            signal = super().evaluate(window, position)
            assert signal.bar_index == len(window) - 1
            return signal

    result = run_backtest(bt_config, stock_bars, Spy("009830", period=20, entry_sigma=1.5))
    assert seen
    # The last bar is never evaluated: there would be no bar left to fill on.
    assert max(seen) <= len(stock_bars) - 1
    assert result.verify_no_lookahead()


# --------------------------------------------------------------------------
# 3. Long-only and costs (rules 11 and the KRX shorting constraint)
# --------------------------------------------------------------------------


def test_long_only_backtest_never_opens_a_short(bt_config, stock_bars):
    result = run_backtest(
        bt_config, stock_bars, MeanReversion("009830", period=20, entry_sigma=1.5)
    )
    entries = [f for f in result.fills if f.kind == "entry"]
    assert entries
    assert all(f.side == "LONG" for f in entries)


def test_costs_are_charged_and_reported(bt_config, stock_bars):
    result = run_backtest(
        bt_config, stock_bars, MeanReversion("009830", period=20, entry_sigma=1.5)
    )
    assert result.total_costs > 0
    assert result.total_slippage > 0

    from backtest import format_result

    text = format_result(result)
    assert "Cost assumptions" in text
    assert "sell tax" in text
    assert "Slippage cost" in text


def test_selling_costs_more_than_buying_in_the_backtest(bt_config, stock_bars):
    """The sell-side tax must actually reach the ledger, not just the config."""
    result = run_backtest(
        bt_config, stock_bars, MeanReversion("009830", period=20, entry_sigma=1.5)
    )
    entry_cost_bps = [
        f.costs / (f.price * f.qty) * 10_000 for f in result.fills if f.kind == "entry"
    ]
    exit_cost_bps = [
        f.costs / (f.price * f.qty) * 10_000 for f in result.fills if f.kind != "entry"
    ]
    assert entry_cost_bps and exit_cost_bps
    assert min(exit_cost_bps) > max(entry_cost_bps)


def test_result_summary_flags_synthetic_data(bt_config, stock_bars):
    from backtest import format_result

    result = run_backtest(
        bt_config, stock_bars, MeanReversion("009830", period=20, entry_sigma=1.5)
    )
    assert result.synthetic_data is True
    assert "SYNTHETIC DATA" in format_result(result)


# --------------------------------------------------------------------------
# `format_result`'s trade-by-trade breakdown (2026-08-31, user request:
# "정말 심각하다.방법을 강구해봐") -- the per-stock P&L rollup can't answer
# "why did this stock lose ten times in a row", so a small trade list is
# only useful, never noise, at low counts.
# --------------------------------------------------------------------------


def _make_result(n_trades: int):
    from backtest import BacktestResult, ClosedTrade

    base = pd.Timestamp("2026-05-07 09:10", tz="Asia/Seoul")
    trades = [
        ClosedTrade(
            code="900300", side="LONG", qty=10,
            entry_time=base + pd.Timedelta(days=i),
            exit_time=base + pd.Timedelta(days=i, minutes=30),
            entry_price=10_000.0, exit_price=9_800.0,
            gross_pnl=-2_000.0, costs=50.0, pnl=-2_050.0, exit_reason="stop",
        )
        for i in range(n_trades)
    ]
    return BacktestResult(
        start=None, end=None, starting_equity=550_000.0, ending_equity=500_000.0,
        trades=trades,
    )


def test_trade_list_appears_for_a_small_number_of_trades():
    from backtest import format_result

    text = format_result(_make_result(10))
    assert "-- Trades --" in text
    assert text.count("900300") >= 10  # header/per-code line + one per trade
    assert "[stop]" in text


def test_trade_list_is_omitted_past_the_readable_threshold():
    from backtest import format_result

    text = format_result(_make_result(41))
    assert "-- Trades --" not in text


def test_kosdaq_and_kospi_can_carry_different_tax(bt_config):
    costs = TradingCosts.from_config(
        {"costs": {"commission_bps": 1.0, "sell_tax_bps": {"KOSPI": 10.0, "KOSDAQ": 20.0}}}
    )
    assert costs.sell_cost_bps(KOSDAQ) > costs.sell_cost_bps(KOSPI)
