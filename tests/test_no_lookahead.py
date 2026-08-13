"""Safety rule 10 — a signal bar can never be its own fill bar.

Three independent angles:

1. every signal-driven fill lands on ``signal_index + 1``;
2. the fill price is the next bar's open (adjusted for slippage), and is not
   the signal bar's close;
3. rewriting future bars leaves past signals untouched — the strategies
   genuinely cannot see forward.
"""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from backtest import Backtester, CostModel
from data import BarSet
from indicators import atr, ema, rolling_max, sma
from strategies.breakout import Breakout
from strategies.mean_reversion import MeanReversion
from strategies.trend_following import TrendFollowing
from tests.conftest import make_bars


@pytest.fixture
def bt_config():
    return {
        "account": {"starting_equity": 100_000.0},
        "risk": {
            "per_trade_pct": 0.01,
            "max_drawdown_pct": 0.10,
            "atr_period": 14,
            "hard_stop_atr_mult": 1.0,
        },
        "costs": {
            "equity_commission_bps": 1.0,
            "equity_slippage_bps": 2.0,
            "crypto_commission_bps": 25.0,
            "crypto_slippage_bps": 5.0,
        },
        "backtest": {"fill_delay_bars": 1},
    }


@pytest.fixture
def spy_bars():
    return make_bars(n=600, start_price=500.0, seed=11, volatility=0.0035)


def run_backtest(config, bars, strategy, asset_cfg=None, apply_guard=False):
    symbol = strategy.symbol
    barsets = {symbol: BarSet(symbol, strategy.timeframe, bars, "synthetic")}
    engine = Backtester(config, apply_drawdown_guard=apply_guard)
    return engine.run(
        barsets,
        {symbol: strategy},
        {symbol: asset_cfg or {"asset_class": "us_equity", "min_qty": 1}},
    )


# --------------------------------------------------------------------------
# 1. The fill index invariant
# --------------------------------------------------------------------------


def test_backtest_produces_trades_at_all(bt_config, spy_bars):
    """Guard against a vacuous suite: the other tests need real fills."""
    result = run_backtest(bt_config, spy_bars, MeanReversion("SPY", period=20, entry_sigma=1.5))
    assert len(result.fills) > 0
    assert any(f.signal_driven for f in result.fills)


def test_every_signal_driven_fill_is_on_the_next_bar(bt_config, spy_bars):
    result = run_backtest(bt_config, spy_bars, MeanReversion("SPY", period=20, entry_sigma=1.5))
    signal_fills = [f for f in result.fills if f.signal_driven]
    assert signal_fills
    for fill in signal_fills:
        assert fill.fill_index == fill.signal_index + 1, (
            f"{fill.symbol} filled on bar {fill.fill_index} "
            f"from a signal on bar {fill.signal_index}"
        )
    assert result.verify_no_lookahead() is True


def test_fill_time_is_strictly_after_signal_time(bt_config, spy_bars):
    result = run_backtest(bt_config, spy_bars, MeanReversion("SPY", period=20, entry_sigma=1.5))
    for fill in result.fills:
        if fill.signal_driven and fill.signal_time is not None:
            assert fill.fill_time > fill.signal_time


def test_fill_price_is_the_next_bar_open_not_the_signal_close(bt_config, spy_bars):
    costs = CostModel.from_config(bt_config)
    result = run_backtest(bt_config, spy_bars, MeanReversion("SPY", period=20, entry_sigma=1.5))

    checked = 0
    for fill in result.fills:
        if not fill.signal_driven:
            continue
        next_open = float(spy_bars.iloc[fill.fill_index]["open"])
        buying = (fill.kind == "entry" and fill.side == "LONG") or (
            fill.kind != "entry" and fill.side == "SHORT"
        )
        expected = costs.apply_slippage(next_open, buying, "us_equity")
        assert fill.price == pytest.approx(expected, rel=1e-9)
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


def test_rewriting_future_bars_leaves_past_signals_unchanged(bt_config, spy_bars):
    """The decisive test: perturb the tail, the head must not move."""
    cut = 400
    strategy = MeanReversion("SPY", period=20, entry_sigma=1.5)

    original = run_backtest(bt_config, spy_bars, strategy)

    tampered_bars = spy_bars.copy()
    # A violent, obvious change to everything from `cut` onwards.
    tampered_bars.iloc[cut:, tampered_bars.columns.get_indexer(["open", "high", "low", "close"])] *= 3.0
    tampered_bars.iloc[cut:, tampered_bars.columns.get_loc("volume")] *= 9.0
    tampered = run_backtest(bt_config, tampered_bars, MeanReversion("SPY", period=20, entry_sigma=1.5))

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
def test_indicators_do_not_use_future_bars(spy_bars, indicator):
    cut = 300
    full = indicator(spy_bars)
    truncated = indicator(spy_bars.iloc[: cut + 1])
    # The value at bar `cut` must be identical whether or not later bars exist.
    assert truncated.iloc[cut] == pytest.approx(full.iloc[cut], rel=1e-12, nan_ok=True)


@pytest.mark.parametrize(
    "strategy_factory",
    [
        lambda: MeanReversion("SPY", period=20, entry_sigma=1.5),
        lambda: Breakout("BTC/USD", period=20, volume_mult=1.5),
        lambda: TrendFollowing("GLD", fast_ema=20, slow_ema=50),
    ],
)
def test_strategy_signal_depends_only_on_the_window(spy_bars, strategy_factory):
    cut = 350
    window = spy_bars.iloc[: cut + 1]
    from_window = strategy_factory().evaluate(window, None)
    # Same window, but produced by slicing a frame that contains future bars.
    from_full = strategy_factory().evaluate(spy_bars.iloc[: cut + 1], None)
    assert from_window.action == from_full.action
    assert from_window.reason == from_full.reason
    assert from_window.bar_index == cut


def test_strategy_never_receives_more_than_the_signal_bar(bt_config, spy_bars):
    """Instrument the strategy and assert the window always ends at bar i."""
    seen: list[int] = []

    class Spy(MeanReversion):
        def evaluate(self, window, position=None):
            seen.append(len(window))
            signal = super().evaluate(window, position)
            assert signal.bar_index == len(window) - 1
            return signal

    result = run_backtest(bt_config, spy_bars, Spy("SPY", period=20, entry_sigma=1.5))
    assert seen
    # The last bar is never evaluated: there would be no bar left to fill on.
    assert max(seen) <= len(spy_bars) - 1
    assert result.verify_no_lookahead()


# --------------------------------------------------------------------------
# 3. Costs are applied and reported (rule 11)
# --------------------------------------------------------------------------


def test_slippage_always_works_against_the_order():
    costs = CostModel(equity_slippage_bps=10.0)
    assert costs.apply_slippage(100.0, buying=True, asset_class="us_equity") > 100.0
    assert costs.apply_slippage(100.0, buying=False, asset_class="us_equity") < 100.0


def test_costs_are_charged_and_reported(bt_config, spy_bars):
    result = run_backtest(bt_config, spy_bars, MeanReversion("SPY", period=20, entry_sigma=1.5))
    assert result.total_commission > 0
    assert result.total_slippage > 0
    from backtest import format_result

    text = format_result(result)
    assert "Cost assumptions" in text
    assert "bps commission" in text
    assert "Slippage cost" in text


def test_crypto_and_equity_use_different_cost_assumptions(bt_config):
    costs = CostModel.from_config(bt_config)
    assert costs.commission_bps("crypto") > costs.commission_bps("us_equity")
    assert costs.slippage_bps("crypto") > costs.slippage_bps("us_equity")


def test_result_summary_flags_synthetic_data(bt_config, spy_bars):
    from backtest import format_result

    result = run_backtest(bt_config, spy_bars, MeanReversion("SPY", period=20, entry_sigma=1.5))
    assert result.synthetic_data is True
    assert "SYNTHETIC DATA" in format_result(result)
