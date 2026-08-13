"""Safety rules 4-7 — sizing, hard stops, the kill switch and pre-trade checks."""

from __future__ import annotations

import math

import pytest

from portfolio import LONG, SHORT, Portfolio, Position
from risk.manager import (
    RiskManager,
    TradeContext,
    correlation_block,
    evaluate_drawdown,
    hard_stop_price,
    position_size,
    pre_trade_checks,
)

EQUITY = 100_000.0
RISK_PCT = 0.01


# --------------------------------------------------------------------------
# Rule 4 — one ATR of adverse movement costs exactly 1% of equity
# --------------------------------------------------------------------------


def test_one_atr_adverse_move_loses_exactly_one_percent():
    result = position_size(
        equity=EQUITY, atr=2.5, price=400.0, risk_pct=RISK_PCT, fractional=True, min_qty=0.0
    )
    assert result.ok
    # qty_exact is the unrounded size; a 1-ATR move against it is the budget.
    loss_at_one_atr = result.qty_exact * result.stop_distance
    assert loss_at_one_atr == pytest.approx(EQUITY * RISK_PCT, rel=1e-12)


@pytest.mark.parametrize("atr", [0.05, 0.5, 2.5, 17.0, 850.0])
def test_risk_is_constant_across_volatility(atr):
    result = position_size(
        equity=EQUITY, atr=atr, price=max(atr * 40, 10.0), risk_pct=RISK_PCT,
        fractional=True, min_qty=0.0,
    )
    assert result.ok
    assert result.qty_exact * result.stop_distance == pytest.approx(EQUITY * RISK_PCT, rel=1e-12)


def test_quiet_asset_gets_more_size_than_a_violent_one():
    quiet = position_size(EQUITY, atr=0.40, price=100.0, fractional=True, min_qty=0.0)
    violent = position_size(EQUITY, atr=8.00, price=100.0, fractional=True, min_qty=0.0)
    assert quiet.qty > violent.qty
    # ... but both risk the same dollars.
    assert quiet.qty_exact * quiet.stop_distance == pytest.approx(
        violent.qty_exact * violent.stop_distance
    )


def test_rounded_quantity_never_risks_more_than_the_budget():
    result = position_size(EQUITY, atr=3.3, price=250.0, fractional=False)
    assert result.ok
    assert result.qty == math.floor(result.qty_exact)
    assert result.realised_risk() <= EQUITY * RISK_PCT


def test_sizing_scales_with_current_equity():
    small = position_size(50_000.0, atr=2.0, price=100.0, fractional=True, min_qty=0.0)
    large = position_size(200_000.0, atr=2.0, price=100.0, fractional=True, min_qty=0.0)
    assert large.qty_exact == pytest.approx(4 * small.qty_exact)


# --------------------------------------------------------------------------
# Rule 4 — a non-positive stop distance skips the order
# --------------------------------------------------------------------------


@pytest.mark.parametrize("atr", [0.0, -1.0, -0.0001, float("nan")])
def test_non_positive_stop_distance_skips_the_order(atr):
    result = position_size(EQUITY, atr=atr, price=100.0)
    assert result.skipped is True
    assert result.qty == 0
    assert not result.ok
    assert "stop distance" in result.reason


def test_zero_hard_stop_multiplier_also_skips():
    result = position_size(EQUITY, atr=5.0, price=100.0, hard_stop_atr_mult=0.0)
    assert result.skipped is True


@pytest.mark.parametrize("equity", [0.0, -100.0])
def test_non_positive_equity_skips(equity):
    assert position_size(equity, atr=1.0, price=100.0).skipped is True


def test_price_must_be_positive():
    assert position_size(EQUITY, atr=1.0, price=0.0).skipped is True


def test_quantity_below_minimum_is_skipped():
    # A huge stop distance buys less than one whole share.
    result = position_size(EQUITY, atr=5_000.0, price=100.0, fractional=False, min_qty=1)
    assert result.skipped is True
    assert result.qty == 0


def test_cash_cap_reduces_size():
    uncapped = position_size(EQUITY, atr=0.10, price=100.0, fractional=False)
    capped = position_size(EQUITY, atr=0.10, price=100.0, fractional=False, available_cash=5_000.0)
    assert capped.qty < uncapped.qty
    assert capped.notional <= 5_000.0


def test_notional_cap_reduces_size():
    capped = position_size(
        EQUITY, atr=0.10, price=100.0, fractional=False, max_position_notional_pct=0.10
    )
    assert capped.notional <= EQUITY * 0.10


# --------------------------------------------------------------------------
# Hard stops on every trade
# --------------------------------------------------------------------------


def test_hard_stop_sits_one_stop_distance_away():
    assert hard_stop_price(LONG, 100.0, 2.0) == 98.0
    assert hard_stop_price(SHORT, 100.0, 2.0) == 102.0


def test_hard_stop_refuses_a_non_positive_distance():
    with pytest.raises(ValueError):
        hard_stop_price(LONG, 100.0, 0.0)


def test_hard_stop_loss_equals_the_risk_budget():
    sizing = position_size(EQUITY, atr=4.0, price=200.0, fractional=True, min_qty=0.0)
    stop = hard_stop_price(LONG, 200.0, sizing.stop_distance)
    loss = (200.0 - stop) * sizing.qty_exact
    assert loss == pytest.approx(EQUITY * RISK_PCT)


# --------------------------------------------------------------------------
# Rule 5/6 — the drawdown kill switch
# --------------------------------------------------------------------------


def test_drawdown_below_the_limit_does_not_trip():
    status = evaluate_drawdown(peak_equity=100_000, equity=95_000, limit_pct=0.10)
    assert status.drawdown_pct == pytest.approx(0.05)
    assert status.breached is False


@pytest.mark.parametrize("equity", [90_000.0, 89_999.0, 50_000.0])
def test_drawdown_at_or_beyond_the_limit_trips(equity):
    assert evaluate_drawdown(100_000, equity, 0.10).breached is True


def test_kill_switch_stops_cancels_and_flattens(portfolio, recording_broker, config):
    manager = RiskManager(config, portfolio)
    portfolio.mark_equity(100_000.0)
    portfolio.open_position(
        Position(
            symbol="SPY", side=LONG, qty=10, entry_price=500.0,
            stop_price=495.0, stop_distance=5.0, strategy="mean_reversion",
        )
    )

    status = manager.check_drawdown(88_000.0)
    assert status.breached is True

    manager.trip_kill_switch(status, recording_broker)

    assert portfolio.stopped is True
    assert recording_broker.cancelled == 1
    assert recording_broker.closed == 1
    assert portfolio.state.positions == {}
    assert "drawdown" in portfolio.state.stopped_reason


def test_stopped_state_blocks_new_orders(portfolio, config):
    manager = RiskManager(config, portfolio)
    portfolio.stop("drawdown kill switch: test")
    allowed, reason = manager.can_open_new("SPY", LONG)
    assert allowed is False
    assert "STOPPED" in reason
    assert "resume" in reason


def test_stopped_state_survives_a_restart(tmp_path, config):
    """Rule 6: only an explicit resume clears STOPPED."""
    first = Portfolio(
        state_path=tmp_path / "state.json",
        trades_path=tmp_path / "trades.csv",
        daily_path=tmp_path / "daily_pnl.csv",
    )
    first.mark_equity(100_000.0)
    first.stop("drawdown kill switch: test")

    # A brand new process reading the same state file.
    reloaded = Portfolio(
        state_path=tmp_path / "state.json",
        trades_path=tmp_path / "trades.csv",
        daily_path=tmp_path / "daily_pnl.csv",
    )
    assert reloaded.stopped is True
    assert RiskManager(config, reloaded).can_open_new("SPY", LONG)[0] is False

    reloaded.resume()
    assert reloaded.stopped is False
    assert RiskManager(config, reloaded).can_open_new("SPY", LONG)[0] is True

    again = Portfolio(
        state_path=tmp_path / "state.json",
        trades_path=tmp_path / "trades.csv",
        daily_path=tmp_path / "daily_pnl.csv",
    )
    assert again.stopped is False


def test_resume_reanchors_the_drawdown_peak(portfolio):
    portfolio.mark_equity(100_000.0)
    portfolio.mark_equity(85_000.0)
    assert portfolio.drawdown_pct() == pytest.approx(0.15)
    portfolio.stop("kill switch")
    portfolio.resume()
    assert portfolio.state.peak_equity == pytest.approx(85_000.0)
    assert portfolio.drawdown_pct() == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Correlation filter
# --------------------------------------------------------------------------


def _pos(symbol: str, side: str) -> Position:
    return Position(
        symbol=symbol, side=side, qty=1, entry_price=100.0,
        stop_price=99.0, stop_distance=1.0,
    )


def test_btc_long_blocked_when_spy_and_qqq_are_both_long():
    positions = {"SPY": _pos("SPY", LONG), "QQQ": _pos("QQQ", LONG)}
    blocked, reason = correlation_block("BTC/USD", LONG, positions)
    assert blocked is True
    assert "SPY" in reason and "QQQ" in reason


@pytest.mark.parametrize(
    "positions",
    [
        {},
        {"SPY": _pos("SPY", LONG)},
        {"QQQ": _pos("QQQ", LONG)},
        {"SPY": _pos("SPY", LONG), "QQQ": _pos("QQQ", SHORT)},
        {"SPY": _pos("SPY", SHORT), "QQQ": _pos("QQQ", SHORT)},
    ],
)
def test_btc_long_allowed_otherwise(positions):
    assert correlation_block("BTC/USD", LONG, positions)[0] is False


def test_correlation_filter_does_not_block_btc_shorts_or_exits():
    positions = {"SPY": _pos("SPY", LONG), "QQQ": _pos("QQQ", LONG)}
    assert correlation_block("BTC/USD", SHORT, positions)[0] is False
    assert correlation_block("GLD", LONG, positions)[0] is False


def test_correlation_filter_can_be_disabled():
    positions = {"SPY": _pos("SPY", LONG), "QQQ": _pos("QQQ", LONG)}
    assert correlation_block("BTC/USD", LONG, positions, enabled=False)[0] is False


def test_manager_wires_the_filter_from_config(portfolio, config):
    manager = RiskManager(config, portfolio)
    portfolio.open_position(_pos("SPY", LONG))
    portfolio.open_position(_pos("QQQ", LONG))
    allowed, reason = manager.can_open_new("BTC/USD", LONG)
    assert allowed is False
    assert "correlation filter" in reason


# --------------------------------------------------------------------------
# Rule 7 — pre-trade checks
# --------------------------------------------------------------------------


def base_ctx(**overrides) -> TradeContext:
    defaults = dict(
        symbol="SPY",
        side=LONG,
        qty=10,
        price=500.0,
        asset_class="us_equity",
        tradable=True,
        market_open=True,
        min_qty=1,
        open_order_symbols=(),
        existing_position=None,
        available_cash=1_000_000.0,
    )
    defaults.update(overrides)
    return TradeContext(**defaults)


def test_all_checks_pass_on_a_clean_order():
    result = pre_trade_checks(base_ctx())
    assert result.passed is True
    assert all(result.checks.values())


def test_untradable_asset_is_rejected():
    result = pre_trade_checks(base_ctx(tradable=False))
    assert result.passed is False
    assert result.checks["tradable"] is False


def test_closed_market_rejects_equity_orders():
    result = pre_trade_checks(base_ctx(market_open=False))
    assert result.passed is False
    assert result.checks["market_hours"] is False


def test_crypto_trades_when_the_stock_market_is_closed():
    """Rule 8: BTC/USD is 24/7, equity ETFs are not."""
    result = pre_trade_checks(
        base_ctx(symbol="BTC/USD", asset_class="crypto", market_open=False, min_qty=0.0001, qty=0.05)
    )
    assert result.passed is True
    assert result.checks["market_hours"] is True


def test_quantity_below_minimum_is_rejected():
    result = pre_trade_checks(base_ctx(qty=0.5, min_qty=1))
    assert result.passed is False
    assert result.checks["min_qty"] is False


def test_zero_quantity_is_rejected():
    assert pre_trade_checks(base_ctx(qty=0)).passed is False


def test_duplicate_working_order_is_rejected():
    result = pre_trade_checks(base_ctx(open_order_symbols=("SPY",)))
    assert result.passed is False
    assert result.checks["no_duplicate"] is False


def test_duplicate_position_in_the_same_direction_is_rejected():
    result = pre_trade_checks(base_ctx(existing_position=_pos("SPY", LONG)))
    assert result.passed is False
    assert result.checks["no_duplicate"] is False


def test_insufficient_cash_is_rejected():
    result = pre_trade_checks(base_ctx(qty=10, price=500.0, available_cash=1_000.0))
    assert result.passed is False
    assert result.checks["cash"] is False


def test_exits_are_exempt_from_the_cash_check():
    result = pre_trade_checks(
        base_ctx(
            side=SHORT,
            available_cash=0.0,
            existing_position=_pos("SPY", LONG),
            is_exit=True,
        )
    )
    assert result.passed is True


def test_multiple_failures_are_all_reported():
    result = pre_trade_checks(
        base_ctx(tradable=False, market_open=False, qty=0, available_cash=0.0)
    )
    assert result.passed is False
    assert len(result.failures) >= 3


# --------------------------------------------------------------------------
# Manager-level integration
# --------------------------------------------------------------------------


def test_manager_reports_max_loss_per_trade(portfolio, config):
    manager = RiskManager(config, portfolio)
    assert manager.max_loss_per_trade(250_000.0) == pytest.approx(2_500.0)


def test_manager_sizes_crypto_fractionally(portfolio, config):
    manager = RiskManager(config, portfolio)
    result = manager.size(
        "BTC/USD",
        equity=EQUITY,
        atr=1_500.0,
        price=62_000.0,
        asset_cfg=config["assets"]["BTC/USD"],
    )
    assert result.ok
    assert 0 < result.qty < 1  # fractional size, not rounded down to zero
    assert result.realised_risk() <= EQUITY * RISK_PCT


def test_manager_skips_when_atr_is_unavailable(portfolio, config):
    manager = RiskManager(config, portfolio)
    result = manager.size(
        "SPY", equity=EQUITY, atr=0.0, price=500.0, asset_cfg=config["assets"]["SPY"]
    )
    assert result.skipped is True
    assert result.qty == 0
