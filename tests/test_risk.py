"""Safety rules 4-7 — sizing, hard stops, kill switch, portfolio caps, pre-trade checks."""

from __future__ import annotations

import math

import pytest

from portfolio import LONG, SHORT, Portfolio, Position
from risk.manager import (
    RiskManager,
    TradeContext,
    evaluate_drawdown,
    hard_stop_price,
    portfolio_capacity,
    position_risk,
    position_size,
    pre_trade_checks,
    theme_block,
)

EQUITY = 10_000_000.0   # 10M KRW
RISK_PCT = 0.01


def _pos(code: str, side: str = LONG, qty: float = 10, entry: float = 20_000.0,
         stop: float | None = None) -> Position:
    distance = entry * 0.05
    return Position(
        symbol=code,
        side=side,
        qty=qty,
        entry_price=entry,
        stop_price=stop if stop is not None else (entry - distance),
        stop_distance=distance,
    )


# --------------------------------------------------------------------------
# Rule 4 — one ATR of adverse movement costs exactly 1% of equity
# --------------------------------------------------------------------------


def test_one_atr_adverse_move_loses_exactly_one_percent():
    result = position_size(
        equity=EQUITY, atr=500.0, price=20_000.0, risk_pct=RISK_PCT,
        fractional=True, min_qty=0.0,
    )
    assert result.ok
    loss_at_one_atr = result.qty_exact * result.stop_distance
    assert loss_at_one_atr == pytest.approx(EQUITY * RISK_PCT, rel=1e-12)


@pytest.mark.parametrize("atr", [10.0, 120.0, 500.0, 3_000.0, 25_000.0])
def test_risk_is_constant_across_volatility(atr):
    result = position_size(
        equity=EQUITY, atr=atr, price=max(atr * 40, 1_000.0), risk_pct=RISK_PCT,
        fractional=True, min_qty=0.0,
    )
    assert result.ok
    assert result.qty_exact * result.stop_distance == pytest.approx(EQUITY * RISK_PCT, rel=1e-12)


def test_quiet_stock_gets_more_size_than_a_violent_one():
    quiet = position_size(EQUITY, atr=100.0, price=20_000.0, fractional=True, min_qty=0.0)
    violent = position_size(EQUITY, atr=2_000.0, price=20_000.0, fractional=True, min_qty=0.0)
    assert quiet.qty > violent.qty
    assert quiet.qty_exact * quiet.stop_distance == pytest.approx(
        violent.qty_exact * violent.stop_distance
    )


def test_krx_shares_are_whole_and_never_exceed_the_budget():
    result = position_size(EQUITY, atr=333.0, price=17_500.0, fractional=False)
    assert result.ok
    assert result.qty == math.floor(result.qty_exact)
    assert result.qty == int(result.qty)  # whole shares only
    assert result.realised_risk() <= EQUITY * RISK_PCT


def test_sizing_scales_with_current_equity():
    small = position_size(5_000_000.0, atr=500.0, price=20_000.0, fractional=True, min_qty=0.0)
    large = position_size(20_000_000.0, atr=500.0, price=20_000.0, fractional=True, min_qty=0.0)
    assert large.qty_exact == pytest.approx(4 * small.qty_exact)


# --------------------------------------------------------------------------
# Rule 4 — a non-positive stop distance skips the order
# --------------------------------------------------------------------------


@pytest.mark.parametrize("atr", [0.0, -1.0, -0.0001, float("nan")])
def test_non_positive_stop_distance_skips_the_order(atr):
    result = position_size(EQUITY, atr=atr, price=20_000.0)
    assert result.skipped is True
    assert result.qty == 0
    assert "stop distance" in result.reason


def test_zero_hard_stop_multiplier_also_skips():
    assert position_size(EQUITY, atr=500.0, price=20_000.0, hard_stop_atr_mult=0.0).skipped


@pytest.mark.parametrize("equity", [0.0, -100.0])
def test_non_positive_equity_skips(equity):
    assert position_size(equity, atr=500.0, price=20_000.0).skipped is True


def test_price_must_be_positive():
    assert position_size(EQUITY, atr=500.0, price=0.0).skipped is True


def test_quantity_below_one_share_is_skipped():
    # A huge stop distance buys less than one whole share.
    result = position_size(EQUITY, atr=200_000.0, price=20_000.0, fractional=False, min_qty=1)
    assert result.skipped is True
    assert result.qty == 0


def test_cash_cap_reduces_size():
    uncapped = position_size(EQUITY, atr=50.0, price=20_000.0, fractional=False)
    capped = position_size(
        EQUITY, atr=50.0, price=20_000.0, fractional=False, available_cash=500_000.0
    )
    assert capped.qty < uncapped.qty
    assert capped.notional <= 500_000.0


def test_notional_cap_reduces_size():
    capped = position_size(
        EQUITY, atr=50.0, price=20_000.0, fractional=False, max_position_notional_pct=0.10
    )
    assert capped.notional <= EQUITY * 0.10


def test_risk_budget_caps_size_below_the_per_trade_amount():
    full = position_size(EQUITY, atr=500.0, price=20_000.0, fractional=True, min_qty=0.0)
    partial = position_size(
        EQUITY, atr=500.0, price=20_000.0, fractional=True, min_qty=0.0,
        risk_budget=EQUITY * 0.004,
    )
    assert partial.qty_exact == pytest.approx(full.qty_exact * 0.4)
    assert partial.risk_amount == pytest.approx(EQUITY * 0.004)


def test_exhausted_risk_budget_skips():
    assert position_size(EQUITY, atr=500.0, price=20_000.0, risk_budget=0.0).skipped is True


# --------------------------------------------------------------------------
# Hard stops on every trade
# --------------------------------------------------------------------------


def test_hard_stop_sits_one_stop_distance_away():
    assert hard_stop_price(LONG, 20_000.0, 500.0) == 19_500.0
    assert hard_stop_price(SHORT, 20_000.0, 500.0) == 20_500.0


def test_hard_stop_refuses_a_non_positive_distance():
    with pytest.raises(ValueError):
        hard_stop_price(LONG, 20_000.0, 0.0)


def test_hard_stop_loss_equals_the_risk_budget():
    sizing = position_size(EQUITY, atr=800.0, price=40_000.0, fractional=True, min_qty=0.0)
    stop = hard_stop_price(LONG, 40_000.0, sizing.stop_distance)
    assert (40_000.0 - stop) * sizing.qty_exact == pytest.approx(EQUITY * RISK_PCT)


# --------------------------------------------------------------------------
# Rules 5/6 — the drawdown kill switch
# --------------------------------------------------------------------------


def test_drawdown_below_the_limit_does_not_trip():
    status = evaluate_drawdown(peak_equity=10_000_000, equity=9_500_000, limit_pct=0.10)
    assert status.drawdown_pct == pytest.approx(0.05)
    assert status.breached is False


@pytest.mark.parametrize("equity", [9_000_000.0, 8_999_000.0, 5_000_000.0])
def test_drawdown_at_or_beyond_the_limit_trips(equity):
    assert evaluate_drawdown(10_000_000, equity, 0.10).breached is True


def test_kill_switch_stops_cancels_and_flattens(portfolio, recording_broker, config):
    manager = RiskManager(config, portfolio)
    portfolio.mark_equity(EQUITY)
    portfolio.open_position(_pos("009830"))

    status = manager.check_drawdown(8_800_000.0)
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
    allowed, reason = manager.can_open_new("009830", LONG)
    assert allowed is False
    assert "STOPPED" in reason and "resume" in reason


def test_stopped_state_survives_a_restart(tmp_path, config):
    """Rule 6: only an explicit resume clears STOPPED."""
    paths = dict(
        state_path=tmp_path / "state.json",
        trades_path=tmp_path / "trades.csv",
        daily_path=tmp_path / "daily_pnl.csv",
    )
    first = Portfolio(**paths)
    first.mark_equity(EQUITY)
    first.stop("drawdown kill switch: test")

    reloaded = Portfolio(**paths)   # a brand new process
    assert reloaded.stopped is True
    assert RiskManager(config, reloaded).can_open_new("009830", LONG)[0] is False

    reloaded.resume()
    assert RiskManager(config, reloaded).can_open_new("009830", LONG)[0] is True
    assert Portfolio(**paths).stopped is False


def test_resume_reanchors_the_drawdown_peak(portfolio):
    portfolio.mark_equity(10_000_000.0)
    portfolio.mark_equity(8_500_000.0)
    assert portfolio.drawdown_pct() == pytest.approx(0.15)
    portfolio.stop("kill switch")
    portfolio.resume()
    assert portfolio.state.peak_equity == pytest.approx(8_500_000.0)
    assert portfolio.drawdown_pct() == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Portfolio caps — 6% total, 6 positions
# --------------------------------------------------------------------------


def test_position_risk_measures_distance_to_the_stop():
    position = _pos("009830", qty=10, entry=20_000.0, stop=19_000.0)
    assert position_risk(position, price=20_000.0) == pytest.approx(10_000.0)


def test_trailing_stop_past_entry_reports_zero_risk():
    position = _pos("009830", qty=10, entry=20_000.0)
    position.trail_stop = 21_000.0   # ratcheted above entry: the trade is free
    assert position_risk(position, price=22_000.0) == pytest.approx(10_000.0)
    assert position_risk(position, price=21_000.0) == pytest.approx(0.0)


def test_capacity_reports_remaining_risk():
    positions = {"009830": _pos("009830", qty=10, entry=20_000.0, stop=19_000.0)}
    cap = portfolio_capacity(positions, EQUITY, max_total_risk_pct=0.06, max_positions=6)
    assert cap.max_risk == pytest.approx(600_000.0)
    assert cap.open_risk == pytest.approx(10_000.0)
    assert cap.remaining_risk == pytest.approx(590_000.0)
    assert cap.ok is True


def test_position_count_cap_blocks_a_seventh_entry(portfolio, config):
    manager = RiskManager(config, portfolio)
    portfolio.mark_equity(EQUITY)
    for code in ("009830", "002990", "093370", "006340", "460930", "101730"):
        portfolio.open_position(_pos(code, qty=1, entry=20_000.0))
    assert len(portfolio.positions()) == 6

    # 439960 is the only stock whose theme (robotics) is untouched, so the
    # count cap is what blocks it rather than the theme filter.
    allowed, reason = manager.can_open_new("439960", LONG, equity=EQUITY)
    assert allowed is False
    assert "portfolio full" in reason


def test_total_risk_cap_blocks_further_entries(portfolio, config):
    manager = RiskManager(config, portfolio)
    portfolio.mark_equity(EQUITY)
    # Two positions, each risking 3% of equity => the 6% cap is exactly used up.
    for code in ("009830", "093370"):
        portfolio.open_position(
            Position(
                symbol=code, side=LONG, qty=300, entry_price=20_000.0,
                stop_price=19_000.0, stop_distance=1_000.0,
            )
        )
    cap = manager.capacity(EQUITY)
    assert cap.open_risk == pytest.approx(600_000.0)
    assert cap.remaining_risk == pytest.approx(0.0)

    allowed, reason = manager.can_open_new("460930", LONG, equity=EQUITY)
    assert allowed is False
    assert "risk cap" in reason


def test_capacity_allows_entry_while_room_remains(portfolio, config):
    manager = RiskManager(config, portfolio)
    portfolio.mark_equity(EQUITY)
    portfolio.open_position(_pos("009830", qty=10, entry=20_000.0))
    assert manager.can_open_new("460930", LONG, equity=EQUITY)[0] is True


# --------------------------------------------------------------------------
# Long-only (futures and short selling are out of scope)
# --------------------------------------------------------------------------


def test_short_entries_are_refused(portfolio, config):
    manager = RiskManager(config, portfolio)
    portfolio.mark_equity(EQUITY)
    allowed, reason = manager.can_open_new("009830", SHORT, equity=EQUITY)
    assert allowed is False
    assert "long-only" in reason


def test_long_entries_are_allowed(portfolio, config):
    manager = RiskManager(config, portfolio)
    portfolio.mark_equity(EQUITY)
    assert manager.can_open_new("009830", LONG, equity=EQUITY)[0] is True


# --------------------------------------------------------------------------
# Theme filter
# --------------------------------------------------------------------------


THEMES = {"construction": ["002990", "228340"], "power_energy": ["006340", "009830"]}


def test_second_stock_in_a_held_theme_is_blocked():
    positions = {"002990": _pos("002990")}
    blocked, reason = theme_block("228340", positions, THEMES)
    assert blocked is True
    assert "002990" in reason and "construction" in reason


def test_different_theme_is_allowed():
    positions = {"002990": _pos("002990")}
    assert theme_block("009830", positions, THEMES)[0] is False


def test_stock_outside_every_theme_is_allowed():
    positions = {"002990": _pos("002990")}
    assert theme_block("005930", positions, THEMES)[0] is False


def test_adding_to_the_same_stock_is_not_a_theme_block():
    """Duplicate positions are the duplicate check's job, not the theme filter's."""
    positions = {"002990": _pos("002990")}
    assert theme_block("002990", positions, THEMES)[0] is False


def test_theme_filter_can_be_disabled():
    positions = {"002990": _pos("002990")}
    assert theme_block("228340", positions, THEMES, enabled=False)[0] is False


def test_manager_wires_the_theme_filter_from_config(portfolio, config):
    manager = RiskManager(config, portfolio)
    portfolio.mark_equity(EQUITY)
    portfolio.open_position(_pos("002990"))
    allowed, reason = manager.can_open_new("228340", LONG, equity=EQUITY)
    assert allowed is False
    assert "theme filter" in reason


# --------------------------------------------------------------------------
# Rule 7 — pre-trade checks
# --------------------------------------------------------------------------


def base_ctx(**overrides) -> TradeContext:
    defaults = dict(
        code="009830",
        side=LONG,
        qty=10,
        price=20_000.0,
        tradable=True,
        session_ok=True,
        min_qty=1,
        open_order_codes=(),
        existing_position=None,
        available_cash=10_000_000.0,
    )
    defaults.update(overrides)
    return TradeContext(**defaults)


def test_all_checks_pass_on_a_clean_order():
    result = pre_trade_checks(base_ctx())
    assert result.passed is True
    assert all(result.checks.values())


def test_untradable_stock_is_rejected():
    result = pre_trade_checks(base_ctx(tradable=False))
    assert result.passed is False
    assert result.checks["tradable"] is False


def test_closed_session_rejects_orders():
    result = pre_trade_checks(
        base_ctx(session_ok=False, session_reason="closing call auction")
    )
    assert result.passed is False
    assert result.checks["session"] is False
    assert "auction" in result.describe()


def test_quantity_below_minimum_is_rejected():
    result = pre_trade_checks(base_ctx(qty=0.5, min_qty=1))
    assert result.passed is False
    assert result.checks["min_qty"] is False


def test_zero_quantity_is_rejected():
    assert pre_trade_checks(base_ctx(qty=0)).passed is False


def test_duplicate_working_order_is_rejected():
    result = pre_trade_checks(base_ctx(open_order_codes=("009830",)))
    assert result.passed is False
    assert result.checks["no_duplicate"] is False


def test_duplicate_position_in_the_same_direction_is_rejected():
    result = pre_trade_checks(base_ctx(existing_position=_pos("009830")))
    assert result.passed is False
    assert result.checks["no_duplicate"] is False


def test_insufficient_cash_is_rejected():
    result = pre_trade_checks(base_ctx(qty=10, price=20_000.0, available_cash=1_000.0))
    assert result.passed is False
    assert result.checks["cash"] is False


def test_exits_are_exempt_from_the_cash_check():
    result = pre_trade_checks(
        base_ctx(side=SHORT, available_cash=0.0, existing_position=_pos("009830"), is_exit=True)
    )
    assert result.passed is True


def test_price_limit_blocks_the_order():
    result = pre_trade_checks(
        base_ctx(at_price_limit=True, price_limit_reason="limit-up: no sellers")
    )
    assert result.passed is False
    assert result.checks["price_limit"] is False
    assert "limit-up" in result.describe()


def test_multiple_failures_are_all_reported():
    result = pre_trade_checks(
        base_ctx(tradable=False, session_ok=False, qty=0, available_cash=0.0, at_price_limit=True)
    )
    assert result.passed is False
    assert len(result.failures) >= 4


# --------------------------------------------------------------------------
# Manager-level integration
# --------------------------------------------------------------------------


def test_manager_reports_loss_limits(portfolio, config):
    manager = RiskManager(config, portfolio)
    assert manager.max_loss_per_trade(EQUITY) == pytest.approx(100_000.0)
    assert manager.max_portfolio_loss(EQUITY) == pytest.approx(600_000.0)


def test_manager_sizes_in_whole_shares(portfolio, config):
    manager = RiskManager(config, portfolio)
    result = manager.size(
        "009830", equity=EQUITY, atr=900.0, price=30_000.0,
        asset_cfg=config["universe"]["009830"],
    )
    assert result.ok
    assert result.qty == int(result.qty)
    assert result.realised_risk() <= EQUITY * RISK_PCT


def test_manager_skips_when_atr_is_unavailable(portfolio, config):
    manager = RiskManager(config, portfolio)
    result = manager.size(
        "009830", equity=EQUITY, atr=0.0, price=30_000.0,
        asset_cfg=config["universe"]["009830"],
    )
    assert result.skipped is True
    assert result.qty == 0


# --------------------------------------------------------------------------
# The broker may only touch the configured universe
# --------------------------------------------------------------------------


def test_kill_switch_leaves_unrelated_holdings_alone(config):
    """A real account held 073240, which this bot never bought and must not sell."""
    from broker import KiwoomBroker
    from settings import Secret, resolve_mode
    from settings import Credentials as Creds

    decision = resolve_mode({}, cli_live=False)
    credentials = Creds(
        app_key=Secret("K" * 43),
        secret_key=Secret("S" * 43),
        account_no=Secret("12345678"),
        telegram_token=Secret(""),
        telegram_chat_id=Secret(""),
        loaded_for="PAPER",
    )
    broker = KiwoomBroker(
        decision, credentials, config, allowed_codes=list(config["universe"])
    )

    from broker import Holding

    sold: list[str] = []
    broker.get_holdings = lambda: {
        "002990": Holding("002990", 21.0, 24_000.0, 25_000.0),
        "073240": Holding("073240", 36.0, 4_500.0, 4_600.0),
    }
    broker.submit_order = lambda code, side, qty, **kw: sold.append(code)

    assert broker.close_all_positions() == 1
    assert sold == ["002990"]        # in the universe
    assert "073240" not in sold      # not ours to sell


def test_broker_refuses_orders_outside_the_universe(config):
    from broker import BrokerError, KiwoomBroker
    from settings import Secret, resolve_mode
    from settings import Credentials as Creds

    decision = resolve_mode({}, cli_live=False)
    credentials = Creds(
        app_key=Secret("K" * 43),
        secret_key=Secret("S" * 43),
        account_no=Secret("12345678"),
        telegram_token=Secret(""),
        telegram_chat_id=Secret(""),
        loaded_for="PAPER",
    )
    broker = KiwoomBroker(
        decision, credentials, config, allowed_codes=list(config["universe"])
    )

    with pytest.raises(BrokerError, match="not in this bot's configured universe"):
        broker.submit_order("073240", LONG, 10)
