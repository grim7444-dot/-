"""KRX market rules - tick sizes, daily price limits and asymmetric costs.

These are the three places where a US-shaped assumption silently breaks on a
Korean venue: an off-grid price is rejected, a limit-locked stock cannot be
traded at all, and the sell side pays tax that the buy side does not.
"""

from __future__ import annotations

import pytest

from market.rules import (
    KOSDAQ,
    KOSPI,
    KrxRules,
    TradingCosts,
    price_limits,
    round_to_tick,
    tick_size,
)


# --------------------------------------------------------------------------
# Tick sizes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "price,expected",
    [
        (500, 1),
        (1_999, 1),
        (2_000, 5),
        (4_999, 5),
        (5_000, 10),
        (19_999, 10),
        (20_000, 50),
        (49_999, 50),
        (50_000, 100),
        (199_999, 100),
        (200_000, 500),
        (499_999, 500),
        (500_000, 1_000),
        (1_250_000, 1_000),
    ],
)
def test_kospi_tick_ladder(price, expected):
    assert tick_size(price, KOSPI) == expected


@pytest.mark.parametrize(
    "price,expected",
    [(1_500, 1), (3_000, 5), (12_000, 10), (30_000, 50), (80_000, 100), (600_000, 100)],
)
def test_kosdaq_tick_ladder(price, expected):
    """KOSDAQ tops out at a 100 KRW tick - it does not have the 500/1000 rungs."""
    assert tick_size(price, KOSDAQ) == expected


def test_tick_ladders_differ_between_markets():
    assert tick_size(300_000, KOSPI) == 500
    assert tick_size(300_000, KOSDAQ) == 100


@pytest.mark.parametrize("price", [1_234, 7_777, 23_456, 187_654, 654_321])
def test_rounding_always_lands_on_the_grid(price):
    for mode in ("nearest", "up", "down"):
        snapped = round_to_tick(price, KOSPI, mode)
        assert snapped % tick_size(snapped, KOSPI) == 0


def test_rounding_directions_bracket_the_input():
    assert round_to_tick(20_030, KOSPI, "down") == 20_000
    assert round_to_tick(20_030, KOSPI, "up") == 20_050
    assert round_to_tick(20_030, KOSPI, "nearest") in (20_000, 20_050)


def test_rounding_never_returns_zero():
    """A price rounded down near zero would be an invalid order."""
    assert round_to_tick(0.4, KOSPI, "down") >= 1


def test_long_stop_rounds_up_so_the_loss_stays_inside_the_budget():
    """Rounding a long stop *down* would risk more than 1% of equity."""
    rules = KrxRules({})
    entry, distance = 20_000.0, 530.0
    stop = rules.stop_price(entry, distance, KOSPI)
    assert stop >= entry - distance          # never further away than budgeted
    assert stop % rules.tick_size(stop, KOSPI) == 0


# --------------------------------------------------------------------------
# Daily price limits
# --------------------------------------------------------------------------


def test_price_limits_are_thirty_percent_either_side():
    limits = price_limits(10_000, KOSPI)
    assert limits.upper == pytest.approx(13_000, abs=50)
    assert limits.lower == pytest.approx(7_000, abs=50)


def test_price_limits_land_on_the_tick_grid():
    limits = price_limits(33_333, KOSPI)
    assert limits.upper % tick_size(limits.upper, KOSPI) == 0
    assert limits.lower % tick_size(limits.lower, KOSPI) == 0


def test_limit_up_blocks_buying_but_not_selling():
    limits = price_limits(10_000, KOSPI)
    assert limits.blocks_buy(limits.upper) is True
    assert limits.blocks_sell(limits.upper) is False


def test_limit_down_blocks_selling_but_not_buying():
    limits = price_limits(10_000, KOSPI)
    assert limits.blocks_sell(limits.lower) is True
    assert limits.blocks_buy(limits.lower) is False


def test_mid_range_price_blocks_nothing():
    limits = price_limits(10_000, KOSPI)
    assert limits.at_limit(10_500) is False
    assert limits.blocks_buy(10_500) is False
    assert limits.blocks_sell(10_500) is False


def test_limit_pct_is_configurable():
    rules = KrxRules({"krx": {"price_limit_pct": 0.15}})
    limits = rules.price_limits(10_000, KOSPI)
    assert limits.upper == pytest.approx(11_500, abs=20)


# --------------------------------------------------------------------------
# Asymmetric transaction costs
# --------------------------------------------------------------------------


def test_selling_costs_more_than_buying():
    """The defining difference from a US venue: tax lands on the sell side."""
    costs = TradingCosts(commission_bps=1.5, sell_tax_bps={KOSPI: 15.0, KOSDAQ: 15.0})
    assert costs.sell_cost_bps(KOSPI) > costs.buy_cost_bps(KOSPI)
    assert costs.sell_cost_bps(KOSPI) == pytest.approx(16.5)
    assert costs.buy_cost_bps(KOSPI) == pytest.approx(1.5)


def test_cost_of_a_fill_reflects_the_side():
    costs = TradingCosts(commission_bps=1.5, sell_tax_bps={KOSPI: 15.0, KOSDAQ: 15.0})
    buy = costs.cost(10_000_000, "BUY", KOSPI)
    sell = costs.cost(10_000_000, "SELL", KOSPI)
    assert buy == pytest.approx(1_500.0)
    assert sell == pytest.approx(16_500.0)
    assert sell > buy


def test_round_trip_is_the_hurdle_rate():
    costs = TradingCosts(commission_bps=1.5, sell_tax_bps={KOSPI: 15.0, KOSDAQ: 15.0})
    assert costs.round_trip_bps(KOSPI) == pytest.approx(18.0)


def test_market_specific_tax_rates_apply():
    costs = TradingCosts(commission_bps=0.0, sell_tax_bps={KOSPI: 10.0, KOSDAQ: 20.0})
    assert costs.cost(1_000_000, "SELL", KOSPI) == pytest.approx(1_000.0)
    assert costs.cost(1_000_000, "SELL", KOSDAQ) == pytest.approx(2_000.0)


def test_slippage_always_works_against_the_order():
    costs = TradingCosts(slippage_bps=10.0)
    assert costs.apply_slippage(20_000, buying=True) > 20_000
    assert costs.apply_slippage(20_000, buying=False) < 20_000


def test_costs_load_from_config(config):
    costs = TradingCosts.from_config(config)
    assert costs.commission_bps > 0
    assert costs.sell_cost_bps(KOSPI) > costs.buy_cost_bps(KOSPI)
    assert "sell tax" in costs.describe()


# --------------------------------------------------------------------------
# Config-driven ladder
# --------------------------------------------------------------------------


def test_ladder_can_be_overridden_from_config():
    """KRX has revised the tick grid before; it must be changeable without code."""
    rules = KrxRules(
        {
            "krx": {
                "tick_ladder": {
                    KOSPI: [{"below": 1_000, "tick": 1}, {"below": "inf", "tick": 25}]
                }
            }
        }
    )
    assert rules.tick_size(500, KOSPI) == 1
    assert rules.tick_size(50_000, KOSPI) == 25


def test_project_config_ladder_is_well_formed(config):
    rules = KrxRules(config)
    for market in (KOSPI, KOSDAQ):
        previous = 0
        for price in (1_000, 3_000, 10_000, 30_000, 100_000, 300_000):
            tick = rules.tick_size(price, market)
            assert tick >= previous  # ticks never shrink as price rises
            previous = tick
