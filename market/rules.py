"""KRX trading rules: tick sizes, daily price limits and transaction costs.

Three things here differ fundamentally from the US market and each one can
silently invalidate an order or a backtest:

**Tick size.** KRX quotes on a price-dependent grid (1 / 5 / 10 / 50 / 100 /
500 / 1000 KRW). An order priced off the grid is rejected, so every stop and
limit price has to be snapped to it before submission.

**Daily price limit (±30%).** At the limit-up or limit-down price there is
essentially no counterparty, so an order placed there will not fill. The bot
skips rather than pretending it traded.

**Asymmetric costs.** US equities cost the same to buy and sell. In Korea the
*sell* side additionally pays transaction tax, which makes short-horizon
strategies far more expensive than the US-shaped cost model would suggest.

The tick table and tax rates live in ``config.yaml`` rather than being
hard-coded: KRX has revised both, and a stale constant compiled into the code
would quietly mis-price every order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

KOSPI = "KOSPI"
KOSDAQ = "KOSDAQ"

#: Default tick ladders as (price_below, tick). The last entry's bound is None,
#: meaning "at or above the previous bound".  Verify against the current KRX
#: rulebook and override in config.yaml if they change.
DEFAULT_TICK_LADDER: dict[str, tuple[tuple[float | None, int], ...]] = {
    KOSPI: (
        (2_000, 1),
        (5_000, 5),
        (20_000, 10),
        (50_000, 50),
        (200_000, 100),
        (500_000, 500),
        (None, 1_000),
    ),
    KOSDAQ: (
        (2_000, 1),
        (5_000, 5),
        (20_000, 10),
        (50_000, 50),
        (None, 100),
    ),
}

#: Daily price limit, as a fraction of the previous close.
DEFAULT_PRICE_LIMIT_PCT = 0.30


def tick_size(
    price: float,
    market: str = KOSPI,
    ladder: Mapping[str, Sequence[tuple[float | None, int]]] | None = None,
) -> int:
    """Return the quote increment applying at *price*."""
    ladder = ladder or DEFAULT_TICK_LADDER
    rungs = ladder.get(market) or ladder.get(KOSPI) or DEFAULT_TICK_LADDER[KOSPI]
    for bound, tick in rungs:
        if bound is None or price < bound:
            return int(tick)
    return int(rungs[-1][1])


def round_to_tick(
    price: float,
    market: str = KOSPI,
    mode: str = "nearest",
    ladder: Mapping[str, Sequence[tuple[float | None, int]]] | None = None,
) -> int:
    """Snap *price* onto the KRX quote grid.

    ``mode`` picks the rounding direction: ``"nearest"``, ``"down"`` (safer for
    a long entry limit) or ``"up"`` (safer for a long stop, since rounding a
    stop down would widen the loss beyond the risk budget).
    """
    if price <= 0:
        return 0
    tick = tick_size(price, market, ladder)
    if mode == "down":
        snapped = math.floor(price / tick) * tick
    elif mode == "up":
        snapped = math.ceil(price / tick) * tick
    else:
        snapped = round(price / tick) * tick
    # Rounding down near zero can land on 0, which is never a valid price.
    return int(max(snapped, tick))


@dataclass(frozen=True)
class PriceLimit:
    """The upper and lower bound a stock may trade at today."""

    reference_price: float
    upper: int
    lower: int
    limit_pct: float

    def at_limit(self, price: float) -> bool:
        return price >= self.upper or price <= self.lower

    def blocks_buy(self, price: float) -> bool:
        """Limit-up: buyers queue with no sellers, so a buy will not fill."""
        return price >= self.upper

    def blocks_sell(self, price: float) -> bool:
        """Limit-down: sellers queue with no buyers."""
        return price <= self.lower


def price_limits(
    previous_close: float,
    market: str = KOSPI,
    limit_pct: float = DEFAULT_PRICE_LIMIT_PCT,
    ladder: Mapping[str, Sequence[tuple[float | None, int]]] | None = None,
) -> PriceLimit:
    """Compute today's ±30% band from the previous close."""
    upper = round_to_tick(previous_close * (1 + limit_pct), market, "down", ladder)
    lower = round_to_tick(previous_close * (1 - limit_pct), market, "up", ladder)
    return PriceLimit(
        reference_price=previous_close, upper=upper, lower=lower, limit_pct=limit_pct
    )


@dataclass(frozen=True)
class TradingCosts:
    """Commission and tax assumptions, in basis points of notional.

    Buying pays commission only.  Selling pays commission **and** transaction
    tax, so ``sell_cost_bps`` is materially larger than ``buy_cost_bps`` - the
    asymmetry that makes high-turnover strategies expensive here.
    """

    commission_bps: float = 1.5
    #: Transaction tax on sales, by market. Verify against the current rate.
    sell_tax_bps: Mapping[str, float] = field(
        default_factory=lambda: {KOSPI: 15.0, KOSDAQ: 15.0}
    )
    slippage_bps: float = 10.0

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "TradingCosts":
        costs = config.get("costs") or {}
        tax = costs.get("sell_tax_bps") or {}
        return cls(
            commission_bps=float(costs.get("commission_bps", 1.5)),
            sell_tax_bps={
                KOSPI: float(tax.get("KOSPI", 15.0)),
                KOSDAQ: float(tax.get("KOSDAQ", 15.0)),
            },
            slippage_bps=float(costs.get("slippage_bps", 10.0)),
        )

    def buy_cost_bps(self, market: str = KOSPI) -> float:
        return self.commission_bps

    def sell_cost_bps(self, market: str = KOSPI) -> float:
        return self.commission_bps + float(self.sell_tax_bps.get(market, 15.0))

    def cost(self, notional: float, side: str, market: str = KOSPI) -> float:
        """Total cash cost of a fill, tax included on sells."""
        bps = self.sell_cost_bps(market) if side == "SELL" else self.buy_cost_bps(market)
        return abs(notional) * bps / 10_000.0

    def apply_slippage(self, price: float, buying: bool) -> float:
        """Slippage always works against the order."""
        bps = self.slippage_bps / 10_000.0
        return price * (1.0 + bps) if buying else price * (1.0 - bps)

    def round_trip_bps(self, market: str = KOSPI) -> float:
        """What a full in-and-out costs before slippage - the hurdle rate."""
        return self.buy_cost_bps(market) + self.sell_cost_bps(market)

    def describe(self) -> str:
        return (
            f"commission {self.commission_bps:.1f}bps/side, "
            f"sell tax KOSPI {self.sell_tax_bps.get(KOSPI, 0):.1f}bps / "
            f"KOSDAQ {self.sell_tax_bps.get(KOSDAQ, 0):.1f}bps, "
            f"slippage {self.slippage_bps:.1f}bps "
            f"(round trip ~ {self.round_trip_bps():.1f}bps + slippage)"
        )


class KrxRules:
    """Config-driven facade over the functions above."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        config = config or {}
        krx = config.get("krx") or {}
        self.limit_pct = float(krx.get("price_limit_pct", DEFAULT_PRICE_LIMIT_PCT))
        self.ladder = self._parse_ladder(krx.get("tick_ladder"))
        self.costs = TradingCosts.from_config(config)

    @staticmethod
    def _parse_ladder(raw: Any) -> dict[str, tuple[tuple[float | None, int], ...]]:
        if not raw:
            return dict(DEFAULT_TICK_LADDER)
        parsed: dict[str, tuple[tuple[float | None, int], ...]] = {}
        for market, rungs in raw.items():
            entries: list[tuple[float | None, int]] = []
            for rung in rungs:
                below = rung.get("below")
                entries.append((None if below in (None, "inf") else float(below), int(rung["tick"])))
            parsed[market] = tuple(entries)
        return parsed or dict(DEFAULT_TICK_LADDER)

    def tick_size(self, price: float, market: str = KOSPI) -> int:
        return tick_size(price, market, self.ladder)

    def round_to_tick(self, price: float, market: str = KOSPI, mode: str = "nearest") -> int:
        return round_to_tick(price, market, mode, self.ladder)

    def price_limits(self, previous_close: float, market: str = KOSPI) -> PriceLimit:
        return price_limits(previous_close, market, self.limit_pct, self.ladder)

    def stop_price(self, entry: float, stop_distance: float, market: str = KOSPI) -> int:
        """Snap a long stop onto the grid, rounding *up*.

        Rounding a long stop down would place it further from entry than the
        risk budget allows, so the conservative direction is up.
        """
        return self.round_to_tick(entry - stop_distance, market, mode="up")
