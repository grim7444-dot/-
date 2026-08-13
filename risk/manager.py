"""Risk control: sizing, hard stops, the drawdown kill switch and pre-trade checks.

Safety rules enforced here:

* **Rule 4** — every order risks exactly 1% of *current* equity.  Quantity is
  ``(equity x risk_pct) / stop_distance``; a non-positive stop distance means
  the order is skipped rather than sized on a guess.
* **Rule 5** — a 10% drawdown from peak equity blocks new orders, cancels
  working orders, flattens positions and persists ``STOPPED``.
* **Rule 6** — ``STOPPED`` is sticky: only an explicit ``resume`` clears it.
* **Rule 7** — tradability, session hours, minimum quantity, duplicate orders
  and available cash are all checked before anything is sent.

The sizing maths is deliberately plain: because the hard stop sits exactly one
ATR away, a 1-ATR adverse move loses ``equity x risk_pct`` regardless of the
instrument.  A quiet asset (small ATR) therefore gets a large quantity and a
violent one gets a small quantity, while the dollars at risk never move.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from portfolio import LONG, SHORT, Portfolio, Position

logger = logging.getLogger("bot.risk")


# --------------------------------------------------------------------------
# Sizing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SizingResult:
    """Outcome of a position-sizing calculation."""

    qty: float
    #: Unrounded quantity — a 1-ATR move against this loses exactly risk_amount.
    qty_exact: float
    stop_distance: float
    risk_amount: float
    notional: float
    skipped: bool = False
    reason: str = ""

    @property
    def ok(self) -> bool:
        return not self.skipped and self.qty > 0

    def realised_risk(self) -> float:
        """Dollars actually at risk after rounding/capping."""
        return self.qty * self.stop_distance


def _floor_to(value: float, precision: int) -> float:
    factor = 10**precision
    return math.floor(value * factor) / factor


def position_size(
    equity: float,
    atr: float,
    price: float,
    risk_pct: float = 0.01,
    hard_stop_atr_mult: float = 1.0,
    fractional: bool = False,
    min_qty: float = 1.0,
    qty_precision: int = 6,
    available_cash: float | None = None,
    max_position_notional_pct: float | None = None,
) -> SizingResult:
    """Size an order so a 1-ATR adverse move costs ``equity * risk_pct``."""
    stop_distance = float(atr) * float(hard_stop_atr_mult)
    risk_amount = float(equity) * float(risk_pct)

    def _skip(reason: str) -> SizingResult:
        return SizingResult(
            qty=0.0,
            qty_exact=0.0,
            stop_distance=max(stop_distance, 0.0),
            risk_amount=max(risk_amount, 0.0),
            notional=0.0,
            skipped=True,
            reason=reason,
        )

    if not math.isfinite(stop_distance) or stop_distance <= 0:
        # Rule 4: no stop distance, no order.  Never fall back to a default size.
        return _skip(f"stop distance is not positive ({stop_distance:.6f}); order skipped")
    if not math.isfinite(price) or price <= 0:
        return _skip(f"price is not positive ({price})")
    if not math.isfinite(equity) or equity <= 0:
        return _skip(f"equity is not positive ({equity})")
    if risk_pct <= 0:
        return _skip(f"risk_pct is not positive ({risk_pct})")

    qty_exact = risk_amount / stop_distance
    qty = _floor_to(qty_exact, qty_precision) if fractional else float(math.floor(qty_exact))

    if qty <= 0:
        return _skip(
            f"risk budget {risk_amount:.2f} buys less than one tradable unit "
            f"at a {stop_distance:.4f} stop distance"
        )

    # Cap 1: never let one position dominate the account.
    if max_position_notional_pct:
        max_notional = equity * float(max_position_notional_pct)
        if qty * price > max_notional:
            capped = max_notional / price
            qty = _floor_to(capped, qty_precision) if fractional else float(math.floor(capped))

    # Cap 2: never order more than the cash on hand supports.
    if available_cash is not None and available_cash >= 0:
        if qty * price > available_cash:
            capped = available_cash / price
            qty = _floor_to(capped, qty_precision) if fractional else float(math.floor(capped))

    if qty <= 0:
        return _skip("position capped to zero by notional/cash limits")
    if qty < min_qty:
        return _skip(f"quantity {qty} is below the minimum order size {min_qty}")

    return SizingResult(
        qty=qty,
        qty_exact=qty_exact,
        stop_distance=stop_distance,
        risk_amount=risk_amount,
        notional=qty * price,
    )


def hard_stop_price(side: str, entry_price: float, stop_distance: float) -> float:
    """The mandatory 1%-of-equity stop attached to every position."""
    if stop_distance <= 0:
        raise ValueError("stop_distance must be positive to place a hard stop")
    if side == LONG:
        return entry_price - stop_distance
    if side == SHORT:
        return entry_price + stop_distance
    raise ValueError(f"unknown side {side!r}")


# --------------------------------------------------------------------------
# Drawdown guard
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DrawdownStatus:
    peak_equity: float
    equity: float
    drawdown_pct: float
    limit_pct: float
    breached: bool

    def describe(self) -> str:
        return (
            f"drawdown {self.drawdown_pct:.2%} of peak {self.peak_equity:,.2f} "
            f"(limit {self.limit_pct:.2%})"
        )


def evaluate_drawdown(peak_equity: float, equity: float, limit_pct: float) -> DrawdownStatus:
    if peak_equity <= 0:
        return DrawdownStatus(peak_equity, equity, 0.0, limit_pct, False)
    drawdown = max(0.0, (peak_equity - equity) / peak_equity)
    return DrawdownStatus(
        peak_equity=peak_equity,
        equity=equity,
        drawdown_pct=drawdown,
        limit_pct=limit_pct,
        breached=drawdown >= limit_pct,
    )


# --------------------------------------------------------------------------
# Correlation filter
# --------------------------------------------------------------------------


def correlation_block(
    symbol: str,
    side: str,
    positions: Mapping[str, Position],
    blockers: Sequence[str] = ("SPY", "QQQ"),
    blocked_symbol: str = "BTC/USD",
    blocked_side: str = LONG,
    enabled: bool = True,
) -> tuple[bool, str]:
    """Return ``(blocked, reason)`` for a proposed new entry.

    When SPY and QQQ are *both* long the book already carries a full helping of
    risk-on beta; adding a BTC long on top would stack the same bet a third
    time.  Exits are never blocked.
    """
    if not enabled:
        return False, ""
    if symbol != blocked_symbol or side != blocked_side:
        return False, ""
    longs = [
        s
        for s in blockers
        if (pos := positions.get(s)) is not None and pos.side == LONG
    ]
    if len(longs) == len(list(blockers)) and longs:
        return True, (
            f"correlation filter: {' and '.join(longs)} are both long, "
            f"so a new {blocked_symbol} {blocked_side} is blocked"
        )
    return False, ""


# --------------------------------------------------------------------------
# Pre-trade checks (rule 7)
# --------------------------------------------------------------------------


@dataclass
class TradeContext:
    """Everything the pre-trade gate needs to know about a proposed order."""

    symbol: str
    side: str
    qty: float
    price: float
    asset_class: str = "us_equity"
    tradable: bool = True
    market_open: bool = True
    min_qty: float = 1.0
    open_order_symbols: Iterable[str] = field(default_factory=tuple)
    existing_position: Position | None = None
    available_cash: float = 0.0
    is_exit: bool = False


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    failures: tuple[str, ...] = ()
    checks: Mapping[str, bool] = field(default_factory=dict)

    def describe(self) -> str:
        return "; ".join(self.failures) if self.failures else "all pre-trade checks passed"


def pre_trade_checks(ctx: TradeContext, crypto_24_7: bool = True) -> CheckResult:
    """Run the five mandatory pre-order checks."""
    checks: dict[str, bool] = {}
    failures: list[str] = []

    # 1. Is the asset tradable at all?
    checks["tradable"] = bool(ctx.tradable)
    if not ctx.tradable:
        failures.append(f"{ctx.symbol} is not tradable")

    # 2. Session hours — crypto is 24/7, equity ETFs are regular-session only.
    session_ok = True if (ctx.asset_class == "crypto" and crypto_24_7) else bool(ctx.market_open)
    checks["market_hours"] = session_ok
    if not session_ok:
        failures.append(f"{ctx.symbol} market is closed")

    # 3. Minimum order quantity.
    qty_ok = ctx.qty > 0 and ctx.qty >= ctx.min_qty
    checks["min_qty"] = qty_ok
    if not qty_ok:
        failures.append(f"quantity {ctx.qty} is below the minimum {ctx.min_qty}")

    # 4. Duplicate order / duplicate exposure.
    open_symbols = set(ctx.open_order_symbols or ())
    duplicate_order = ctx.symbol in open_symbols
    duplicate_position = (
        not ctx.is_exit
        and ctx.existing_position is not None
        and ctx.existing_position.side == ctx.side
    )
    checks["no_duplicate"] = not (duplicate_order or duplicate_position)
    if duplicate_order:
        failures.append(f"an order for {ctx.symbol} is already working")
    if duplicate_position:
        failures.append(f"already {ctx.side} {ctx.symbol}")

    # 5. Cash on hand.  Exits release cash, so they are exempt.
    notional = ctx.qty * ctx.price
    cash_ok = ctx.is_exit or notional <= ctx.available_cash
    checks["cash"] = cash_ok
    if not cash_ok:
        failures.append(
            f"notional {notional:,.2f} exceeds available cash {ctx.available_cash:,.2f}"
        )

    return CheckResult(passed=not failures, failures=tuple(failures), checks=checks)


# --------------------------------------------------------------------------
# Manager
# --------------------------------------------------------------------------


class RiskManager:
    """Config-driven facade over the functions above."""

    def __init__(self, config: Mapping[str, Any], portfolio: Portfolio) -> None:
        self.config = config
        self.portfolio = portfolio
        risk_cfg = config.get("risk") or {}
        self.risk_pct = float(risk_cfg.get("per_trade_pct", 0.01))
        self.max_drawdown_pct = float(risk_cfg.get("max_drawdown_pct", 0.10))
        self.atr_period = int(risk_cfg.get("atr_period", 14))
        self.hard_stop_atr_mult = float(risk_cfg.get("hard_stop_atr_mult", 1.0))
        self.max_position_notional_pct = risk_cfg.get("max_position_notional_pct")
        self.corr_cfg = dict(risk_cfg.get("correlation_filter") or {})
        self.crypto_24_7 = bool((config.get("schedule") or {}).get("crypto_24_7", True))

    # -- sizing ------------------------------------------------------------

    def size(
        self,
        symbol: str,
        equity: float,
        atr: float,
        price: float,
        asset_cfg: Mapping[str, Any] | None = None,
        available_cash: float | None = None,
    ) -> SizingResult:
        asset_cfg = asset_cfg or {}
        return position_size(
            equity=equity,
            atr=atr,
            price=price,
            risk_pct=self.risk_pct,
            hard_stop_atr_mult=self.hard_stop_atr_mult,
            fractional=bool(asset_cfg.get("fractional", False)),
            min_qty=float(asset_cfg.get("min_qty", 1)),
            qty_precision=int(asset_cfg.get("qty_precision", 6)),
            available_cash=available_cash,
            max_position_notional_pct=self.max_position_notional_pct,
        )

    def max_loss_per_trade(self, equity: float) -> float:
        """The headline number printed in the live banner (rule 3)."""
        return max(0.0, equity * self.risk_pct)

    # -- drawdown ----------------------------------------------------------

    def check_drawdown(self, equity: float) -> DrawdownStatus:
        return evaluate_drawdown(
            peak_equity=max(self.portfolio.state.peak_equity, equity),
            equity=equity,
            limit_pct=self.max_drawdown_pct,
        )

    def trip_kill_switch(self, status: DrawdownStatus, broker: Any) -> None:
        """Rule 5: block, cancel, flatten, persist STOPPED."""
        reason = f"drawdown kill switch: {status.describe()}"
        logger.critical("KILL SWITCH TRIPPED — %s", reason)
        # Order matters: stop first so a crash mid-teardown still leaves the
        # bot in STOPPED rather than RUNNING.
        self.portfolio.stop(reason)
        try:
            broker.cancel_all_orders()
        except Exception as exc:
            logger.error("failed to cancel open orders during kill switch: %s", exc)
        try:
            broker.close_all_positions()
        except Exception as exc:
            logger.error("failed to close positions during kill switch: %s", exc)
        for symbol in list(self.portfolio.state.positions):
            self.portfolio.state.positions.pop(symbol, None)
        self.portfolio.save()

    # -- gating ------------------------------------------------------------

    def correlation_block(self, symbol: str, side: str) -> tuple[bool, str]:
        return correlation_block(
            symbol=symbol,
            side=side,
            positions=self.portfolio.positions(),
            blockers=tuple(self.corr_cfg.get("blockers", ("SPY", "QQQ"))),
            blocked_symbol=self.corr_cfg.get("blocked_symbol", "BTC/USD"),
            blocked_side=self.corr_cfg.get("blocked_side", LONG),
            enabled=bool(self.corr_cfg.get("enabled", True)),
        )

    def can_open_new(self, symbol: str, side: str) -> tuple[bool, str]:
        """Cheap gate applied before any sizing work is done."""
        if self.portfolio.stopped:
            return False, (
                "bot is STOPPED "
                f"({self.portfolio.state.stopped_reason or 'no reason recorded'}); "
                "run `python main.py resume` to clear"
            )
        blocked, reason = self.correlation_block(symbol, side)
        if blocked:
            return False, reason
        return True, ""

    def pre_trade_checks(self, ctx: TradeContext) -> CheckResult:
        return pre_trade_checks(ctx, crypto_24_7=self.crypto_24_7)
