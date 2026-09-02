"""Risk control: sizing, hard stops, kill switch, portfolio caps and pre-trade checks.

Safety rules enforced here:

* **Rule 4** - every order risks exactly 1% of *current* equity. Quantity is
  ``(equity x risk_pct) / stop_distance``; a non-positive stop distance means
  the order is skipped rather than sized on a guess.
* **Rule 5** - a 10% drawdown from peak equity blocks new orders, cancels
  working orders, flattens positions and persists ``STOPPED``.
* **Rule 6** - ``STOPPED`` is sticky: only an explicit ``resume`` clears it.
* **Rule 7** - tradability, session phase, minimum quantity, duplicate orders,
  available cash and the daily price limit are all checked before anything
  is sent.

Two gates exist here that the US version did not need, because this portfolio
is eight individual small/mid caps rather than five broad instruments:

* a **portfolio risk cap** - 1% per position across 8 names would put 8% at
  risk simultaneously, so total open risk and position count are both capped;
* a **theme filter** - two names in the same theme are, in practice, one bet
  sized twice, so only one position per theme is allowed.

The sizing maths is deliberately plain: because the hard stop sits exactly one
ATR away, a 1-ATR adverse move loses ``equity x risk_pct`` regardless of the
instrument. A quiet stock gets a large quantity and a violent one gets a small
quantity, while the money at risk never moves.
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
    #: Unrounded quantity - a 1-ATR move against this loses exactly risk_amount.
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
        """Money actually at risk after rounding and capping."""
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
    qty_precision: int = 0,
    available_cash: float | None = None,
    max_position_notional_pct: float | None = None,
    risk_budget: float | None = None,
) -> SizingResult:
    """Size an order so a 1-ATR adverse move costs ``equity * risk_pct``.

    ``risk_budget`` optionally caps the money at risk below the per-trade
    amount - used when the portfolio-wide risk cap has partial room left.
    """
    stop_distance = float(atr) * float(hard_stop_atr_mult)
    risk_amount = float(equity) * float(risk_pct)
    if risk_budget is not None:
        risk_amount = min(risk_amount, max(0.0, float(risk_budget)))

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
        # Rule 4: no stop distance, no order. Never fall back to a default size.
        return _skip(f"stop distance is not positive ({stop_distance:.6f}); order skipped")
    if not math.isfinite(price) or price <= 0:
        return _skip(f"price is not positive ({price})")
    if not math.isfinite(equity) or equity <= 0:
        return _skip(f"equity is not positive ({equity})")
    if risk_pct <= 0:
        return _skip(f"risk_pct is not positive ({risk_pct})")
    if risk_amount <= 0:
        return _skip("no risk budget left under the portfolio cap")

    qty_exact = risk_amount / stop_distance
    qty = _floor_to(qty_exact, qty_precision) if fractional else float(math.floor(qty_exact))

    if qty <= 0:
        return _skip(
            f"risk budget {risk_amount:,.0f} buys less than one share "
            f"at a {stop_distance:,.2f} stop distance"
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
            f"drawdown {self.drawdown_pct:.2%} of peak {self.peak_equity:,.0f} "
            f"(limit {self.limit_pct:.2%})"
        )


def evaluate_drawdown(peak_equity: float, equity: float, limit_pct: float) -> DrawdownStatus:
    """Rule 5's drawdown kill switch.

    limit_pct <= 0 disables it entirely (2026-09-02, explicit user request:
    "킬스위치는 없애는게 좋을듯 하다" -- confirmed "완전히 제거" after being
    told this removes the only floor on the live account) -- never breached,
    regardless of how far equity has fallen from its peak. Only ``limit_pct
    <= 0`` disables it; ``limit_pct == 0`` would otherwise breach on almost
    every call, since drawdown >= 0 is true anywhere off the exact peak --
    the opposite of what "disabled" has to mean here.
    """
    if peak_equity <= 0:
        return DrawdownStatus(peak_equity, equity, 0.0, limit_pct, False)
    drawdown = max(0.0, (peak_equity - equity) / peak_equity)
    return DrawdownStatus(
        peak_equity=peak_equity,
        equity=equity,
        drawdown_pct=drawdown,
        limit_pct=limit_pct,
        # Real drawdown is still reported either way (status/reports keep
        # showing it) -- only whether it can ever *breach* changes.
        breached=(limit_pct > 0 and drawdown >= limit_pct),
    )


# --------------------------------------------------------------------------
# Portfolio-level caps
# --------------------------------------------------------------------------


def position_risk(position: Position, price: float | None = None) -> float:
    """Money still at risk on an open position.

    Measured against the stop actually in force, so a trailing stop that has
    ratcheted past entry correctly reports zero remaining risk.
    """
    reference = price if price is not None else position.entry_price
    stop = position.effective_stop()
    distance = (reference - stop) if position.is_long else (stop - reference)
    return max(0.0, distance * position.qty)


@dataclass(frozen=True)
class PortfolioCapacity:
    open_positions: int
    max_positions: int
    open_risk: float
    max_risk: float
    remaining_risk: float

    @property
    def has_slot(self) -> bool:
        return self.open_positions < self.max_positions

    @property
    def has_risk_room(self) -> bool:
        return self.remaining_risk > 0

    @property
    def ok(self) -> bool:
        return self.has_slot and self.has_risk_room

    def describe(self) -> str:
        return (
            f"{self.open_positions}/{self.max_positions} positions, "
            f"open risk {self.open_risk:,.0f} of {self.max_risk:,.0f} "
            f"({self.remaining_risk:,.0f} left)"
        )


def portfolio_capacity(
    positions: Mapping[str, Position],
    equity: float,
    max_total_risk_pct: float = 0.06,
    max_positions: int = 6,
    prices: Mapping[str, float] | None = None,
) -> PortfolioCapacity:
    """How much new risk the book can still take on."""
    prices = prices or {}
    open_risk = sum(position_risk(p, prices.get(s)) for s, p in positions.items())
    max_risk = max(0.0, equity * max_total_risk_pct)
    return PortfolioCapacity(
        open_positions=len(positions),
        max_positions=max_positions,
        open_risk=open_risk,
        max_risk=max_risk,
        remaining_risk=max(0.0, max_risk - open_risk),
    )


# --------------------------------------------------------------------------
# Theme filter
# --------------------------------------------------------------------------


def theme_of(code: str, themes: Mapping[str, Sequence[str]]) -> str | None:
    for theme, members in themes.items():
        if code in members:
            return theme
    return None


def theme_block(
    code: str,
    positions: Mapping[str, Position],
    themes: Mapping[str, Sequence[str]],
    enabled: bool = True,
) -> tuple[bool, str]:
    """Refuse a second position in a theme that is already held.

    Two construction names or two robotics names move together; sizing each at
    1% would put 2% behind what is really one thesis. Exits are never blocked.
    """
    if not enabled or not themes:
        return False, ""
    theme = theme_of(code, themes)
    if theme is None:
        return False, ""
    held = [s for s in positions if s != code and theme_of(s, themes) == theme]
    if held:
        return True, (
            f"theme filter: already holding {', '.join(held)} in theme '{theme}', "
            f"so a new {code} position is blocked"
        )
    return False, ""


# --------------------------------------------------------------------------
# Pre-trade checks (rule 7)
# --------------------------------------------------------------------------


@dataclass
class TradeContext:
    """Everything the pre-trade gate needs to know about a proposed order."""

    code: str
    side: str
    qty: float
    price: float
    tradable: bool = True
    #: True only during continuous trading - auctions and closed count as False.
    session_ok: bool = True
    session_reason: str = ""
    min_qty: float = 1.0
    open_order_codes: Iterable[str] = field(default_factory=tuple)
    existing_position: Position | None = None
    available_cash: float = 0.0
    is_exit: bool = False
    #: True when the price is pinned at the daily limit, where fills do not happen.
    at_price_limit: bool = False
    price_limit_reason: str = ""


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    failures: tuple[str, ...] = ()
    checks: Mapping[str, bool] = field(default_factory=dict)

    def describe(self) -> str:
        return "; ".join(self.failures) if self.failures else "all pre-trade checks passed"


def pre_trade_checks(ctx: TradeContext) -> CheckResult:
    """Run the six mandatory pre-order checks."""
    checks: dict[str, bool] = {}
    failures: list[str] = []

    # 1. Is the stock tradable at all (not suspended, not delisted)?
    checks["tradable"] = bool(ctx.tradable)
    if not ctx.tradable:
        failures.append(f"{ctx.code} is not tradable (suspended or delisted)")

    # 2. Session phase - continuous trading only, never a call auction.
    checks["session"] = bool(ctx.session_ok)
    if not ctx.session_ok:
        failures.append(ctx.session_reason or f"{ctx.code}: market not in continuous trading")

    # 3. Minimum order quantity.
    qty_ok = ctx.qty > 0 and ctx.qty >= ctx.min_qty
    checks["min_qty"] = qty_ok
    if not qty_ok:
        failures.append(f"quantity {ctx.qty} is below the minimum {ctx.min_qty}")

    # 4. Duplicate order / duplicate exposure.
    open_codes = set(ctx.open_order_codes or ())
    duplicate_order = ctx.code in open_codes
    duplicate_position = (
        not ctx.is_exit
        and ctx.existing_position is not None
        and ctx.existing_position.side == ctx.side
    )
    checks["no_duplicate"] = not (duplicate_order or duplicate_position)
    if duplicate_order:
        failures.append(f"an order for {ctx.code} is already working")
    if duplicate_position:
        failures.append(f"already {ctx.side} {ctx.code}")

    # 5. Cash on hand. Exits release cash, so they are exempt.
    notional = ctx.qty * ctx.price
    cash_ok = ctx.is_exit or notional <= ctx.available_cash
    checks["cash"] = cash_ok
    if not cash_ok:
        failures.append(
            f"notional {notional:,.0f} exceeds available cash {ctx.available_cash:,.0f}"
        )

    # 6. Daily price limit. At limit-up/down there is no counterparty.
    checks["price_limit"] = not ctx.at_price_limit
    if ctx.at_price_limit:
        failures.append(ctx.price_limit_reason or f"{ctx.code} is at its daily price limit")

    return CheckResult(passed=not failures, failures=tuple(failures), checks=checks)


# --------------------------------------------------------------------------
# Manager
# --------------------------------------------------------------------------


class RiskManager:
    """Config-driven facade over the functions above."""

    def __init__(
        self,
        config: Mapping[str, Any],
        portfolio: Portfolio,
        calendar: Any | None = None,
    ) -> None:
        self.config = config
        self.portfolio = portfolio
        #: Optional KrxCalendar. Without it the kill switch cannot tell whether
        #: a market order would even reach a book, so it flattens regardless.
        self.calendar = calendar
        risk_cfg = config.get("risk") or {}
        self.risk_pct = float(risk_cfg.get("per_trade_pct", 0.01))
        self.max_drawdown_pct = float(risk_cfg.get("max_drawdown_pct", 0.10))
        self.atr_period = int(risk_cfg.get("atr_period", 14))
        self.hard_stop_atr_mult = float(risk_cfg.get("hard_stop_atr_mult", 1.0))
        self.max_position_notional_pct = risk_cfg.get("max_position_notional_pct")
        self.max_total_risk_pct = float(risk_cfg.get("max_total_risk_pct", 0.06))
        self.max_positions = int(risk_cfg.get("max_open_positions", 6))
        self.long_only = bool(risk_cfg.get("long_only", True))
        self.themes: dict[str, list[str]] = {
            k: list(v) for k, v in (config.get("themes") or {}).items()
        }
        self.theme_filter_enabled = bool(
            (risk_cfg.get("theme_filter") or {}).get("enabled", True)
        )

    # -- sizing ------------------------------------------------------------

    def size(
        self,
        code: str,
        equity: float,
        atr: float,
        price: float,
        asset_cfg: Mapping[str, Any] | None = None,
        available_cash: float | None = None,
        risk_budget: float | None = None,
    ) -> SizingResult:
        asset_cfg = asset_cfg or {}
        return position_size(
            equity=equity,
            atr=atr,
            price=price,
            risk_pct=self.risk_pct,
            hard_stop_atr_mult=self.hard_stop_atr_mult,
            fractional=False,  # KRX equities trade in whole shares
            min_qty=float(asset_cfg.get("min_qty", 1)),
            qty_precision=0,
            available_cash=available_cash,
            max_position_notional_pct=self.max_position_notional_pct,
            risk_budget=risk_budget,
        )

    def max_loss_per_trade(self, equity: float) -> float:
        """The headline number printed in the live banner (rule 3)."""
        return max(0.0, equity * self.risk_pct)

    def max_portfolio_loss(self, equity: float) -> float:
        return max(0.0, equity * self.max_total_risk_pct)

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
        logger.critical("KILL SWITCH TRIPPED - %s", reason)
        # Order matters: stop first so a crash mid-teardown still leaves the
        # bot in STOPPED rather than RUNNING.
        self.portfolio.stop(reason)
        try:
            broker.cancel_all_orders()
        except Exception as exc:
            logger.error("failed to cancel open orders during kill switch: %s", exc)

        # Market orders sent outside continuous trading do not reach a book.
        # Firing them anyway produces order IDs that never come back and a log
        # that claims positions were closed when they were not.
        if self.calendar is not None:
            tradable, why = self.calendar.can_place_market_order()
            if not tradable:
                logger.critical(
                    "kill switch could NOT flatten: %s. Positions are still open "
                    "and unprotected; they will be closed at the next open, or "
                    "close them by hand.",
                    why,
                )
                self.portfolio.save()
                return

        try:
            broker.close_all_positions()
        except Exception as exc:
            logger.error("failed to close positions during kill switch: %s", exc)
        for code in list(self.portfolio.state.positions):
            self.portfolio.state.positions.pop(code, None)
        self.portfolio.save()

    # -- gating ------------------------------------------------------------

    def capacity(
        self, equity: float, prices: Mapping[str, float] | None = None
    ) -> PortfolioCapacity:
        return portfolio_capacity(
            positions=self.portfolio.positions(),
            equity=equity,
            max_total_risk_pct=self.max_total_risk_pct,
            max_positions=self.max_positions,
            prices=prices,
        )

    def theme_block(self, code: str) -> tuple[bool, str]:
        return theme_block(
            code=code,
            positions=self.portfolio.positions(),
            themes=self.themes,
            enabled=self.theme_filter_enabled,
        )

    def can_open_new(
        self,
        code: str,
        side: str,
        equity: float = 0.0,
        prices: Mapping[str, float] | None = None,
    ) -> tuple[bool, str]:
        """Cheap gate applied before any sizing work is done."""
        if self.portfolio.stopped:
            return False, (
                "bot is STOPPED "
                f"({self.portfolio.state.stopped_reason or 'no reason recorded'}); "
                "run `python main.py resume` to clear"
            )
        if self.long_only and side == SHORT:
            return False, (
                "long-only mode: short signals close a position but never open one "
                "(retail short selling of KRX equities is not available)"
            )
        blocked, reason = self.theme_block(code)
        if blocked:
            return False, reason
        if equity > 0:
            cap = self.capacity(equity, prices)
            if not cap.has_slot:
                return False, f"portfolio full: {cap.describe()}"
            if not cap.has_risk_room:
                return False, f"portfolio risk cap reached: {cap.describe()}"
        return True, ""

    def pre_trade_checks(self, ctx: TradeContext) -> CheckResult:
        return pre_trade_checks(ctx)
