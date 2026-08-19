"""Event-driven backtester with a strict next-bar fill rule.

Safety rule 10: a signal produced on bar ``i`` can only be filled from bar
``i+1`` onwards. This is enforced structurally, in two layers:

1. ``Strategy.evaluate`` is handed ``bars.iloc[:i+1]`` - a future bar is simply
   not reachable from inside a strategy;
2. an actionable signal schedules a *pending order* whose execution index is
   ``i+1``, and the fill price is that bar's **open** (never bar ``i``'s
   close). Every fill records both indices and the engine asserts
   ``fill_index == signal_index + 1``.

Safety rule 11: commission, transaction tax and slippage come from
``config.yaml`` and are reported alongside every result set.

All stocks share one equity curve and one event stream ordered by time, so the
theme filter, the portfolio risk cap and the drawdown kill switch behave the
same way they do in live trading.

Two KRX specifics are modelled that the US version had no need for: the sell
side pays transaction tax on top of commission, and a bar pinned at the daily
±30% limit cannot be traded.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

import pandas as pd

from data import BarSet
from market.rules import KOSPI, KrxRules, TradingCosts
from portfolio import LONG, Position
from risk.manager import (
    hard_stop_price,
    evaluate_drawdown,
    portfolio_capacity,
    position_size,
    theme_block,
)
from strategies.base import Action, Strategy

logger = logging.getLogger("bot.backtest")

ENTRY = "entry"
EXIT = "exit"
STOP = "stop"


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Fill:
    """A single execution, tagged with the bar that produced it."""

    code: str
    kind: str                 # entry / exit / stop
    side: str
    #: Index of the bar whose close produced the signal. -1 for stop-outs.
    signal_index: int
    #: Index of the bar the order actually executed on.
    fill_index: int
    signal_time: pd.Timestamp | None
    fill_time: pd.Timestamp
    #: Close of the signal bar - recorded so tests can prove we did not fill there.
    signal_close: float
    price: float
    qty: float
    costs: float
    slippage_cost: float
    reason: str = ""

    @property
    def signal_driven(self) -> bool:
        return self.signal_index >= 0


@dataclass
class ClosedTrade:
    code: str
    side: str
    qty: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    gross_pnl: float
    costs: float
    pnl: float
    exit_reason: str
    strategy: str = ""


@dataclass
class BacktestResult:
    start: datetime | None
    end: datetime | None
    starting_equity: float
    ending_equity: float
    equity_curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    trades: list[ClosedTrade] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    costs: TradingCosts = field(default_factory=TradingCosts)
    synthetic_data: bool = False
    kill_switch_tripped_at: pd.Timestamp | None = None
    kill_switch_reason: str = ""
    skipped_orders: list[str] = field(default_factory=list)

    @property
    def total_return_pct(self) -> float:
        if self.starting_equity <= 0:
            return 0.0
        return (self.ending_equity - self.starting_equity) / self.starting_equity

    @property
    def total_costs(self) -> float:
        return sum(f.costs for f in self.fills)

    @property
    def total_slippage(self) -> float:
        return sum(f.slippage_cost for f in self.fills)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.pnl > 0) / len(self.trades)

    @property
    def max_drawdown_pct(self) -> float:
        peak = -math.inf
        worst = 0.0
        for _, equity in self.equity_curve:
            peak = max(peak, equity)
            if peak > 0:
                worst = max(worst, (peak - equity) / peak)
        return worst

    def cagr(self) -> float:
        if not self.equity_curve or self.starting_equity <= 0:
            return 0.0
        first, last = self.equity_curve[0][0], self.equity_curve[-1][0]
        years = (last - first).total_seconds() / (365.25 * 24 * 3600)
        if years <= 0 or self.ending_equity <= 0:
            return 0.0
        return (self.ending_equity / self.starting_equity) ** (1 / years) - 1

    def per_code_pnl(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for trade in self.trades:
            out[trade.code] = out.get(trade.code, 0.0) + trade.pnl
        return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))

    def verify_no_lookahead(self) -> bool:
        """Every signal-driven fill happened on the bar after its signal."""
        return all(f.fill_index == f.signal_index + 1 for f in self.fills if f.signal_driven)


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


@dataclass
class _PendingOrder:
    code: str
    action: Action
    signal_index: int
    fill_index: int
    signal_time: pd.Timestamp
    signal_close: float
    atr: float
    stop_distance: float
    trail_stop: float | None
    reason: str


class Backtester:
    """Portfolio backtest across the KRX universe."""

    def __init__(
        self,
        config: Mapping[str, Any],
        starting_equity: float | None = None,
        apply_drawdown_guard: bool = True,
    ) -> None:
        self.config = config
        risk_cfg = config.get("risk") or {}
        self.risk_pct = float(risk_cfg.get("per_trade_pct", 0.01))
        self.max_drawdown_pct = float(risk_cfg.get("max_drawdown_pct", 0.10))
        self.hard_stop_atr_mult = float(risk_cfg.get("hard_stop_atr_mult", 1.0))
        self.max_position_notional_pct = risk_cfg.get("max_position_notional_pct")
        self.max_total_risk_pct = float(risk_cfg.get("max_total_risk_pct", 0.06))
        self.max_positions = int(risk_cfg.get("max_open_positions", 6))
        self.long_only = bool(risk_cfg.get("long_only", True))
        self.themes = {k: list(v) for k, v in (config.get("themes") or {}).items()}
        self.theme_filter_enabled = bool(
            (risk_cfg.get("theme_filter") or {}).get("enabled", True)
        )

        self.rules = KrxRules(config)
        self.costs = self.rules.costs
        self.starting_equity = float(
            starting_equity
            if starting_equity is not None
            else (config.get("account") or {}).get("starting_equity", 10_000_000.0)
        )
        self.apply_drawdown_guard = apply_drawdown_guard
        bt_cfg = config.get("backtest") or {}
        self.fill_delay_bars = int(bt_cfg.get("fill_delay_bars", 1))
        if self.fill_delay_bars < 1:
            raise ValueError("fill_delay_bars must be >= 1: same-bar fills are look-ahead")
        self.respect_price_limits = bool(bt_cfg.get("respect_price_limits", True))

    # ----------------------------------------------------------------------

    def run(
        self,
        barsets: Mapping[str, BarSet],
        strategies: Mapping[str, Strategy],
        universe: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> BacktestResult:
        universe = universe or {}

        equity = self.starting_equity
        peak_equity = equity
        positions: dict[str, Position] = {}
        pending: dict[str, _PendingOrder] = {}
        fills: list[Fill] = []
        trades: list[ClosedTrade] = []
        curve: list[tuple[pd.Timestamp, float]] = []
        skipped: list[str] = []
        last_price: dict[str, float] = {}
        halted = False
        halt_time: pd.Timestamp | None = None
        halt_reason = ""

        frames = {c: bs.bars for c, bs in barsets.items() if len(bs.bars) > 0}
        markets = {c: bs.market for c, bs in barsets.items()}
        events = self._build_events(frames, strategies)
        if not events:
            return BacktestResult(
                start=None,
                end=None,
                starting_equity=self.starting_equity,
                ending_equity=equity,
                costs=self.costs,
                synthetic_data=any(bs.synthetic for bs in barsets.values()),
            )

        for timestamp, code, index in events:
            bars = frames[code]
            strategy = strategies[code]
            asset_cfg = universe.get(code, {})
            market = markets.get(code, KOSPI)
            bar = bars.iloc[index]

            # -- 1. execute an order scheduled for this bar -----------------
            order = pending.get(code)
            if order is not None and order.fill_index == index:
                pending.pop(code, None)
                equity, fill, trade = self._execute(
                    order=order,
                    bars=bars,
                    index=index,
                    positions=positions,
                    equity=equity,
                    asset_cfg=asset_cfg,
                    market=market,
                    strategy=strategy,
                    skipped=skipped,
                    halted=halted,
                )
                if fill is not None:
                    fills.append(fill)
                if trade is not None:
                    trades.append(trade)

            # -- 2. stop-outs, checked on this bar's range ------------------
            position = positions.get(code)
            if position is not None and position.stop_hit(float(bar["low"]), float(bar["high"])):
                equity, fill, trade = self._close_at(
                    code=code,
                    position=position,
                    raw_price=position.effective_stop(),
                    index=index,
                    fill_time=timestamp,
                    positions=positions,
                    equity=equity,
                    market=market,
                    kind=STOP,
                    reason="stop hit",
                    strategy_name=strategy.name,
                )
                fills.append(fill)
                trades.append(trade)
                pending.pop(code, None)
                position = None

            # -- 3. mark to market & drawdown guard -------------------------
            last_price[code] = float(bar["close"])
            mark_equity = equity + sum(
                p.unrealized(last_price[c]) for c, p in positions.items() if c in last_price
            )
            peak_equity = max(peak_equity, mark_equity)
            curve.append((timestamp, mark_equity))

            if self.apply_drawdown_guard and not halted:
                status = evaluate_drawdown(peak_equity, mark_equity, self.max_drawdown_pct)
                if status.breached:
                    halted = True
                    halt_time = timestamp
                    halt_reason = status.describe()
                    logger.warning("backtest kill switch tripped at %s - %s", timestamp, halt_reason)
                    for held, pos in list(positions.items()):
                        held_frame = frames[held]
                        held_index = max(
                            0,
                            min(
                                held_frame.index.searchsorted(timestamp, side="right") - 1,
                                len(held_frame) - 1,
                            ),
                        )
                        equity, fill, trade = self._close_at(
                            code=held,
                            position=pos,
                            raw_price=last_price.get(held, pos.entry_price),
                            index=held_index,
                            fill_time=timestamp,
                            positions=positions,
                            equity=equity,
                            market=markets.get(held, KOSPI),
                            kind=EXIT,
                            reason="kill switch",
                            strategy_name=strategies[held].name,
                        )
                        fills.append(fill)
                        trades.append(trade)
                    pending.clear()
                    continue

            if halted:
                continue

            # -- 4. trailing stop ratchet -----------------------------------
            position = positions.get(code)
            window = bars.iloc[: index + 1]
            if position is not None:
                new_trail = strategy.update_trailing_stop(window, position)
                if new_trail is not None:
                    position.trail_stop = new_trail

            # -- 5. evaluate the signal, schedule for the NEXT bar ----------
            if index + self.fill_delay_bars >= len(bars):
                continue  # no future bar to fill on: never trade the last bar
            if code in pending:
                continue  # one working order per code (duplicate-order check)

            signal = strategy.evaluate(window, position)
            if not signal.actionable:
                continue

            if signal.action.is_entry:
                if position is not None:
                    continue
                if self.long_only and signal.action is Action.ENTER_SHORT:
                    continue
                blocked, reason = theme_block(
                    code, positions, self.themes, self.theme_filter_enabled
                )
                if blocked:
                    skipped.append(f"{timestamp} {code}: {reason}")
                    continue
                cap = portfolio_capacity(
                    positions, equity, self.max_total_risk_pct, self.max_positions, last_price
                )
                if not cap.ok:
                    skipped.append(f"{timestamp} {code}: {cap.describe()}")
                    continue
            elif position is None:
                continue  # exit signal with nothing to exit

            pending[code] = _PendingOrder(
                code=code,
                action=signal.action,
                signal_index=index,
                fill_index=index + self.fill_delay_bars,
                signal_time=timestamp,
                signal_close=float(bar["close"]),
                atr=signal.atr,
                stop_distance=signal.stop_distance,
                trail_stop=signal.trail_stop,
                reason=signal.reason,
            )

        # Liquidate whatever is still open at the final bar of its own series.
        for code, position in list(positions.items()):
            bars = frames[code]
            last_index = len(bars) - 1
            equity, fill, trade = self._close_at(
                code=code,
                position=position,
                raw_price=float(bars.iloc[last_index]["close"]),
                index=last_index,
                fill_time=bars.index[last_index],
                positions=positions,
                equity=equity,
                market=markets.get(code, KOSPI),
                kind=EXIT,
                reason="end of backtest",
                strategy_name=strategies[code].name,
            )
            fills.append(fill)
            trades.append(trade)

        result = BacktestResult(
            start=events[0][0].to_pydatetime(),
            end=events[-1][0].to_pydatetime(),
            starting_equity=self.starting_equity,
            ending_equity=equity,
            equity_curve=curve,
            trades=trades,
            fills=fills,
            costs=self.costs,
            synthetic_data=any(bs.synthetic for bs in barsets.values()),
            kill_switch_tripped_at=halt_time,
            kill_switch_reason=halt_reason,
            skipped_orders=skipped,
        )
        if not result.verify_no_lookahead():
            # A hard invariant, not a warning: a violated fill rule invalidates
            # every number above it.
            raise AssertionError("look-ahead detected: a fill landed on its own signal bar")
        return result

    # -- helpers -----------------------------------------------------------

    def _build_events(
        self,
        frames: Mapping[str, pd.DataFrame],
        strategies: Mapping[str, Strategy],
    ) -> list[tuple[pd.Timestamp, str, int]]:
        events: list[tuple[pd.Timestamp, str, int]] = []
        for code, bars in frames.items():
            strategy = strategies.get(code)
            if strategy is None:
                continue
            for i in range(min(strategy.warmup, len(bars)), len(bars)):
                events.append((bars.index[i], code, i))
        events.sort(key=lambda e: (e[0], e[1]))
        return events

    def _limit_blocked(
        self, bars: pd.DataFrame, index: int, market: str, buying: bool
    ) -> str:
        """Is this bar pinned at the daily ±30% limit, where fills do not happen?"""
        if not self.respect_price_limits or index == 0:
            return ""
        previous_close = float(bars.iloc[index - 1]["close"])
        if previous_close <= 0:
            return ""
        limits = self.rules.price_limits(previous_close, market)
        open_price = float(bars.iloc[index]["open"])
        if buying and limits.blocks_buy(open_price):
            return f"limit-up at {open_price:,.0f}: no sellers"
        if not buying and limits.blocks_sell(open_price):
            return f"limit-down at {open_price:,.0f}: no buyers"
        return ""

    def _execute(
        self,
        order: _PendingOrder,
        bars: pd.DataFrame,
        index: int,
        positions: dict[str, Position],
        equity: float,
        asset_cfg: Mapping[str, Any],
        market: str,
        strategy: Strategy,
        skipped: list[str],
        halted: bool,
    ) -> tuple[float, Fill | None, ClosedTrade | None]:
        code = order.code
        fill_time = bars.index[index]
        # Rule 10: the fill reference is the *next* bar's open, not the signal
        # bar's close.
        raw_price = float(bars.iloc[index]["open"])
        position = positions.get(code)

        if order.action is Action.EXIT:
            if position is None:
                return equity, None, None
            blocked = self._limit_blocked(bars, index, market, buying=position.is_short)
            if blocked:
                skipped.append(f"{fill_time} {code}: exit skipped, {blocked}")
                return equity, None, None
            return self._close_at(
                code=code,
                position=position,
                raw_price=raw_price,
                index=index,
                fill_time=fill_time,
                positions=positions,
                equity=equity,
                market=market,
                kind=EXIT,
                reason=order.reason,
                strategy_name=strategy.name,
                signal_index=order.signal_index,
                signal_time=order.signal_time,
                signal_close=order.signal_close,
            )

        if halted or position is not None:
            return equity, None, None

        side = order.action.side
        buying = side == LONG
        blocked = self._limit_blocked(bars, index, market, buying=buying)
        if blocked:
            skipped.append(f"{fill_time} {code}: entry skipped, {blocked}")
            return equity, None, None

        fill_price = self.costs.apply_slippage(raw_price, buying)
        cap = portfolio_capacity(
            positions, equity, self.max_total_risk_pct, self.max_positions
        )
        sizing = position_size(
            equity=equity,
            atr=order.atr,
            price=fill_price,
            risk_pct=self.risk_pct,
            hard_stop_atr_mult=self.hard_stop_atr_mult,
            fractional=False,
            min_qty=float(asset_cfg.get("min_qty", 1)),
            qty_precision=0,
            max_position_notional_pct=self.max_position_notional_pct,
            risk_budget=cap.remaining_risk,
        )
        if not sizing.ok:
            skipped.append(f"{fill_time} {code}: {sizing.reason}")
            return equity, None, None

        notional = sizing.qty * fill_price
        cost = self.costs.cost(notional, "BUY" if buying else "SELL", market)
        slippage_cost = abs(fill_price - raw_price) * sizing.qty
        equity -= cost

        stop = self.rules.stop_price(fill_price, sizing.stop_distance, market)
        positions[code] = Position(
            symbol=code,
            side=side,
            qty=sizing.qty,
            entry_price=fill_price,
            stop_price=stop,
            stop_distance=sizing.stop_distance,
            strategy=strategy.name,
            entry_time=str(fill_time),
            trail_stop=order.trail_stop,
        )
        fill = Fill(
            code=code,
            kind=ENTRY,
            side=side,
            signal_index=order.signal_index,
            fill_index=index,
            signal_time=order.signal_time,
            fill_time=fill_time,
            signal_close=order.signal_close,
            price=fill_price,
            qty=sizing.qty,
            costs=cost,
            slippage_cost=slippage_cost,
            reason=order.reason,
        )
        return equity, fill, None

    def _close_at(
        self,
        code: str,
        position: Position,
        raw_price: float,
        index: int,
        fill_time: pd.Timestamp,
        positions: dict[str, Position],
        equity: float,
        market: str,
        kind: str,
        reason: str,
        strategy_name: str,
        signal_index: int = -1,
        signal_time: pd.Timestamp | None = None,
        signal_close: float = 0.0,
    ) -> tuple[float, Fill, ClosedTrade]:
        buying = position.is_short  # closing a short means buying it back
        fill_price = self.costs.apply_slippage(raw_price, buying)
        gross = position.unrealized(fill_price)
        notional = abs(fill_price * position.qty)
        # Closing a long is a SELL, which is where the transaction tax lands.
        cost = self.costs.cost(notional, "BUY" if buying else "SELL", market)
        slippage_cost = abs(fill_price - raw_price) * position.qty
        equity += gross - cost

        fill = Fill(
            code=code,
            kind=kind,
            side=position.side,
            signal_index=signal_index,
            fill_index=index,
            signal_time=signal_time,
            fill_time=fill_time,
            signal_close=signal_close,
            price=fill_price,
            qty=position.qty,
            costs=cost,
            slippage_cost=slippage_cost,
            reason=reason,
        )
        trade = ClosedTrade(
            code=code,
            side=position.side,
            qty=position.qty,
            entry_time=pd.Timestamp(position.entry_time),
            exit_time=fill_time,
            entry_price=position.entry_price,
            exit_price=fill_price,
            gross_pnl=gross,
            costs=cost + slippage_cost,
            pnl=gross - cost,
            exit_reason=reason,
            strategy=strategy_name,
        )
        positions.pop(code, None)
        return equity, fill, trade


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------


def format_result(
    result: BacktestResult,
    names: Mapping[str, str] | None = None,
    title: str = "BACKTEST RESULT",
) -> str:
    names = names or {}
    lines: list[str] = []
    width = 78
    lines.append("=" * width)
    lines.append(title.center(width))
    lines.append("=" * width)

    if result.synthetic_data:
        lines.append("")
        lines.append("!" * width)
        lines.append("  SYNTHETIC DATA - no live source and no cache were available.")
        lines.append("  These bars were generated locally. The numbers below are a")
        lines.append("  pipeline demonstration, NOT a claim about real performance.")
        lines.append("!" * width)

    lines.append("")
    lines.append(f"  Period            : {result.start} -> {result.end}")
    lines.append(f"  Starting equity   : {result.starting_equity:>16,.0f} KRW")
    lines.append(f"  Ending equity     : {result.ending_equity:>16,.0f} KRW")
    lines.append(f"  Total return      : {result.total_return_pct:>16.2%}")
    lines.append(f"  CAGR              : {result.cagr():>16.2%}")
    lines.append(f"  Max drawdown      : {result.max_drawdown_pct:>16.2%}")
    lines.append(f"  Trades            : {len(result.trades):>16d}")
    lines.append(f"  Win rate          : {result.win_rate:>16.2%}")

    lines.append("")
    lines.append("  -- Cost assumptions (config.yaml) -------------------------------------")
    lines.append(f"  {result.costs.describe()}")
    lines.append(f"  Commission + tax  : {result.total_costs:>16,.0f} KRW")
    lines.append(f"  Slippage cost     : {result.total_slippage:>16,.0f} KRW")
    lines.append(f"  Total friction    : {result.total_costs + result.total_slippage:>16,.0f} KRW")
    if result.starting_equity > 0:
        friction_pct = (result.total_costs + result.total_slippage) / result.starting_equity
        lines.append(f"  Friction / equity : {friction_pct:>16.2%}")
    lines.append("  Note: selling pays transaction tax on top of commission, so a round")
    lines.append("  trip costs materially more here than on a US venue.")

    per_code = result.per_code_pnl()
    if per_code:
        lines.append("")
        lines.append("  -- P&L by stock --------------------------------------------------------")
        for code, pnl in per_code.items():
            count = sum(1 for t in result.trades if t.code == code)
            label = f"{code} {names.get(code, '')}".strip()
            lines.append(f"  {label:<22} {pnl:>16,.0f} KRW   ({count} trades)")

    if result.kill_switch_tripped_at is not None:
        lines.append("")
        lines.append("  -- Kill switch ---------------------------------------------------------")
        lines.append(f"  Tripped at {result.kill_switch_tripped_at}: {result.kill_switch_reason}")
        lines.append("  Trading halted for the remainder of the period (safety rule 5).")

    if result.skipped_orders:
        lines.append("")
        lines.append(f"  -- Skipped orders ({len(result.skipped_orders)}) " + "-" * 42)
        for note in result.skipped_orders[:10]:
            lines.append(f"  {note}")
        if len(result.skipped_orders) > 10:
            lines.append(f"  ... and {len(result.skipped_orders) - 10} more")

    lines.append("")
    lines.append("  Fill rule: signals execute at the OPEN of the following bar")
    lines.append("  (no same-bar close fills, safety rule 10).")
    lines.append("=" * width)
    return "\n".join(lines)
