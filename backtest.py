"""Event-driven backtester with a strict next-bar fill rule.

Safety rule 10: a signal produced on bar ``i`` can only be filled from bar
``i+1`` onwards.  This is enforced structurally, in two layers:

1. ``Strategy.evaluate`` is handed ``bars.iloc[:i+1]`` — a future bar is simply
   not reachable from inside a strategy;
2. an actionable signal schedules a *pending order* whose execution index is
   ``i+1``, and the fill price is that bar's **open** (never bar ``i``'s
   close).  Every fill records both indices and the engine asserts
   ``fill_index == signal_index + 1``.

Safety rule 11: commission and slippage come from ``config.yaml`` and are
reported alongside every result set.

All symbols share one equity curve and one event stream ordered by time, so
the correlation filter and the drawdown kill switch behave the same way they
do in live trading.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence

import pandas as pd

from data import BarSet
from portfolio import LONG, Position
from risk.manager import (
    correlation_block,
    evaluate_drawdown,
    hard_stop_price,
    position_size,
)
from strategies.base import Action, Strategy

logger = logging.getLogger("bot.backtest")

ENTRY = "entry"
EXIT = "exit"
STOP = "stop"


# --------------------------------------------------------------------------
# Costs (rule 11)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CostModel:
    """Commission and slippage assumptions, in basis points of notional."""

    equity_commission_bps: float = 0.0
    equity_slippage_bps: float = 2.0
    crypto_commission_bps: float = 25.0
    crypto_slippage_bps: float = 5.0

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CostModel":
        costs = config.get("costs") or {}
        return cls(
            equity_commission_bps=float(costs.get("equity_commission_bps", 0.0)),
            equity_slippage_bps=float(costs.get("equity_slippage_bps", 2.0)),
            crypto_commission_bps=float(costs.get("crypto_commission_bps", 25.0)),
            crypto_slippage_bps=float(costs.get("crypto_slippage_bps", 5.0)),
        )

    def commission_bps(self, asset_class: str) -> float:
        return self.crypto_commission_bps if asset_class == "crypto" else self.equity_commission_bps

    def slippage_bps(self, asset_class: str) -> float:
        return self.crypto_slippage_bps if asset_class == "crypto" else self.equity_slippage_bps

    def apply_slippage(self, price: float, buying: bool, asset_class: str) -> float:
        """Slippage always works against the order."""
        bps = self.slippage_bps(asset_class) / 10_000.0
        return price * (1.0 + bps) if buying else price * (1.0 - bps)

    def commission(self, notional: float, asset_class: str) -> float:
        return abs(notional) * self.commission_bps(asset_class) / 10_000.0

    def describe(self) -> str:
        return (
            f"equity: {self.equity_commission_bps:.1f} bps commission + "
            f"{self.equity_slippage_bps:.1f} bps slippage | "
            f"crypto: {self.crypto_commission_bps:.1f} bps commission + "
            f"{self.crypto_slippage_bps:.1f} bps slippage"
        )


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Fill:
    """A single execution, tagged with the bar that produced it."""

    symbol: str
    kind: str                 # entry / exit / stop
    side: str                 # side of the resulting or closed position
    #: Index of the bar whose close produced the signal. -1 for stop-outs.
    signal_index: int
    #: Index of the bar the order actually executed on.
    fill_index: int
    signal_time: pd.Timestamp | None
    fill_time: pd.Timestamp
    #: Close of the signal bar — recorded purely so tests can prove we did
    #: not fill there.
    signal_close: float
    price: float
    qty: float
    commission: float
    slippage_cost: float
    reason: str = ""

    @property
    def signal_driven(self) -> bool:
        return self.signal_index >= 0


@dataclass
class ClosedTrade:
    symbol: str
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
    costs: CostModel = field(default_factory=CostModel)
    synthetic_data: bool = False
    kill_switch_tripped_at: pd.Timestamp | None = None
    kill_switch_reason: str = ""
    skipped_orders: list[str] = field(default_factory=list)

    # -- metrics -----------------------------------------------------------

    @property
    def total_return_pct(self) -> float:
        if self.starting_equity <= 0:
            return 0.0
        return (self.ending_equity - self.starting_equity) / self.starting_equity

    @property
    def total_commission(self) -> float:
        return sum(f.commission for f in self.fills)

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

    def per_symbol_pnl(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for trade in self.trades:
            out[trade.symbol] = out.get(trade.symbol, 0.0) + trade.pnl
        return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))

    def verify_no_lookahead(self) -> bool:
        """Every signal-driven fill happened on the bar after its signal."""
        return all(f.fill_index == f.signal_index + 1 for f in self.fills if f.signal_driven)


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


@dataclass
class _PendingOrder:
    symbol: str
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
    """Portfolio backtest across assets with heterogeneous timeframes."""

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
        self.corr_cfg = dict(risk_cfg.get("correlation_filter") or {})
        self.costs = CostModel.from_config(config)
        self.starting_equity = float(
            starting_equity
            if starting_equity is not None
            else (config.get("account") or {}).get("starting_equity", 100_000.0)
        )
        self.apply_drawdown_guard = apply_drawdown_guard
        bt_cfg = config.get("backtest") or {}
        self.fill_delay_bars = int(bt_cfg.get("fill_delay_bars", 1))
        if self.fill_delay_bars < 1:
            raise ValueError("fill_delay_bars must be >= 1: same-bar fills are look-ahead")

    # ----------------------------------------------------------------------

    def run(
        self,
        barsets: Mapping[str, BarSet],
        strategies: Mapping[str, Strategy],
        asset_configs: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> BacktestResult:
        asset_configs = asset_configs or {}

        equity = self.starting_equity
        peak_equity = equity
        positions: dict[str, Position] = {}
        pending: dict[str, _PendingOrder] = {}
        fills: list[Fill] = []
        trades: list[ClosedTrade] = []
        curve: list[tuple[pd.Timestamp, float]] = []
        skipped: list[str] = []
        #: Latest close seen per symbol, used to mark open positions to market.
        last_price: dict[str, float] = {}
        halted = False
        halt_time: pd.Timestamp | None = None
        halt_reason = ""

        frames = {s: bs.bars for s, bs in barsets.items() if len(bs.bars) > 0}
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

        for timestamp, symbol, index in events:
            bars = frames[symbol]
            strategy = strategies[symbol]
            asset_cfg = asset_configs.get(symbol, {})
            asset_class = asset_cfg.get("asset_class", "us_equity")
            bar = bars.iloc[index]

            # -- 1. execute an order scheduled for this bar -----------------
            order = pending.get(symbol)
            if order is not None and order.fill_index == index:
                pending.pop(symbol, None)
                equity, fill, trade = self._execute(
                    order=order,
                    bars=bars,
                    index=index,
                    positions=positions,
                    equity=equity,
                    asset_cfg=asset_cfg,
                    asset_class=asset_class,
                    strategy=strategy,
                    skipped=skipped,
                    halted=halted,
                )
                if fill is not None:
                    fills.append(fill)
                if trade is not None:
                    trades.append(trade)

            # -- 2. stop-outs, checked on this bar's range ------------------
            position = positions.get(symbol)
            if position is not None and position.stop_hit(float(bar["low"]), float(bar["high"])):
                stop = position.effective_stop()
                equity, fill, trade = self._close_at(
                    symbol=symbol,
                    position=position,
                    raw_price=stop,
                    index=index,
                    fill_time=timestamp,
                    positions=positions,
                    equity=equity,
                    asset_class=asset_class,
                    kind=STOP,
                    reason="stop hit",
                    strategy_name=strategy.name,
                )
                fills.append(fill)
                trades.append(trade)
                pending.pop(symbol, None)
                position = None

            # -- 3. mark to market & drawdown guard -------------------------
            last_price[symbol] = float(bar["close"])
            mark_equity = equity + sum(
                p.unrealized(last_price[s]) for s, p in positions.items() if s in last_price
            )
            peak_equity = max(peak_equity, mark_equity)
            curve.append((timestamp, mark_equity))

            if self.apply_drawdown_guard and not halted:
                status = evaluate_drawdown(peak_equity, mark_equity, self.max_drawdown_pct)
                if status.breached:
                    halted = True
                    halt_time = timestamp
                    halt_reason = status.describe()
                    logger.warning("backtest kill switch tripped at %s — %s", timestamp, halt_reason)
                    # Flatten everything at this bar's close, then stop trading.
                    for sym, pos in list(positions.items()):
                        px = last_price.get(sym, pos.entry_price)
                        # The kill switch flattens across symbols at once, so
                        # the exit bar index is this symbol's own last bar.
                        sym_index = frames[sym].index.searchsorted(timestamp, side="right") - 1
                        sym_index = max(0, min(sym_index, len(frames[sym]) - 1))
                        equity, fill, trade = self._close_at(
                            symbol=sym,
                            position=pos,
                            raw_price=px,
                            index=sym_index,
                            fill_time=timestamp,
                            positions=positions,
                            equity=equity,
                            asset_class=asset_configs.get(sym, {}).get("asset_class", "us_equity"),
                            kind=EXIT,
                            reason="kill switch",
                            strategy_name=strategies[sym].name,
                        )
                        fills.append(fill)
                        trades.append(trade)
                    pending.clear()
                    continue

            if halted:
                continue

            # -- 4. trailing stop ratchet -----------------------------------
            position = positions.get(symbol)
            window = bars.iloc[: index + 1]
            if position is not None:
                new_trail = strategy.update_trailing_stop(window, position)
                if new_trail is not None:
                    position.trail_stop = new_trail

            # -- 5. evaluate the signal, schedule for the NEXT bar ----------
            if index + self.fill_delay_bars >= len(bars):
                continue  # no future bar to fill on: do not trade the last bar
            if symbol in pending:
                continue  # one working order per symbol (duplicate-order check)

            signal = strategy.evaluate(window, position)
            if not signal.actionable:
                continue

            if signal.action.is_entry:
                if position is not None:
                    continue
                blocked, reason = correlation_block(
                    symbol=symbol,
                    side=signal.action.side,
                    positions=positions,
                    blockers=tuple(self.corr_cfg.get("blockers", ("SPY", "QQQ"))),
                    blocked_symbol=self.corr_cfg.get("blocked_symbol", "BTC/USD"),
                    blocked_side=self.corr_cfg.get("blocked_side", LONG),
                    enabled=bool(self.corr_cfg.get("enabled", True)),
                )
                if blocked:
                    skipped.append(f"{timestamp} {symbol}: {reason}")
                    continue
            elif position is None:
                continue  # exit signal with nothing to exit

            pending[symbol] = _PendingOrder(
                symbol=symbol,
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
        for symbol, position in list(positions.items()):
            bars = frames[symbol]
            last_index = len(bars) - 1
            equity, fill, trade = self._close_at(
                symbol=symbol,
                position=position,
                raw_price=float(bars.iloc[last_index]["close"]),
                index=last_index,
                fill_time=bars.index[last_index],
                positions=positions,
                equity=equity,
                asset_class=asset_configs.get(symbol, {}).get("asset_class", "us_equity"),
                kind=EXIT,
                reason="end of backtest",
                strategy_name=strategies[symbol].name,
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
        for symbol, bars in frames.items():
            strategy = strategies.get(symbol)
            if strategy is None:
                continue
            start = min(strategy.warmup, len(bars))
            for i in range(start, len(bars)):
                events.append((bars.index[i], symbol, i))
        events.sort(key=lambda e: (e[0], e[1]))
        return events

    def _execute(
        self,
        order: _PendingOrder,
        bars: pd.DataFrame,
        index: int,
        positions: dict[str, Position],
        equity: float,
        asset_cfg: Mapping[str, Any],
        asset_class: str,
        strategy: Strategy,
        skipped: list[str],
        halted: bool,
    ) -> tuple[float, Fill | None, ClosedTrade | None]:
        symbol = order.symbol
        fill_time = bars.index[index]
        # Rule 10: the fill reference is the *next* bar's open, not the signal
        # bar's close.
        raw_price = float(bars.iloc[index]["open"])
        position = positions.get(symbol)

        if order.action is Action.EXIT:
            if position is None:
                return equity, None, None
            return self._close_at(
                symbol=symbol,
                position=position,
                raw_price=raw_price,
                index=index,
                fill_time=fill_time,
                positions=positions,
                equity=equity,
                asset_class=asset_class,
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
        fill_price = self.costs.apply_slippage(raw_price, buying, asset_class)

        sizing = position_size(
            equity=equity,
            atr=order.atr,
            price=fill_price,
            risk_pct=self.risk_pct,
            hard_stop_atr_mult=self.hard_stop_atr_mult,
            fractional=bool(asset_cfg.get("fractional", False)),
            min_qty=float(asset_cfg.get("min_qty", 1)),
            qty_precision=int(asset_cfg.get("qty_precision", 6)),
            available_cash=None,
            max_position_notional_pct=self.max_position_notional_pct,
        )
        if not sizing.ok:
            skipped.append(f"{fill_time} {symbol}: {sizing.reason}")
            return equity, None, None

        notional = sizing.qty * fill_price
        commission = self.costs.commission(notional, asset_class)
        slippage_cost = abs(fill_price - raw_price) * sizing.qty
        equity -= commission

        stop = hard_stop_price(side, fill_price, sizing.stop_distance)
        positions[symbol] = Position(
            symbol=symbol,
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
            symbol=symbol,
            kind=ENTRY,
            side=side,
            signal_index=order.signal_index,
            fill_index=index,
            signal_time=order.signal_time,
            fill_time=fill_time,
            signal_close=order.signal_close,
            price=fill_price,
            qty=sizing.qty,
            commission=commission,
            slippage_cost=slippage_cost,
            reason=order.reason,
        )
        return equity, fill, None

    def _close_at(
        self,
        symbol: str,
        position: Position,
        raw_price: float,
        index: int,
        fill_time: pd.Timestamp,
        positions: dict[str, Position],
        equity: float,
        asset_class: str,
        kind: str,
        reason: str,
        strategy_name: str,
        signal_index: int = -1,
        signal_time: pd.Timestamp | None = None,
        signal_close: float = 0.0,
    ) -> tuple[float, Fill, ClosedTrade]:
        buying = position.is_short  # closing a short means buying it back
        fill_price = self.costs.apply_slippage(raw_price, buying, asset_class)
        gross = position.unrealized(fill_price)
        notional = abs(fill_price * position.qty)
        commission = self.costs.commission(notional, asset_class)
        slippage_cost = abs(fill_price - raw_price) * position.qty
        equity += gross - commission

        fill = Fill(
            symbol=symbol,
            kind=kind,
            side=position.side,
            signal_index=signal_index,
            fill_index=index,
            signal_time=signal_time,
            fill_time=fill_time,
            signal_close=signal_close,
            price=fill_price,
            qty=position.qty,
            commission=commission,
            slippage_cost=slippage_cost,
            reason=reason,
        )
        trade = ClosedTrade(
            symbol=symbol,
            side=position.side,
            qty=position.qty,
            entry_time=pd.Timestamp(position.entry_time),
            exit_time=fill_time,
            entry_price=position.entry_price,
            exit_price=fill_price,
            gross_pnl=gross,
            costs=commission + slippage_cost,
            pnl=gross - commission,
            exit_reason=reason,
            strategy=strategy_name,
        )
        positions.pop(symbol, None)
        return equity, fill, trade


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------


def format_result(result: BacktestResult, title: str = "BACKTEST RESULT") -> str:
    lines: list[str] = []
    width = 74
    lines.append("=" * width)
    lines.append(title.center(width))
    lines.append("=" * width)

    if result.synthetic_data:
        lines.append("")
        lines.append("!" * width)
        lines.append("  SYNTHETIC DATA — no Alpaca credentials and no cache were available.".ljust(width))
        lines.append("  These bars were generated locally. The numbers below are a".ljust(width))
        lines.append("  pipeline demonstration, NOT a claim about real performance.".ljust(width))
        lines.append("!" * width)

    lines.append("")
    lines.append(f"  Period            : {result.start} -> {result.end}")
    lines.append(f"  Starting equity   : {result.starting_equity:>14,.2f}")
    lines.append(f"  Ending equity     : {result.ending_equity:>14,.2f}")
    lines.append(f"  Total return      : {result.total_return_pct:>14.2%}")
    lines.append(f"  CAGR              : {result.cagr():>14.2%}")
    lines.append(f"  Max drawdown      : {result.max_drawdown_pct:>14.2%}")
    lines.append(f"  Trades            : {len(result.trades):>14d}")
    lines.append(f"  Win rate          : {result.win_rate:>14.2%}")

    lines.append("")
    lines.append("  -- Cost assumptions (config.yaml) ---------------------------------")
    lines.append(f"  {result.costs.describe()}")
    lines.append(f"  Commission paid   : {result.total_commission:>14,.2f}")
    lines.append(f"  Slippage cost     : {result.total_slippage:>14,.2f}")
    lines.append(
        f"  Total friction    : {result.total_commission + result.total_slippage:>14,.2f}"
    )

    per_symbol = result.per_symbol_pnl()
    if per_symbol:
        lines.append("")
        lines.append("  -- P&L by symbol ---------------------------------------------------")
        for symbol, pnl in per_symbol.items():
            count = sum(1 for t in result.trades if t.symbol == symbol)
            lines.append(f"  {symbol:<10} {pnl:>14,.2f}   ({count} trades)")

    if result.kill_switch_tripped_at is not None:
        lines.append("")
        lines.append("  -- Kill switch ------------------------------------------------------")
        lines.append(f"  Tripped at {result.kill_switch_tripped_at}: {result.kill_switch_reason}")
        lines.append("  Trading halted for the remainder of the period (safety rule 5).")

    if result.skipped_orders:
        lines.append("")
        lines.append(f"  -- Skipped orders ({len(result.skipped_orders)}) ------------------------------------")
        for note in result.skipped_orders[:10]:
            lines.append(f"  {note}")
        if len(result.skipped_orders) > 10:
            lines.append(f"  ... and {len(result.skipped_orders) - 10} more")

    lines.append("")
    lines.append("  Fill rule: signals are executed at the OPEN of the following bar")
    lines.append("  (no same-bar close fills, safety rule 10).")
    lines.append("=" * width)
    return "\n".join(lines)
