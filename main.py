#!/usr/bin/env python3
"""Alpaca 5-asset trading bot — command line entry point.

    python main.py backtest --months 6
    python main.py paper --dry-run
    python main.py paper --once
    python main.py live --once
    python main.py status
    python main.py stop --close-all
    python main.py resume

Paper trading is the default and the fallback.  Live trading needs three
independent confirmations (see ``settings.resolve_mode``); miss any one of them
and the run is demoted to paper with the reason printed.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from broker import BrokerBase, BrokerError, DryRunBroker, build_broker
from data import MarketData, months_to_start, timeframe_delta
from portfolio import LONG, SHORT, Portfolio, Position
from risk.manager import RiskManager, TradeContext, hard_stop_price
from settings import (
    Credentials,
    ModeDecision,
    load_config,
    load_credentials,
    load_env,
    resolve_mode,
    setup_logging,
)
from strategies import build_strategies
from strategies.base import Action, Strategy

logger = logging.getLogger("bot.main")

RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

EQUITY_CLASSES = ("us_equity", "etf")


# --------------------------------------------------------------------------
# Console helpers
# --------------------------------------------------------------------------


def _supports_color() -> bool:
    return sys.stdout.isatty()


def _c(text: str, *codes: str) -> str:
    if not _supports_color():
        return text
    return "".join(codes) + text + RESET


def print_mode_notice(decision: ModeDecision) -> None:
    """Always tell the operator which account this run will touch."""
    if decision.live:
        return  # the full banner is printed separately
    if decision.demoted:
        print(_c("=" * 74, YELLOW))
        print(_c("  LIVE TRADING WAS REQUESTED BUT DEMOTED TO PAPER", BOLD, YELLOW))
        for reason in decision.reasons:
            print(_c(f"    - {reason}", YELLOW))
        print(_c(f"  Running against {decision.endpoint}", YELLOW))
        print(_c("=" * 74, YELLOW))
    else:
        print(f"[PAPER] endpoint: {decision.endpoint}")


def print_live_banner(
    account_equity: float,
    max_loss_per_trade: float,
    endpoint: str,
    countdown_seconds: int = 10,
    sleep=time.sleep,
) -> None:
    """Safety rule 3: warning, balance, worst-case loss, then a countdown."""
    bar = "!" * 74
    print(_c(bar, BOLD, RED))
    print(_c("  *** LIVE TRADING — REAL MONEY IS AT RISK ***".center(74), BOLD, RED))
    print(_c(bar, BOLD, RED))
    print(_c(f"  Endpoint              : {endpoint}", RED))
    print(_c(f"  Account equity        : {account_equity:,.2f}", RED))
    print(_c(f"  Max loss per trade    : {max_loss_per_trade:,.2f}", RED))
    print(_c("  Every order carries a hard stop at that amount.", RED))
    print(_c(bar, BOLD, RED))
    print(_c(f"  Starting in {countdown_seconds} seconds — press Ctrl-C to abort.", BOLD, YELLOW))
    for remaining in range(countdown_seconds, 0, -1):
        sys.stdout.write(_c(f"\r  {remaining:>2d} ...  ", YELLOW))
        sys.stdout.flush()
        sleep(1)
    sys.stdout.write("\r" + " " * 30 + "\r")
    sys.stdout.flush()
    print(_c("  Live trading started.", BOLD, RED))


# --------------------------------------------------------------------------
# Runtime container
# --------------------------------------------------------------------------


@dataclass
class Runtime:
    config: Mapping[str, Any]
    decision: ModeDecision
    credentials: Credentials
    broker: BrokerBase
    portfolio: Portfolio
    risk: RiskManager
    market_data: MarketData
    strategies: dict[str, Strategy]

    @property
    def assets(self) -> Mapping[str, Mapping[str, Any]]:
        return self.config.get("assets") or {}


def build_runtime(
    args: argparse.Namespace,
    cli_live: bool,
    force_dry_run: bool = False,
) -> Runtime:
    env = load_env()
    config = load_config(getattr(args, "config", None))
    decision = resolve_mode(env, cli_live=cli_live)
    paths = config.get("paths") or {}

    setup_logging(decision, log_file=paths.get("log_file", "logs/bot.log"))
    print_mode_notice(decision)
    if decision.demoted:
        logger.warning("demoted to paper: %s", "; ".join(decision.reasons))

    credentials = load_credentials(env)
    broker = build_broker(decision, credentials, config, force_dry_run=force_dry_run)
    portfolio = Portfolio(
        state_path=paths.get("state_file", "state.json"),
        trades_path=paths.get("trades_csv", "trades.csv"),
        daily_path=paths.get("daily_pnl_csv", "daily_pnl.csv"),
        mode_label=decision.label,
    )
    return Runtime(
        config=config,
        decision=decision,
        credentials=credentials,
        broker=broker,
        portfolio=portfolio,
        risk=RiskManager(config, portfolio),
        market_data=MarketData(credentials, cache_dir=paths.get("cache_dir", "data/cache")),
        strategies=build_strategies(config),
    )


# --------------------------------------------------------------------------
# Trading engine
# --------------------------------------------------------------------------


class TradingEngine:
    """One signal-check cycle per strategy cadence, plus the outer loop."""

    def __init__(self, rt: Runtime) -> None:
        self.rt = rt
        schedule = rt.config.get("schedule") or {}
        self.intervals: dict[str, int] = dict(schedule.get("intervals") or {})
        self.tick_seconds = int(schedule.get("tick_seconds", 30))
        self.closed_sleep = int(schedule.get("closed_market_sleep_seconds", 300))
        self.equity_session_only = bool(schedule.get("equity_session_only", True))
        self._next_due: dict[str, float] = {}
        self._closed_logged = False

    # -- scheduling --------------------------------------------------------

    def interval_for(self, symbol: str) -> int:
        timeframe = self.rt.assets.get(symbol, {}).get("timeframe", "1Hour")
        if timeframe in self.intervals:
            return int(self.intervals[timeframe])
        return int(timeframe_delta(timeframe).total_seconds())

    def due_symbols(self, now: float) -> list[str]:
        due = []
        for symbol in self.rt.strategies:
            if now >= self._next_due.get(symbol, 0.0):
                due.append(symbol)
        return due

    def mark_ran(self, symbol: str, now: float) -> None:
        self._next_due[symbol] = now + self.interval_for(symbol)

    # -- one cycle ---------------------------------------------------------

    def run_cycle(self, symbols: Sequence[str] | None = None) -> None:
        rt = self.rt
        try:
            account = rt.broker.get_account()
        except BrokerError as exc:
            logger.error("account fetch failed, skipping this cycle: %s", exc)
            return

        rt.portfolio.mark_equity(account.equity)

        # Rule 5 / 6 — the kill switch is evaluated before anything else.
        status = rt.risk.check_drawdown(account.equity)
        if status.breached and not rt.portfolio.stopped:
            rt.risk.trip_kill_switch(status, rt.broker)
        if rt.portfolio.stopped:
            logger.warning(
                "bot is STOPPED (%s) — no new orders. Run `python main.py resume` to clear.",
                rt.portfolio.state.stopped_reason,
            )
            return

        try:
            market_open = rt.broker.is_market_open()
        except BrokerError as exc:
            logger.error("clock fetch failed, assuming market closed: %s", exc)
            market_open = False

        try:
            open_orders = rt.broker.open_order_symbols()
        except BrokerError as exc:
            logger.error("open-order fetch failed, assuming none: %s", exc)
            open_orders = []

        for symbol in symbols if symbols is not None else list(rt.strategies):
            try:
                self._process_symbol(symbol, account, market_open, open_orders)
            except BrokerError as exc:
                logger.error("%s: broker error, skipping: %s", symbol, exc)
            except Exception as exc:  # never let one symbol kill the loop
                logger.exception("%s: unexpected error, skipping: %s", symbol, exc)

    def _process_symbol(
        self,
        symbol: str,
        account,
        market_open: bool,
        open_orders: Sequence[str],
    ) -> None:
        rt = self.rt
        asset_cfg = rt.assets.get(symbol, {})
        asset_class = asset_cfg.get("asset_class", "us_equity")
        strategy = rt.strategies[symbol]

        # Rule 8: equity ETFs only during the US session; crypto is 24/7.
        if asset_class != "crypto" and self.equity_session_only and not market_open:
            logger.info("%s: US market closed, skipping", symbol)
            return

        barset = rt.market_data.get_bars(
            symbol=symbol,
            timeframe=asset_cfg.get("timeframe", "1Hour"),
            lookback_bars=max(strategy.warmup + 50, 260),
            asset_class=asset_class,
        )
        bars = barset.bars
        if len(bars) < strategy.warmup:
            logger.info("%s: only %d bars, need %d", symbol, len(bars), strategy.warmup)
            return
        if barset.synthetic:
            logger.warning("%s: SYNTHETIC bars in use — signals are illustrative only", symbol)

        position = rt.portfolio.get(symbol)

        # Ratchet the trailing stop before evaluating anything else.
        if position is not None:
            new_trail = strategy.update_trailing_stop(bars, position)
            if new_trail is not None and new_trail != position.trail_stop:
                position.trail_stop = new_trail
                rt.portfolio.update_position(position)
                logger.info("%s: trailing stop moved to %.4f", symbol, new_trail)

        signal = strategy.evaluate(bars, position)
        price = float(bars["close"].iloc[-1])
        logger.info(
            "%s [%s] %s — %s (close=%.4f)",
            symbol,
            asset_cfg.get("timeframe", "?"),
            signal.action.value,
            signal.reason,
            price,
        )

        if not signal.actionable:
            return

        if signal.action is Action.EXIT:
            self._submit_exit(symbol, position, price, signal.reason, open_orders, asset_cfg)
            return

        self._submit_entry(
            symbol=symbol,
            signal=signal,
            price=price,
            account=account,
            market_open=market_open,
            open_orders=open_orders,
            asset_cfg=asset_cfg,
            asset_class=asset_class,
            strategy=strategy,
        )

    # -- order paths -------------------------------------------------------

    def _submit_exit(
        self,
        symbol: str,
        position: Position | None,
        price: float,
        reason: str,
        open_orders: Sequence[str],
        asset_cfg: Mapping[str, Any],
    ) -> None:
        if position is None:
            return
        rt = self.rt
        ctx = TradeContext(
            symbol=symbol,
            side=SHORT if position.is_long else LONG,
            qty=position.qty,
            price=price,
            asset_class=asset_cfg.get("asset_class", "us_equity"),
            tradable=rt.broker.get_asset(symbol).tradable,
            market_open=rt.broker.is_market_open(),
            min_qty=float(asset_cfg.get("min_qty", 1)),
            open_order_symbols=open_orders,
            existing_position=position,
            available_cash=0.0,
            is_exit=True,
        )
        checks = rt.risk.pre_trade_checks(ctx)
        if not checks.passed:
            logger.warning("%s: exit blocked — %s", symbol, checks.describe())
            return
        result = rt.broker.submit_order(
            symbol=symbol,
            side=ctx.side,
            qty=position.qty,
            is_exit=True,
            note=f"exit: {reason}",
        )
        if result.submitted:
            rt.portfolio.close_position(symbol, exit_price=price, exit_reason=reason)
        else:
            logger.info("%s: exit simulated only — position left untouched in state", symbol)

    def _submit_entry(
        self,
        symbol: str,
        signal,
        price: float,
        account,
        market_open: bool,
        open_orders: Sequence[str],
        asset_cfg: Mapping[str, Any],
        asset_class: str,
        strategy: Strategy,
    ) -> None:
        rt = self.rt
        side = signal.action.side

        allowed, reason = rt.risk.can_open_new(symbol, side)
        if not allowed:
            logger.warning("%s: entry blocked — %s", symbol, reason)
            return

        sizing = rt.risk.size(
            symbol=symbol,
            equity=account.equity,
            atr=signal.atr,
            price=price,
            asset_cfg=asset_cfg,
            available_cash=account.cash,
        )
        if not sizing.ok:
            # Rule 4: a non-positive stop distance means no order at all.
            logger.warning("%s: order skipped — %s", symbol, sizing.reason)
            return

        ctx = TradeContext(
            symbol=symbol,
            side=side,
            qty=sizing.qty,
            price=price,
            asset_class=asset_class,
            tradable=rt.broker.get_asset(symbol).tradable,
            market_open=market_open,
            min_qty=float(asset_cfg.get("min_qty", 1)),
            open_order_symbols=open_orders,
            existing_position=rt.portfolio.get(symbol),
            available_cash=account.cash,
        )
        checks = rt.risk.pre_trade_checks(ctx)
        if not checks.passed:
            logger.warning("%s: entry blocked — %s", symbol, checks.describe())
            return

        stop = hard_stop_price(side, price, sizing.stop_distance)
        logger.info(
            "%s: %s qty=%s risk=%.2f (%.2f%% of equity) stop=%.4f",
            symbol,
            side,
            sizing.qty,
            sizing.realised_risk(),
            100 * sizing.realised_risk() / account.equity if account.equity else 0.0,
            stop,
        )
        result = rt.broker.submit_order(
            symbol=symbol,
            side=side,
            qty=sizing.qty,
            stop_price=stop,
            note=signal.reason,
        )
        if result.submitted:
            rt.portfolio.open_position(
                Position(
                    symbol=symbol,
                    side=side,
                    qty=sizing.qty,
                    entry_price=price,
                    stop_price=stop,
                    stop_distance=sizing.stop_distance,
                    strategy=strategy.name,
                    trail_stop=signal.trail_stop,
                )
            )
        else:
            logger.info(
                "%s: order simulated only (dry run) — nothing sent, state unchanged", symbol
            )

    # -- loops -------------------------------------------------------------

    def run_once(self) -> None:
        self.run_cycle()
        self.rt.portfolio.record_day(self.rt.portfolio.state.last_equity)

    def run_forever(self) -> None:
        logger.info(
            "entering continuous loop (tick=%ds); per-symbol cadence: %s",
            self.tick_seconds,
            {s: self.interval_for(s) for s in self.rt.strategies},
        )
        try:
            while True:
                now = time.time()
                due = self.due_symbols(now)
                if due:
                    self.run_cycle(due)
                    for symbol in due:
                        self.mark_ran(symbol, now)
                if self.rt.portfolio.stopped:
                    logger.error(
                        "STOPPED state reached — exiting loop. Run `python main.py resume`."
                    )
                    return
                time.sleep(self.tick_seconds)
        except KeyboardInterrupt:
            logger.info("interrupted by user — shutting down cleanly")
            self.rt.portfolio.record_day(self.rt.portfolio.state.last_equity)


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_backtest(args: argparse.Namespace) -> int:
    from backtest import Backtester, format_result

    rt = build_runtime(args, cli_live=False, force_dry_run=True)
    end = datetime.now(timezone.utc)
    start = months_to_start(args.months, end)
    print(f"\nLoading {args.months} month(s) of bars ({start.date()} -> {end.date()}) ...")

    barsets = {}
    for symbol, asset_cfg in rt.assets.items():
        if not asset_cfg.get("enabled", True):
            continue
        barset = rt.market_data.get_bars(
            symbol=symbol,
            timeframe=asset_cfg.get("timeframe", "1Hour"),
            start=start,
            end=end,
            asset_class=asset_cfg.get("asset_class", "us_equity"),
        )
        barsets[symbol] = barset
        print(f"  {symbol:<10} {len(barset):>6d} bars  [{barset.source}]")

    engine = Backtester(rt.config)
    result = engine.run(barsets, rt.strategies, rt.assets)
    print()
    print(format_result(result))
    return 0


def cmd_trade(args: argparse.Namespace, cli_live: bool) -> int:
    force_dry_run = bool(getattr(args, "dry_run", False))
    rt = build_runtime(args, cli_live=cli_live, force_dry_run=force_dry_run)

    if rt.decision.live:
        try:
            account = rt.broker.get_account()
        except BrokerError as exc:
            print(f"cannot read the live account, aborting: {exc}")
            return 1
        print_live_banner(
            account_equity=account.equity,
            max_loss_per_trade=rt.risk.max_loss_per_trade(account.equity),
            endpoint=rt.decision.endpoint,
            countdown_seconds=int((rt.config.get("live_banner") or {}).get("countdown_seconds", 10)),
        )

    if rt.portfolio.stopped:
        print(
            f"\nBot is STOPPED: {rt.portfolio.state.stopped_reason}\n"
            "Run `python main.py resume` to clear it.\n"
        )
        return 1

    if isinstance(rt.broker, DryRunBroker):
        print(
            f"\n[{rt.broker.label}] no orders will be sent; "
            "intended orders are printed and logged only.\n"
        )

    engine = TradingEngine(rt)
    if getattr(args, "once", False) or force_dry_run:
        engine.run_once()
    else:
        engine.run_forever()
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    rt = build_runtime(args, cli_live=False, force_dry_run=True)
    state = rt.portfolio.state
    print()
    print("=" * 74)
    print("  BOT STATUS".center(74))
    print("=" * 74)
    print(f"  Status            : {state.status}")
    if state.stopped:
        print(f"  Stopped reason    : {state.stopped_reason}")
        print(f"  Stopped at        : {state.stopped_at}")
    print(f"  Mode              : {rt.decision.label}  ({rt.decision.endpoint})")
    print(f"  Last equity       : {state.last_equity:,.2f}")
    print(f"  Peak equity       : {state.peak_equity:,.2f}")
    print(f"  Drawdown          : {rt.portfolio.drawdown_pct():.2%} "
          f"(limit {rt.risk.max_drawdown_pct:.2%})")
    print(f"  Risk per trade    : {rt.risk.risk_pct:.2%} "
          f"= {rt.risk.max_loss_per_trade(state.last_equity):,.2f}")
    print(f"  Last run          : {state.last_run_at or 'never'}")

    positions = rt.portfolio.positions()
    print(f"\n  Open positions    : {len(positions)}")
    for symbol, pos in positions.items():
        print(
            f"    {symbol:<10} {pos.side:<5} qty={pos.qty:<12} "
            f"entry={pos.entry_price:<12.4f} stop={pos.effective_stop():.4f}"
        )

    rows = rt.portfolio.trades.read_all()
    print(f"\n  Recorded trades   : {len(rows)}  ({rt.portfolio.trades.path})")
    for row in rows[-5:]:
        print(
            f"    {row.get('timestamp','')} {row.get('symbol',''):<9} "
            f"{row.get('side',''):<5} pnl={row.get('pnl','')}"
        )
    print("=" * 74)
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    rt = build_runtime(args, cli_live=False, force_dry_run=not args.close_all)
    reason = "manual stop requested by operator"
    rt.portfolio.stop(reason)
    print(f"\nBot state set to STOPPED ({rt.portfolio.store.path}).")
    if args.close_all:
        try:
            cancelled = rt.broker.cancel_all_orders()
            closed = rt.broker.close_all_positions()
            print(f"Cancelled {cancelled} order(s), closed {closed} position(s).")
        except BrokerError as exc:
            print(f"Broker teardown failed: {exc}")
            return 1
        for symbol in list(rt.portfolio.state.positions):
            rt.portfolio.state.positions.pop(symbol, None)
        rt.portfolio.save()
    print("It will stay STOPPED until you run `python main.py resume`.")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    rt = build_runtime(args, cli_live=False, force_dry_run=True)
    if not rt.portfolio.stopped:
        print("\nBot is already RUNNING; nothing to do.")
        return 0
    previous = rt.portfolio.state.stopped_reason
    rt.portfolio.resume()
    print(f"\nCleared STOPPED state (was: {previous}).")
    print(f"Drawdown peak re-anchored to {rt.portfolio.state.peak_equity:,.2f}.")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Alpaca 5-asset trading bot (paper by default)",
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="run a historical backtest")
    bt.add_argument("--months", type=int, default=6, help="months of history (default: 6)")

    paper = sub.add_parser("paper", help="trade the paper account")
    paper.add_argument("--dry-run", action="store_true", help="print orders, send nothing")
    paper.add_argument("--once", action="store_true", help="run a single cycle and exit")
    paper.add_argument(
        "--live", action="store_true", help="request live trading (still needs both env vars)"
    )

    live = sub.add_parser("live", help="trade the live account (requires triple confirmation)")
    live.add_argument("--once", action="store_true", help="run a single cycle and exit")
    live.add_argument("--dry-run", action="store_true", help="print orders, send nothing")
    live.add_argument(
        "--live", action="store_true", help="explicit live flag (the sub-command implies it)"
    )

    sub.add_parser("status", help="show state, positions and drawdown")

    stop = sub.add_parser("stop", help="halt trading and persist STOPPED")
    stop.add_argument(
        "--close-all", action="store_true", help="also cancel orders and flatten positions"
    )

    sub.add_parser("resume", help="clear STOPPED and allow trading again")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "backtest":
        return cmd_backtest(args)
    if args.command == "paper":
        # `paper --live` is still an explicit live request; the env vars decide.
        return cmd_trade(args, cli_live=bool(args.live))
    if args.command == "live":
        # The `live` sub-command is itself the explicit command-line request.
        return cmd_trade(args, cli_live=True)
    if args.command == "status":
        return cmd_status(args)
    if args.command == "stop":
        return cmd_stop(args)
    if args.command == "resume":
        return cmd_resume(args)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
