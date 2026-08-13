#!/usr/bin/env python3
"""Morning and evening reports, with Telegram delivery.

    python report.py morning --dry-run
    python report.py evening --dry-run

Safety rule 12: Telegram sending is dry-run by default — you have to pass
``--send`` to actually deliver anything.  With no bot token configured the
report is printed as a console preview instead of raising.

Safety rule 9: credentials never appear in a report.  The report renders the
token as ``***REDACTED***`` (or "not configured") and nothing else.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from portfolio import Portfolio
from risk.manager import RiskManager
from settings import (
    Credentials,
    load_config,
    load_credentials,
    load_env,
    mask_text,
    resolve_mode,
)

logger = logging.getLogger("bot.report")

DIVIDER = "-" * 46


# --------------------------------------------------------------------------
# Telegram
# --------------------------------------------------------------------------


def send_telegram(
    text: str,
    credentials: Credentials,
    dry_run: bool = True,
    parse_mode: str = "Markdown",
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Deliver *text* to Telegram, or preview it.

    Returns a small result dict describing what happened.  A missing token is
    *not* an error: the caller gets ``sent=False, reason="no credentials"`` and
    a console preview, which is what makes ``--dry-run`` usable on a laptop
    with no bot set up.
    """
    if dry_run or not credentials.has_telegram:
        reason = "dry-run" if dry_run else "no Telegram credentials configured"
        print()
        print("=" * 60)
        print(f"  TELEGRAM PREVIEW ({reason} — nothing was sent)")
        print("=" * 60)
        print(text)
        print("=" * 60)
        return {"sent": False, "reason": reason, "preview": text}

    try:
        import requests

        response = requests.post(
            f"https://api.telegram.org/bot{credentials.telegram_token.reveal()}/sendMessage",
            json={
                "chat_id": credentials.telegram_chat_id.reveal(),
                "text": text,
                "parse_mode": parse_mode,
            },
            timeout=timeout,
        )
        response.raise_for_status()
    except Exception as exc:
        # mask_text guarantees the bot token cannot leak through the URL that
        # requests embeds in its error messages.
        message = mask_text(str(exc))
        logger.error("Telegram send failed: %s", message)
        print(f"\nTelegram send failed ({message}). Message preview:\n\n{text}\n")
        return {"sent": False, "reason": message, "preview": text}

    logger.info("Telegram message delivered (%d chars)", len(text))
    return {"sent": True, "reason": "", "preview": text}


# --------------------------------------------------------------------------
# Report bodies
# --------------------------------------------------------------------------


def _cost_line(config: Mapping[str, Any]) -> str:
    costs = config.get("costs") or {}
    return (
        f"equity {costs.get('equity_commission_bps', 0)}bps comm / "
        f"{costs.get('equity_slippage_bps', 0)}bps slip · "
        f"crypto {costs.get('crypto_commission_bps', 0)}bps comm / "
        f"{costs.get('crypto_slippage_bps', 0)}bps slip"
    )


def build_morning_report(
    config: Mapping[str, Any],
    portfolio: Portfolio,
    risk: RiskManager,
    mode_label: str,
) -> str:
    state = portfolio.state
    lines = [
        f"*Morning report* — {date.today().isoformat()}",
        f"Mode: `{mode_label}`   Status: `{state.status}`",
        DIVIDER,
        f"Equity        : {state.last_equity:,.2f}",
        f"Peak equity   : {state.peak_equity:,.2f}",
        f"Drawdown      : {portfolio.drawdown_pct():.2%} (halt at {risk.max_drawdown_pct:.0%})",
        f"Risk per trade: {risk.risk_pct:.2%} = {risk.max_loss_per_trade(state.last_equity):,.2f}",
    ]
    if state.stopped:
        lines += ["", f"*BOT IS STOPPED*: {state.stopped_reason}", "Run `main.py resume` to clear."]

    positions = portfolio.positions()
    lines += ["", DIVIDER, f"Open positions ({len(positions)})"]
    if positions:
        for symbol, pos in positions.items():
            lines.append(
                f"  {symbol:<9} {pos.side:<5} qty {pos.qty} @ {pos.entry_price:,.4f} "
                f"stop {pos.effective_stop():,.4f}"
            )
    else:
        lines.append("  none")

    lines += ["", DIVIDER, "Today's watchlist"]
    for symbol, cfg in (config.get("assets") or {}).items():
        if not cfg.get("enabled", True):
            continue
        session = "24/7" if cfg.get("asset_class") == "crypto" else "US session"
        lines.append(
            f"  {symbol:<9} {cfg.get('strategy',''):<16} {cfg.get('timeframe',''):<7} {session}"
        )

    lines += ["", DIVIDER, f"Cost assumptions: {_cost_line(config)}"]
    return "\n".join(lines)


def build_evening_report(
    config: Mapping[str, Any],
    portfolio: Portfolio,
    risk: RiskManager,
    mode_label: str,
    day: date | None = None,
) -> str:
    day = day or datetime.now(timezone.utc).date()
    rows = portfolio.trades.rows_for_date(day)
    realized = sum(float(r.get("pnl") or 0.0) for r in rows)
    wins = sum(1 for r in rows if float(r.get("pnl") or 0.0) > 0)
    state = portfolio.state

    lines = [
        f"*Evening report* — {day.isoformat()}",
        f"Mode: `{mode_label}`   Status: `{state.status}`",
        DIVIDER,
        f"Trades closed : {len(rows)}",
        f"Realized P&L  : {realized:+,.2f}",
        f"Win rate      : {(wins / len(rows)):.0%}" if rows else "Win rate      : n/a",
        f"Equity        : {state.last_equity:,.2f}",
        f"Drawdown      : {portfolio.drawdown_pct():.2%} (halt at {risk.max_drawdown_pct:.0%})",
    ]

    if rows:
        lines += ["", DIVIDER, "Fills today"]
        for row in rows:
            lines.append(
                f"  {row.get('symbol',''):<9} {row.get('side',''):<5} "
                f"qty {row.get('qty','')} "
                f"{row.get('entry_price','')} -> {row.get('exit_price','')} "
                f"= {float(row.get('pnl') or 0):+,.2f}"
            )

    daily = portfolio.daily.read_all()[-5:]
    if daily:
        lines += ["", DIVIDER, "Recent days"]
        for row in daily:
            lines.append(
                f"  {row.get('date','')}  pnl {float(row.get('realized_pnl') or 0):+,.2f}  "
                f"equity {float(row.get('ending_equity') or 0):,.2f}"
            )

    if state.stopped:
        lines += ["", f"*BOT IS STOPPED*: {state.stopped_reason}"]

    lines += ["", DIVIDER, f"Cost assumptions: {_cost_line(config)}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _load(args: argparse.Namespace):
    env = load_env()
    config = load_config(args.config)
    decision = resolve_mode(env, cli_live=False)
    credentials = load_credentials(env)
    paths = config.get("paths") or {}
    portfolio = Portfolio(
        state_path=paths.get("state_file", "state.json"),
        trades_path=paths.get("trades_csv", "trades.csv"),
        daily_path=paths.get("daily_pnl_csv", "daily_pnl.csv"),
        mode_label=decision.label,
    )
    return config, credentials, portfolio, RiskManager(config, portfolio), decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="report.py", description="Morning / evening reports (Telegram dry-run by default)"
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("morning", "pre-session briefing"), ("evening", "end-of-day wrap")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument(
            "--dry-run",
            action="store_true",
            default=True,
            help="preview only (default)",
        )
        p.add_argument(
            "--send",
            dest="dry_run",
            action="store_false",
            help="actually deliver to Telegram",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config, credentials, portfolio, risk, decision = _load(args)

    if args.command == "morning":
        text = build_morning_report(config, portfolio, risk, decision.label)
    else:
        text = build_evening_report(config, portfolio, risk, decision.label)

    telegram_cfg = config.get("telegram") or {}
    dry_run = bool(args.dry_run or telegram_cfg.get("dry_run", True))
    result = send_telegram(
        text,
        credentials,
        dry_run=dry_run,
        parse_mode=telegram_cfg.get("parse_mode", "Markdown"),
    )
    if not result["sent"]:
        print(f"\n(not sent: {result['reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
