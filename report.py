#!/usr/bin/env python3
"""Morning and evening reports, with Telegram delivery.

    python report.py morning --dry-run
    python report.py evening --dry-run

Safety rule 12: Telegram sending is dry-run by default - you have to pass
``--send`` to deliver anything. With no bot token configured the report is
printed as a console preview instead of raising.

Safety rule 9: credentials never appear in a report.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from datetime import date, datetime, timedelta
from typing import Any, Mapping, Sequence

from market.calendar import KST, KrxCalendar
from portfolio import Portfolio
from risk.manager import RiskManager
from settings import (
    Credentials,
    load_config,
    load_credentials,
    load_env,
    make_console_tolerant,
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

    A missing token is *not* an error: the caller gets
    ``sent=False, reason="no credentials"`` and a console preview, which is
    what makes ``--dry-run`` usable with no bot set up.
    """
    if dry_run or not credentials.has_telegram:
        reason = "dry-run" if dry_run else "no Telegram credentials configured"
        print()
        print("=" * 60)
        print(f"  TELEGRAM PREVIEW ({reason} - nothing was sent)")
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
        # mask_text keeps the bot token out of the URL requests embeds in errors.
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
    tax = costs.get("sell_tax_bps") or {}
    return (
        f"commission {costs.get('commission_bps', 0)}bps/side · "
        f"sell tax KOSPI {tax.get('KOSPI', 0)}bps / KOSDAQ {tax.get('KOSDAQ', 0)}bps · "
        f"slippage {costs.get('slippage_bps', 0)}bps"
    )


def _label(code: str, config: Mapping[str, Any]) -> str:
    name = ((config.get("universe") or {}).get(code) or {}).get("name") or ""
    return f"{code} {name}".strip()


def _bucket_exit_reason(reason: str) -> str:
    """Group a trade's free-text exit_reason into one of a few known causes.

    Matched against the actual strings _submit_exit() is called with in
    main.py: "stop hit" (hard ATR stop) and each strategy's own Korean
    "N% 손절" for a stop, "확정 익절" for the normal lock_pct trail, "조기
    익절" for the late-session take-profit added 2026-08-27, and
    SessionRules.exit_due()'s "day-trade flat-out at HH:MM" / "planned exit
    at HH:MM ..." for a time-based force exit.
    """
    r = reason or ""
    lower = r.lower()
    if "stop hit" in lower or "손절" in r:
        return "손절"
    if "조기 익절" in r:
        return "조기 익절 (마감 임박)"
    if "확정 익절" in r:
        return "확정 익절"
    if "flat-out" in lower or "planned exit" in lower:
        return "시간 청산"
    if "kill switch" in lower:
        return "킬스위치"
    return "기타"


def _retrospective_lines(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> list[str]:
    """복기 -- always shown, even on a winning day, so problems don't hide behind a good total.

    Every losing trade is listed with its own exit_reason, so "왜 잃었는지"
    is visible without having to go dig through trades.csv by hand.
    """
    if not rows:
        return ["", DIVIDER, "복기: 청산된 거래 없음"]

    losers = sorted(
        (r for r in rows if float(r.get("pnl") or 0.0) < 0),
        key=lambda r: float(r.get("pnl") or 0.0),
    )
    lines = ["", DIVIDER, f"복기 -- 문제점 ({len(losers)}/{len(rows)}건 손실)"]
    if not losers:
        lines.append("  손실 거래 없음")
    else:
        for r in losers:
            code = str(r.get("symbol", ""))
            lines.append(
                f"  {_label(code, config):<20} {float(r.get('pnl') or 0):+,.0f} KRW  "
                f"({r.get('entry_price', '')} -> {r.get('exit_price', '')})  "
                f"-- {r.get('exit_reason', '') or '사유 미기록'}"
            )
        worst = losers[0]
        lines.append(
            f"  최대 손실: {_label(str(worst.get('symbol', '')), config)} "
            f"{float(worst.get('pnl') or 0):+,.0f} KRW"
        )

    total_fees = sum(float(r.get("fees") or 0.0) for r in rows)
    if total_fees > 0:
        lines.append(f"  수수료·세금 총액: {total_fees:,.0f} KRW")
    return lines


def build_morning_report(
    config: Mapping[str, Any],
    portfolio: Portfolio,
    risk: RiskManager,
    mode_label: str,
    calendar: KrxCalendar | None = None,
) -> str:
    state = portfolio.state
    calendar = calendar or KrxCalendar.from_config(config)
    today = datetime.now(KST).date()
    capacity = risk.capacity(state.last_equity)

    lines = [
        f"*Morning report* - {today.isoformat()}",
        f"Mode: `{mode_label}`   Status: `{state.status}`",
        f"KRX today: {'open' if calendar.is_business_day(today) else 'CLOSED'}"
        f"   (session 09:00-15:30 KST)",
        DIVIDER,
        f"Equity        : {state.last_equity:,.0f} KRW",
        f"Peak equity   : {state.peak_equity:,.0f} KRW",
        f"Drawdown      : {portfolio.drawdown_pct():.2%} (halt at {risk.max_drawdown_pct:.0%})",
        f"Risk / trade  : {risk.risk_pct:.2%} = "
        f"{risk.max_loss_per_trade(state.last_equity):,.0f} KRW",
        f"Portfolio cap : {capacity.describe()}",
    ]
    if state.stopped:
        lines += ["", f"*BOT IS STOPPED*: {state.stopped_reason}", "Run `main.py resume` to clear."]

    positions = portfolio.positions()
    lines += ["", DIVIDER, f"Open positions ({len(positions)})"]
    if positions:
        for code, pos in positions.items():
            lines.append(
                f"  {_label(code, config):<20} {pos.side:<5} "
                f"{int(pos.qty)} @ {pos.entry_price:,.0f} stop {pos.effective_stop():,.0f}"
            )
    else:
        lines.append("  none")

    lines += ["", DIVIDER, "Watchlist"]
    for code, cfg in (config.get("universe") or {}).items():
        if not cfg.get("enabled", True):
            continue
        lines.append(
            f"  {_label(str(code), config):<20} {cfg.get('strategy',''):<16} "
            f"{cfg.get('timeframe','')}"
        )

    themes = config.get("themes") or {}
    multi = {t: m for t, m in themes.items() if len(m) > 1}
    if multi:
        lines += ["", DIVIDER, "Theme groups (one position each)"]
        for theme, members in multi.items():
            lines.append(f"  {theme:<18} {', '.join(members)}")

    lines += ["", DIVIDER, f"Costs: {_cost_line(config)}"]
    if risk.long_only:
        lines.append("Long-only: short signals close positions, never open them.")
    return "\n".join(lines)


def build_evening_report(
    config: Mapping[str, Any],
    portfolio: Portfolio,
    risk: RiskManager,
    mode_label: str,
    day: date | None = None,
) -> str:
    day = day or datetime.now(KST).date()
    rows = portfolio.trades.rows_for_date(day)
    realized = sum(float(r.get("pnl") or 0.0) for r in rows)
    wins = sum(1 for r in rows if float(r.get("pnl") or 0.0) > 0)
    state = portfolio.state

    lines = [
        f"*Evening report* - {day.isoformat()}",
        f"Mode: `{mode_label}`   Status: `{state.status}`",
        DIVIDER,
        f"Trades closed : {len(rows)}",
        f"Realized P&L  : {realized:+,.0f} KRW",
        f"Win rate      : {(wins / len(rows)):.0%}" if rows else "Win rate      : n/a",
        f"Equity        : {state.last_equity:,.0f} KRW",
        f"Drawdown      : {portfolio.drawdown_pct():.2%} (halt at {risk.max_drawdown_pct:.0%})",
    ]

    if rows:
        lines += ["", DIVIDER, "Fills today"]
        for row in rows:
            code = str(row.get("symbol", ""))
            lines.append(
                f"  {_label(code, config):<20} {row.get('side',''):<5} "
                f"{row.get('qty','')} sh  "
                f"{row.get('entry_price','')} -> {row.get('exit_price','')} "
                f"= {float(row.get('pnl') or 0):+,.0f}"
            )

    daily = portfolio.daily.read_all()[-5:]
    if daily:
        lines += ["", DIVIDER, "Recent days"]
        for row in daily:
            lines.append(
                f"  {row.get('date','')}  pnl {float(row.get('realized_pnl') or 0):+,.0f}  "
                f"equity {float(row.get('ending_equity') or 0):,.0f}"
            )

    # 2026-09-02 (user request): "종가매매도 전부 보고할 수 있도록" -- a
    # close_auction entry taken tonight (15:15-15:19) does not exit until
    # tomorrow morning, so it never appeared anywhere in this report: not in
    # "Trades closed" (it has not closed) and this report had no open-
    # positions section at all, unlike the morning report. Without this, a
    # close-auction buy was invisible from the moment it was entered until
    # the next evening's report finally showed its outcome.
    positions = portfolio.positions()
    if positions:
        lines += ["", DIVIDER, f"Open positions ({len(positions)})"]
        for code, pos in positions.items():
            lines.append(
                f"  {_label(code, config):<20} {pos.side:<5} "
                f"{int(pos.qty)} @ {pos.entry_price:,.0f} stop {pos.effective_stop():,.0f}"
            )

    lines += _retrospective_lines(rows, config)

    if state.stopped:
        lines += ["", f"*BOT IS STOPPED*: {state.stopped_reason}"]

    lines += ["", DIVIDER, f"Costs: {_cost_line(config)}"]
    return "\n".join(lines)


def build_weekly_report(
    config: Mapping[str, Any],
    portfolio: Portfolio,
    risk: RiskManager,
    mode_label: str,
    week_end: date | None = None,
) -> str:
    """Every Friday: the trailing 7 calendar days rolled up.

    User-requested (2026-08-27): 매주 금요일에 일주일치를 보여줘. Reads the
    same trades.csv/daily_pnl.csv the daily evening report reads -- no new
    instrumentation, just a wider window plus a breakdown of *why* trades
    closed (see _bucket_exit_reason), which a single day's retrospective is
    too small a sample to show a pattern in.
    """
    week_end = week_end or datetime.now(KST).date()
    week_start = week_end - timedelta(days=6)

    def _in_week(day_str: str) -> bool:
        return week_start.isoformat() <= day_str[:10] <= week_end.isoformat()

    rows = [r for r in portfolio.trades.read_all() if _in_week(str(r.get("timestamp", "")))]
    realized = sum(float(r.get("pnl") or 0.0) for r in rows)
    wins = sum(1 for r in rows if float(r.get("pnl") or 0.0) > 0)
    state = portfolio.state

    lines = [
        f"*Weekly report* - {week_start.isoformat()} ~ {week_end.isoformat()}",
        f"Mode: `{mode_label}`   Status: `{state.status}`",
        DIVIDER,
        f"Trades closed : {len(rows)}",
        f"Realized P&L  : {realized:+,.0f} KRW",
        f"Win rate      : {(wins / len(rows)):.0%}" if rows else "Win rate      : n/a",
        f"Equity        : {state.last_equity:,.0f} KRW",
        f"Drawdown      : {portfolio.drawdown_pct():.2%} (halt at {risk.max_drawdown_pct:.0%})",
    ]

    daily_rows = [r for r in portfolio.daily.read_all() if _in_week(str(r.get("date", "")))]
    if daily_rows:
        lines += ["", DIVIDER, "일별 내역"]
        for row in sorted(daily_rows, key=lambda r: str(r.get("date", ""))):
            lines.append(
                f"  {row.get('date', '')}  pnl {float(row.get('realized_pnl') or 0):+,.0f}  "
                f"equity {float(row.get('ending_equity') or 0):,.0f}"
            )

    if rows:
        counts = Counter(_bucket_exit_reason(str(r.get("exit_reason", ""))) for r in rows)
        lines += ["", DIVIDER, "청산 유형별 빈도 (이번 주)"]
        for kind, n in counts.most_common():
            lines.append(f"  {kind:<20} {n}건")

    lines += _retrospective_lines(rows, config)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _load(args: argparse.Namespace):
    env = load_env()
    config = load_config(args.config)
    decision = resolve_mode(env, cli_live=False)
    credentials = load_credentials(env, decision)
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
    for name, help_text in (
        ("morning", "pre-session briefing"),
        ("evening", "end-of-day wrap with 복기 (what went wrong today)"),
        ("weekly", "trailing 7-day rollup, meant to run Fridays"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--dry-run", action="store_true", default=True, help="preview only (default)")
        p.add_argument(
            "--send", dest="dry_run", action="store_false", help="actually deliver to Telegram"
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    make_console_tolerant()
    args = build_parser().parse_args(argv)
    config, credentials, portfolio, risk, decision = _load(args)

    if args.command == "morning":
        text = build_morning_report(config, portfolio, risk, decision.label)
    elif args.command == "weekly":
        text = build_weekly_report(config, portfolio, risk, decision.label)
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
