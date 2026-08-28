#!/usr/bin/env python3
"""KRX 8-stock trading bot - command line entry point.

    python main.py profile
    python main.py backtest --months 6
    python main.py paper --dry-run
    python main.py paper --once
    python main.py live --once
    python main.py status
    python main.py stop --close-all
    python main.py resume

Paper trading is the default and the fallback. Live trading needs three
independent confirmations (see ``settings.resolve_mode``); miss any one and the
run is demoted to paper with the reason printed.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Any, Mapping, Sequence

import pandas as pd

from broker import BrokerAuthError, BrokerBase, BrokerError, DryRunBroker, build_broker
from data import MarketData, months_to_start, timeframe_delta, warmup_start
from indicators import atr as atr_indicator
from indicators import last_valid
from indicators import rolling_mean_volume
from market.calendar import KST, REGULAR_CLOSE, KrxCalendar
from market.rules import KOSDAQ, KOSPI, KrxRules
from market.session_rules import SessionRules, parse_clock
from portfolio import LONG, SHORT, Portfolio, Position
from risk.manager import RiskManager, TradeContext
from settings import (
    LIVE_CONFIRM_TOKEN,
    Credentials,
    ModeDecision,
    load_config,
    load_credentials,
    load_env,
    make_console_tolerant,
    mask_text,
    resolve_mode,
    setup_logging,
)
from strategies import build_strategy, build_strategies
from strategies.base import Action, Strategy
from strategies.orb import ORB
from strategies.pullback import PullbackBounce
from telegram_bot import TelegramCommandHandler, TelegramNotifier, build_telegram
from investor_flow import InvestorFlowScanner
from us_market import USMarketMonitor
from dart_monitor import DartMonitor
from news_monitor import NewsMonitor
from daily_trend import DailyTrendScanner
from macro_gauge import MacroGauge

logger = logging.getLogger("bot.main")

RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


# --------------------------------------------------------------------------
# Console helpers
# --------------------------------------------------------------------------


def _c(text: str, *codes: str) -> str:
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + RESET


def print_mode_notice(decision: ModeDecision) -> None:
    """Always tell the operator which account this run will touch."""
    if decision.live:
        return  # the full banner is printed separately
    if decision.demoted:
        print(_c("=" * 78, YELLOW))
        print(_c("  LIVE TRADING WAS REQUESTED BUT DEMOTED TO PAPER", BOLD, YELLOW))
        for reason in decision.reasons:
            print(_c(f"    - {reason}", YELLOW))
        print(_c(f"  Running against {decision.endpoint}", YELLOW))
        print(_c("=" * 78, YELLOW))
    else:
        print(f"[PAPER] endpoint: {decision.endpoint}")


def print_live_banner(
    account_equity: float,
    max_loss_per_trade: float,
    max_portfolio_loss: float,
    endpoint: str,
    countdown_seconds: int = 10,
    sleep=time.sleep,
) -> None:
    """Safety rule 3: warning, balance, worst-case loss, then a countdown."""
    bar = "!" * 78
    print(_c(bar, BOLD, RED))
    print(_c("  *** LIVE TRADING - REAL MONEY IS AT RISK ***".center(78), BOLD, RED))
    print(_c(bar, BOLD, RED))
    print(_c(f"  Endpoint                  : {endpoint}", RED))
    print(_c(f"  Account equity            : {account_equity:>16,.0f} KRW", RED))
    print(_c(f"  Max loss per trade        : {max_loss_per_trade:>16,.0f} KRW", RED))
    print(_c(f"  Max loss, whole portfolio : {max_portfolio_loss:>16,.0f} KRW", RED))
    print(_c("  Every order carries a hard stop at the per-trade amount.", RED))
    print(_c(bar, BOLD, RED))
    print(_c(f"  Starting in {countdown_seconds} seconds - press Ctrl-C to abort.", BOLD, YELLOW))
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
    calendar: KrxCalendar
    rules: KrxRules

    @property
    def universe(self) -> Mapping[str, Mapping[str, Any]]:
        return {str(k): v for k, v in (self.config.get("universe") or {}).items()}

    def name_of(self, code: str) -> str:
        return str(self.universe.get(code, {}).get("name") or "")

    def market_of(self, code: str) -> str:
        return str(self.universe.get(code, {}).get("market") or KOSPI)


def _fallback_static_stocks(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Disabled *pullback_bounce* universe entries, re-enabled as a last resort.

    Used only when the screener itself fails to fetch data (KRX/pykrx
    unreachable) so the bot isn't left with zero stocks to trade for the
    whole session. Never mutates config.yaml -- purely a runtime fallback.

    Deliberately scoped to ``strategy: pullback_bounce`` only -- that is
    exactly the tag on the five originally-static, day-trade, affordable
    core stocks. Other disabled entries were turned off for a reason that
    still applies: ``close_auction`` stocks carry overnight gap risk with no
    resting stop, and the disabled ``scalping`` entries are priced above
    what this account can size into. Falling back to *those* would silently
    reintroduce risks this session already paid to remove.
    """
    out: dict[str, dict[str, Any]] = {}
    for code, asset_cfg in (config.get("universe") or {}).items():
        asset_cfg = asset_cfg or {}
        if asset_cfg.get("enabled", True):
            continue
        if asset_cfg.get("strategy") != "pullback_bounce":
            continue
        cfg = dict(asset_cfg)
        cfg["enabled"] = True
        cfg["_fallback"] = True
        # The normal 09:00-11:00 window is a morning-only design choice for
        # this strategy; a fallback stepping in mid-day needs the rest of the
        # session to actually be useful, so it gets the wider window the
        # screener's own picks use (still flat by the same force_exit_at).
        cfg["entry_window"] = ["09:00", "14:30"]
        out[str(code)] = cfg
    return out


def _cover_open_positions(
    config: Mapping[str, Any], portfolio: Portfolio
) -> Mapping[str, Any]:
    """Guarantee a strategy config exists for every code with an open position.

    ``due_codes()`` only ever looks at ``rt.strategies``, so a code that
    disappears from the universe -- because it was a disabled static
    fallback, or a screener pick from a process that no longer exists --
    stops being evaluated entirely: no more stop-loss checks, no more
    exit signals, while the position itself stays open for real money.
    This runs on every ``build_runtime()`` call (i.e. every restart) since a
    fresh screener run rebuilds the universe from scratch with no memory of
    what the previous process had open.
    """
    universe = dict(config.get("universe") or {})
    changed = False
    for code, position in portfolio.positions().items():
        entry = universe.get(code)
        if entry and entry.get("enabled", True):
            continue
        if entry:
            cfg = dict(entry)
            cfg["enabled"] = True
            cfg["_open_position_keepalive"] = True
            universe[code] = cfg
            logger.error(
                "%s: open position but disabled in the universe -- "
                "re-enabling so exits keep firing", code,
            )
        else:
            from screener import _PULLBACK_CFG_TEMPLATE, _SCALPING_CFG_TEMPLATE, _ticker_name
            # Match the template to what the position was actually opened
            # with -- pullback_bounce and ORB take different param names
            # (swing_lookback/pullback_bars vs range_minutes/volume_mult),
            # so defaulting to the ORB template regardless of position.strategy
            # would hand a recovered pullback_bounce position ORB's params,
            # silently falling back to PullbackBounce's class defaults for
            # every field ORB doesn't share -- including its tuned stop/lock.
            template = (
                _PULLBACK_CFG_TEMPLATE
                if position.strategy == _PULLBACK_CFG_TEMPLATE["strategy"]
                else _SCALPING_CFG_TEMPLATE
            )
            cfg = {
                **template,
                "name": _ticker_name(code),
                "strategy": position.strategy or template["strategy"],
                "params": dict(template["params"]),
                "_open_position_keepalive": True,
            }
            universe[code] = cfg
            logger.error(
                "%s: open position with no universe entry at all (likely a "
                "dynamic pick from a previous process) -- rebuilding a "
                "fallback config so exits keep firing", code,
            )
        changed = True
    if not changed:
        return config
    return {**config, "universe": universe}


def build_runtime(
    args: argparse.Namespace,
    cli_live: bool,
    force_dry_run: bool = False,
    run_screener: bool = False,
) -> Runtime:
    env = load_env()
    config = load_config(getattr(args, "config", None))
    decision = resolve_mode(env, cli_live=cli_live)
    paths = config.get("paths") or {}

    setup_logging(decision, log_file=paths.get("log_file", "logs/bot.log"))
    print_mode_notice(decision)
    if decision.demoted:
        logger.warning("demoted to paper: %s", "; ".join(decision.reasons))

    # Inject daily screener stocks into the universe before building strategies.
    if run_screener:
        try:
            from screener import DailyScreener
            daily_screener = DailyScreener(config)
            found = daily_screener.scan()
            if found:
                universe = dict(config.get("universe") or {})
                for ticker, asset_cfg in found:
                    universe[ticker] = asset_cfg
                config = {**config, "universe": universe}
                print(f"  Screener added {len(found)} stock(s): {', '.join(t for t, _ in found)}")
            else:
                # Empty and "unreachable" are treated the same: whether KRX/pykrx
                # is actually down, or (observed live 2026-08-27) it's just too
                # early for today's trading-value snapshot to be populated yet,
                # the bot should not start the session with nothing to trade.
                fallback = _fallback_static_stocks(config)
                if fallback:
                    # fallback must win the merge -- it holds the re-enabled,
                    # widened-window version of each code; the base universe
                    # dict still has them disabled under the same keys.
                    universe = {**(config.get("universe") or {}), **fallback}
                    config = {**config, "universe": universe}
                    reason = (
                        "KRX/pykrx unreachable" if daily_screener.last_scan_failed
                        else "0 candidates (today's trading-value data likely not "
                        "posted yet)"
                    )
                    logger.error(
                        "screener: %s - falling back to %d static stock(s) so "
                        "the bot has something to trade: %s",
                        reason, len(fallback), ", ".join(fallback),
                    )
                    print(
                        f"  WARNING: screener found nothing ({reason}). "
                        f"Falling back to {len(fallback)} static stock(s): "
                        f"{', '.join(fallback)}"
                    )
        except Exception as exc:
            logger.warning("screener failed, using static universe: %s", exc)

    # Only the key set matching the resolved mode is read (safety rule 2).
    credentials = load_credentials(env, decision)
    broker = build_broker(decision, credentials, config, force_dry_run=force_dry_run)
    calendar = KrxCalendar.from_config(config)
    # The mode label is what tells the portfolio whether its stored equity is
    # comparable to this run's. A simulated broker reports the configured
    # starting_equity no matter which endpoint was resolved, so "LIVE with a
    # dry-run broker" must not share a label with "LIVE for real" -- otherwise
    # the simulated peak carries over and reads as a crash.
    mode_label = f"{decision.label}{getattr(broker, 'mode_suffix', '-SIM' if broker.dry_run else '')}"
    portfolio = Portfolio(
        state_path=paths.get("state_file", "state.json"),
        trades_path=paths.get("trades_csv", "trades.csv"),
        daily_path=paths.get("daily_pnl_csv", "daily_pnl.csv"),
        mode_label=mode_label,
    )
    config = _cover_open_positions(config, portfolio)
    # KiwoomBroker.allowed_codes is a snapshot taken when build_broker() ran,
    # a few lines above -- before the screener/fallback additions above it and
    # before _cover_open_positions() just now could have added codes that
    # were not in the original static universe. Without this, submit_order()
    # refuses every order for those codes, entries and *exits* alike: a
    # recovered orphaned position could pass every strategy/risk check and
    # still never actually be sellable.
    allowed_codes = getattr(broker, "allowed_codes", None)
    if allowed_codes is not None:
        allowed_codes.update(str(c) for c in (config.get("universe") or {}))
    return Runtime(
        config=config,
        decision=decision,
        credentials=credentials,
        broker=broker,
        portfolio=portfolio,
        risk=RiskManager(config, portfolio, calendar=calendar),
        market_data=MarketData(
            calendar=calendar,
            cache_dir=paths.get("cache_dir", "data/cache"),
            intraday_provider=broker,
        ),
        strategies=build_strategies(config),
        calendar=calendar,
        rules=KrxRules(config),
    )


def _late_exit_reason(
    position: Position,
    price: float,
    now: datetime,
    rules: SessionRules,
    risk_cfg: Mapping[str, Any],
) -> str | None:
    """Reason to take profit right now, or None to keep holding.

    In the last ``risk.late_exit_minutes`` before a same-day force exit, a
    position that has pulled back ``risk.late_exit_stall_pct`` off its peak
    with no fresh high -- but still shows at least
    ``risk.late_exit_min_profit_pct`` -- is worth taking now rather than
    risking the forced exit: it may not clear the normal lock_pct tier before
    then, and giving up the slot costs another signal a chance to use it.
    Close-auction (overnight) holds are exempt -- holding past the close is
    the plan for those, not a stall.
    """
    if position is None or rules.hold_overnight or rules.force_exit_at is None:
        return None
    force_exit_dt = datetime.combine(now.date(), rules.force_exit_at, tzinfo=KST)
    minutes_left = (force_exit_dt - now).total_seconds() / 60.0
    late_minutes = float(risk_cfg.get("late_exit_minutes", 20))
    if not (0 <= minutes_left <= late_minutes):
        return None
    entry = position.entry_price
    peak = position.highest_price or price
    gain = (price - entry) / entry if entry else 0.0
    pullback_from_peak = (peak - price) / peak if peak else 0.0
    stall_pct = float(risk_cfg.get("late_exit_stall_pct", 0.015))
    min_profit_pct = float(risk_cfg.get("late_exit_min_profit_pct", 0.005))
    if pullback_from_peak >= stall_pct and gain >= min_profit_pct:
        return (
            f"장마감 {minutes_left:.0f}분 전, 고점 대비 -{pullback_from_peak:.2%} "
            f"정체 -- 조기 익절 ({gain:+.2%})"
        )
    return None


def _near_limit_decision(
    position: Position,
    price: float,
    now: datetime,
    risk_cfg: Mapping[str, Any],
) -> tuple[bool, str | None]:
    """(should_flag_now, exit_reason) for a 상한가 (upper-limit) candidate.

    User-requested (2026-08-27): a position up risk.near_limit_hold_pct
    (default 26%) or more -- close to KRX's +-30% daily price limit -- skips
    the same-day force exit and profit-lock trail entirely and is sold the
    next morning at or after risk.near_limit_exit_time instead, on the
    theory that a real 상한가 run is worth riding past the close rather than
    selling into it. ``should_flag_now`` is True only the one cycle a
    position first crosses the threshold, so the caller persists it once.
    Once flagged, ``exit_reason`` stays None (keep holding) until the
    morning after entry reaches the exit time, at which point it names the
    reason to sell. The hard ATR stop is a separate, earlier check in
    _process_code and is deliberately not touched here -- it keeps
    protecting the position through the whole hold.
    """
    entry = position.entry_price
    gain = (price - entry) / entry if entry else 0.0
    near_limit = position.near_limit_hold
    should_flag = False
    if not near_limit:
        near_limit_pct = float(risk_cfg.get("near_limit_hold_pct", 0.26))
        if gain >= near_limit_pct:
            should_flag = True
            near_limit = True
    if not near_limit:
        return False, None
    entry_day = position.entry_date()
    exit_time = parse_clock(
        risk_cfg.get("near_limit_exit_time", "09:05"), "risk.near_limit_exit_time"
    )
    reason = None
    if entry_day is not None and now.date() > entry_day and now.time() >= exit_time:
        reason = f"상한가 후보 -- 익일 오전 매도 ({gain:+.2%})"
    return should_flag, reason


def _dip_recovery_reason(
    position: Position,
    price: float,
    risk_cfg: Mapping[str, Any],
) -> str | None:
    """Reason to take a smaller profit now, or None to keep holding.

    User-requested (2026-08-27): -1.8%정도까지 떨어진후에 올라가면 1.5%정도만
    와도 매도 했음해, 2%상승까지 못가고 다시 하한가를 치니까 -- a position
    that dipped risk.dip_recovery_dip_pct (default 1.8%) or more below entry
    at some point (tracked via Position.lowest_price) before bouncing is
    judged less likely to hold all the way to the normal lock_pct (2%)
    target than one that never dipped, so once it recovers to just
    risk.dip_recovery_profit_pct (default 1.5%) it is taken there instead of
    waiting. A position that never dipped that hard is untouched -- this
    only changes behavior for the specific "sharp dip, partial bounce"
    pattern, not every winner.
    """
    entry = position.entry_price
    if not entry:
        return None
    lowest = position.lowest_price
    if lowest is None:
        return None
    dip_pct = (entry - lowest) / entry
    dip_threshold = float(risk_cfg.get("dip_recovery_dip_pct", 0.018))
    if dip_pct < dip_threshold:
        return None
    gain = (price - entry) / entry
    recovery_pct = float(risk_cfg.get("dip_recovery_profit_pct", 0.015))
    if gain >= recovery_pct:
        return f"-{dip_pct:.2%} 눌림 후 반등 -- {recovery_pct:.1%} 조기 익절 ({gain:+.2%})"
    return None


def _dead_market_reason(bars: pd.DataFrame, cfg: Mapping[str, Any]) -> str | None:
    """Reason to suppress a fresh entry this cycle, or None to let it through.

    User-requested (2026-08-27): 거래량과 변동성이 죽은 횡보장에서는 진입
    신호가 나와도 봇이 매매하지 않도록 제한 -- a strategy's own entry
    conditions (breakout, pullback bounce) can still line up technically in a
    stretch where the stock simply is not moving, and trading that costs fees
    for a coin-flip. Checked only against ENTER signals (see _process_code) --
    it never touches an exit. Blocks on either of two independent conditions,
    since either alone can make a signal not worth paying costs for:

    * volatility: the current ATR is below cfg.min_atr_pct of price;
    * volume: the average of the last cfg.recent_bars bars is below
      cfg.min_volume_ratio of the cfg.volume_lookback-bar baseline -- a
      sudden die-off even if the stock was active earlier in the session.

    Fails open (returns None) whenever there isn't enough history to judge
    yet -- blocking every entry because a strategy just went through its own
    warm-up would be worse than not having this filter at all.
    """
    if not cfg.get("enabled", True):
        return None
    atr_period = int(cfg.get("atr_period", 14))
    volume_lookback = int(cfg.get("volume_lookback", 20))
    if len(bars) < max(atr_period, volume_lookback) + 1:
        return None
    price = float(bars["close"].iloc[-1])
    if price <= 0:
        return None

    atr_value = last_valid(atr_indicator(bars, atr_period))
    if atr_value is not None:
        atr_pct = atr_value / price
        min_atr_pct = float(cfg.get("min_atr_pct", 0.004))
        if atr_pct < min_atr_pct:
            return f"변동성 죽음 (ATR {atr_pct:.2%} < 기준 {min_atr_pct:.2%})"

    baseline = last_valid(rolling_mean_volume(bars, volume_lookback))
    if baseline is not None and baseline > 0:
        recent_bars = int(cfg.get("recent_bars", 5))
        recent = float(bars["volume"].iloc[-recent_bars:].mean())
        ratio = recent / baseline
        min_ratio = float(cfg.get("min_volume_ratio", 0.4))
        if ratio < min_ratio:
            return f"거래량 죽음 (최근 {recent_bars}봉 평균이 기준 대비 {ratio:.0%})"

    return None


def _marketable_limit_price(
    orderbook: Any,
    side: str,
    market: str,
    rules: KrxRules,
    buffer_ticks: int,
) -> float | None:
    """A limit price aggressive enough to fill immediately, bounding worst-case slippage.

    User-requested (2026-08-27): 수수료 및 슬리피지 방지 주문 최적화. A plain
    market order has no price protection at all -- in a thin book it can fill
    meaningfully worse than the price the bot decided on. Quoting
    buffer_ticks past the best opposing quote (best_ask for a buy, best_bid
    for a sell) crosses the spread the same way a market order would, so the
    fill should be just as immediate, but the order itself caps how far a
    fill can drift from what was actually on the book when it was sent.

    Returns None -- "send a plain market order instead" -- when there is no
    orderbook to price against. That is a deliberate fail-open: a missing
    quote is not a reason to refuse a trade that otherwise passed every
    other check.
    """
    if orderbook is None:
        return None
    base = orderbook.best_ask if side == LONG else orderbook.best_bid
    if base <= 0:
        return None
    tick = rules.tick_size(base, market)
    offset = tick * max(buffer_ticks, 0)
    if side == LONG:
        # Round UP so the quote is guaranteed to be at or past the ask.
        return float(rules.round_to_tick(base + offset, market, "up"))
    # Round DOWN so the quote is guaranteed to be at or past the bid.
    return float(rules.round_to_tick(base - offset, market, "down"))


#: Not exhaustive -- a new issuer brand or an unusually named company could
#: still slip through either direction -- but catches every case seen live
#: so far (2026-08-28): leveraged/inverse index ETNs and brand-name ETFs
#: pulled in by the volume-surge/VI realtime scan alongside real stocks.
_FUND_LIKE_KEYWORDS = (
    "ETN", "ETF", "레버리지", "인버스", "선물",
    "KODEX", "TIGER", "KBSTAR", "ARIRANG", "SOL", "HANARO", "KOSEF",
    "KINDEX", "TIMEFOLIO", "ACE", "PLUS", "WON", "히어로즈", "파워",
    "마스터", "네비게이터",
)


def _is_fund_like(name: str) -> bool:
    """True if *name* reads like an ETF/ETN/leveraged-index product rather
    than an individual company.

    User-reported (2026-08-28): a live scan added "한투 레버리지코스닥150
    선물 ETN" and "KODEX 미국나스닥AI테크액티브" to the universe alongside
    real stocks. Kiwoom's ranking TRs (ka10023/ka10054) don't expose an
    instrument-type field to filter on directly, so this falls back to
    name matching -- the literal "ETN"/"ETF" markers, index-derivative
    words no ordinary company name would contain, and the major Korean ETF
    issuer brand prefixes.
    """
    upper = name.upper()
    return any(keyword in upper for keyword in _FUND_LIKE_KEYWORDS)


def _ask_bid_ratio(orderbook: Any) -> float:
    """매도잔량 / 매수잔량. 2026-08-27(사용자 요청)부터 이 값이 높을수록
    (매도잔량이 매수잔량을 크게 웃돌수록) 상승 신호로 판단한다 -- 두꺼운
    매수벽은 허매수 받치기 후 물량 털기일 수 있다는 우려에서, 매수세가
    매도벽을 실제로 먹어치우고 있다는 신호(매도잔량 우위)를 더 신뢰하기로
    했다. total_bid_qty가 0이면 나눗셈 예외 대신 매도잔량 유무로 판단."""
    if orderbook.total_bid_qty > 0:
        return orderbook.total_ask_qty / orderbook.total_bid_qty
    return float("inf") if orderbook.total_ask_qty > 0 else 1.0


def _daily_profit_lock_reason(
    equity: float,
    day_start_equity: float,
    lock_pct: float,
) -> str | None:
    """Reason to refuse a fresh entry today, or None to allow one.

    User-requested (2026-08-27): 하루 최소 2%이상은 남겨야 되. Once today's
    equity gain over day_start_equity (set at the first mark_equity() call
    of the KST trading day -- see portfolio.py) reaches lock_pct, no new
    position is opened for the rest of the day: today's gain is not put
    back at risk to chase more. Existing positions are unaffected -- every
    other exit path (hard stop, force exit, late-session/dip-recovery/
    near-limit take-profits) keeps managing them exactly as before, so a
    locked day can still improve, it just cannot add new risk to do it.
    """
    if lock_pct <= 0 or day_start_equity <= 0:
        return None
    gain_pct = (equity - day_start_equity) / day_start_equity
    if gain_pct >= lock_pct:
        return f"오늘 수익 +{gain_pct:.2%} 확보 -- 신규 진입 잠금 (기준 +{lock_pct:.0%})"
    return None


def _reentry_cooldown_reason(
    now: float,
    last_exit_time: float | None,
    last_exit_was_profit: bool,
    cooldown_seconds: float,
) -> str | None:
    """Reason to refuse a fresh entry into a stock just exited, or None to allow one.

    User-reported (2026-08-28): the bot locks in a small gain via the profit
    trail, then watches the stock make a fresh high a few minutes later --
    unable to re-enter because of this same cooldown. A profit-lock exit
    means the strategy's own entry filters (volume, trend, bar strength)
    already vetted this level once today; if they fire again, that is a
    genuine continuation, not "chasing the same resistance", so a profit
    exit skips the cooldown entirely. A stop-out or a forced/defensive exit
    still waits out the full cooldown -- re-entering right after being
    proven wrong is exactly the case this was built to prevent.
    """
    if last_exit_time is None or last_exit_was_profit:
        return None
    elapsed = now - last_exit_time
    if elapsed >= cooldown_seconds:
        return None
    remaining = (cooldown_seconds - elapsed) / 60
    return f"재진입 쿨다운 {remaining:.1f}분 남음 (같은 자리 추격매수 방지)"


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
        self._last_ran: dict[str, float] = {}
        # Re-entry cooldown: code -> monotonic time of last exit. Prevents
        # immediately chasing back into the same level a position was just
        # closed near (e.g. stopped out at a local high, then re-entering
        # the same resistance minutes later on ordinary bar noise).
        self._last_exit_time: dict[str, float] = {}
        # code -> whether that exit was a profit-lock exit ("익절" in the
        # strategy's own reason text), separate from a stop-out or a forced
        # end-of-day/defensive flatten. 2026-08-28 (user-reported): the bot
        # was locking in a small gain via the tight trail and then watching
        # the stock make a fresh high minutes later, unable to re-enter
        # because of this same cooldown. A profit exit means the strategy's
        # own entry filters (volume, trend, bar strength) already vetted
        # this level once today -- if they fire again, that is a genuine
        # continuation signal, not "chasing the same resistance", so the
        # cooldown is skipped only in that case. A stop-out still waits out
        # the full cooldown -- re-entering right after being proven wrong is
        # exactly the case this was built to prevent.
        self._last_exit_was_profit: dict[str, bool] = {}
        self._entry_cooldown_seconds: float = float(
            (rt.config.get("risk") or {}).get("entry_cooldown_minutes", 15)
        ) * 60
        # NXT pre-market bias: code -> % change vs previous close (set at 08:00-08:30)
        self._nxt_bias: dict[str, float] = {}
        self._nxt_scan_date: "date | None" = None
        # Parsed once at construction so a malformed window fails at start-up
        # rather than at 15:19 on the one day it matters.
        self._session_rules: dict[str, SessionRules] = {
            code: SessionRules.from_config(rt.universe.get(code, {}))
            for code in rt.strategies
        }
        # Telegram alerts + remote commands
        self._tg_notifier, self._tg_handler = build_telegram(
            rt.credentials, rt.config
        )
        self._entries_paused: bool = False  # set by /stop, cleared by /resume
        self._wire_telegram_commands()
        # Investor flow scanner (외국인·기관 순매수)
        self._flow_scanner = InvestorFlowScanner(rt.config)
        self._flow_scan_date: "date | None" = None
        # 장마감 자동 복기 리포트 + 금요일 주간 리포트 (2026-08-27, 사용자 요청).
        self._daily_report_date: "date | None" = None
        # 일일 수익 락 (2026-08-27, 사용자 요청): 오늘 수익이 이 비율 이상
        # 확보되면 신규 진입을 잠근다 -- 이미 번 걸 새 리스크로 다시 거는 것
        # 방지. 기존 포지션 관리(손절/청산)는 그대로 계속 작동한다. 0이면
        # 비활성.
        self._daily_profit_lock_pct: float = float(
            (rt.config.get("risk") or {}).get("daily_profit_lock_pct", 0.02)
        )
        self._daily_lock_notified_date: "date | None" = None
        # S&P 500 선물 방향 모니터
        self._us_monitor = USMarketMonitor(rt.config)
        # DART 공시 모니터
        cred_dict = {
            "dart_api_key": rt.credentials.dart_api_key.reveal()
            if hasattr(rt.credentials, "dart_api_key") else "",
        }
        self._dart_monitor = DartMonitor(rt.config, cred_dict)
        self._dart_monitor.set_notifier(self._tg_notifier)
        self._dart_relax_pct: float = float((rt.config.get("dart") or {}).get("entry_relax_pct", 0.2))
        # 트럼프 관련 뉴스 모니터: 텔레그램 알림 + 테마 매칭 시 진입 조건 완화.
        # 사용자 요청으로 실제 진입에도 반영 (_apply_entry_boost 참고) -- 진입을
        # 강제로 걸지는 않고, 기존 기술적 신호의 문턱만 부스트 지속시간 동안 낮춘다.
        self._news_monitor = NewsMonitor(rt.config)
        self._news_monitor.set_notifier(self._tg_notifier)
        self._news_monitor.set_theme_map(rt.config.get("themes") or {})
        self._news_relax_pct: float = float((rt.config.get("news") or {}).get("entry_relax_pct", 0.2))
        # 다중 시간프레임 확인: 일봉 EMA 추세가 하락이면 3분봉 신호와 무관하게
        # 진입 차단. 세션 시작 시 한 번 스캔.
        self._daily_trend = DailyTrendScanner(rt.config)
        # 실시간 호가 불균형 (매수/매도 잔량비) -- 기본은 참고용 로그만.
        # require_confirm을 켜면 잔량비가 기준 미만일 때 진입을 실제로 막는다.
        ob_cfg = rt.config.get("orderbook") or {}
        self._orderbook_enabled: bool = bool(ob_cfg.get("enabled", True))
        self._orderbook_require_confirm: bool = bool(ob_cfg.get("require_confirm", False))
        # min_imbalance(매수잔량 우위=상승) -> min_ask_bid_ratio(매도잔량
        # 우위=상승)로 판단 기준 전환 (2026-08-27, 사용자 요청). 매수벽이
        # 두꺼울수록 상승 탄력이 붙는다는 표준 주문흐름불균형 이론 대신,
        # 매수벽이 "허매수 받치기 후 물량 털기"일 수 있다는 국내 소형주
        # 특유의 우려를 반영 -- 매도잔량이 매수잔량보다 이 배수 이상 많아야
        # (매수세가 매도벽을 실제로 먹어치우고 있다는 뜻) 통과시킨다.
        self._orderbook_min_ask_bid_ratio: float = float(ob_cfg.get("min_ask_bid_ratio", 2.0))
        # 체결강도 (매수/매도 체결량 비율, 100이 중립) -- 기본은 참고용 로그만.
        # require_confirm을 켜면 기준 미만일 때 진입을 실제로 막는다.
        st_cfg = rt.config.get("execution_strength") or {}
        self._strength_enabled: bool = bool(st_cfg.get("enabled", True))
        self._strength_require_confirm: bool = bool(st_cfg.get("require_confirm", False))
        self._strength_min: float = float(st_cfg.get("min_strength", 100.0))
        # 수수료/슬리피지 방지 주문 최적화 (2026-08-27, 사용자 요청): 시장가 대신
        # 최우선호가 + buffer_ticks의 지정가(marketable limit)로 주문해 최악의
        # 체결가에 상한을 둔다.
        #
        # 손절/시간강제청산도 시장가 대신 지정가로 (2026-08-28, 사용자 요청:
        # "익절 손절을 시장가로 하지 말고 다른방법은 없을까 슬리피지도 더
        # 타이트하게"): 이 둘은 반드시 체결돼야 하므로 그냥 시장가를 썼었는데,
        # 시장가는 최악 체결가에 상한이 아예 없다. 대신 stop_limit_buffer_ticks
        # (기본 5틱, 익절용 2틱보다 훨씬 공격적)로 최우선호가를 크게 넘겨
        # 부르면 거의 항상 즉시 체결되면서도 슬리피지 상한은 걸린다. 호가창
        # 자체가 없을 때만(드묾) _marketable_limit_price가 None을 돌려주고
        # 기존처럼 시장가로 자동 폴백된다 -- "반드시 체결" 원칙은 그대로 유지.
        oe_cfg = rt.config.get("order_execution") or {}
        self._use_limit_orders: bool = bool(oe_cfg.get("use_limit_orders", True))
        self._limit_buffer_ticks: int = int(oe_cfg.get("limit_buffer_ticks", 2))
        self._stop_limit_buffer_ticks: int = int(oe_cfg.get("stop_limit_buffer_ticks", 5))
        # Dynamic universe refresh
        scr_cfg = (rt.config.get("screener") or {})
        self._universe_refresh_interval: float = float(
            scr_cfg.get("refresh_interval_minutes", 30)
        ) * 60
        self._last_universe_refresh: float = 0.0
        # 거시 게이지 (미국증시/WTI/미국10년물금리/비트코인/달러지수, 사용자
        # 정의: 다섯 다 "상승 = 강세") -- 강세일 때 진입 조건 완화, 동시 보유
        # 종목 수 확대, 종목당 리스크 확대에 쓴다. 세션 시작 시 한 번만 조회.
        self._macro = MacroGauge(rt.config)
        self._base_risk_pct: float = rt.risk.risk_pct
        self._base_max_positions: int = rt.risk.max_positions

    def session_rules(self, code: str) -> SessionRules:
        return self._session_rules.get(code, SessionRules())

    def _apply_macro_aggressiveness(self) -> None:
        """Loosen/restore entries, position cap, and per-trade risk from the gauge.

        Always recomputed from the *template* defaults (screener.py's
        _SCALPING_CFG_TEMPLATE / _PULLBACK_CFG_TEMPLATE), never from whatever
        the previous call left in place -- otherwise a bullish->neutral->
        bullish flip across two days would compound the relaxation instead of
        resetting it. Static universe entries currently mirror those template
        values exactly, so this applies consistently to both the screener's
        dynamic picks and the fixed fallback stocks; a future stock-specific
        override would get overwritten while the gauge is active, which is a
        known trade-off for keeping this simple.
        """
        bullish = self._macro.is_bullish
        factor = (1.0 - self._macro.entry_relax_pct) if bullish else 1.0

        from screener import _PULLBACK_CFG_TEMPLATE, _SCALPING_CFG_TEMPLATE
        orb_base = _SCALPING_CFG_TEMPLATE["params"]
        pb_base = _PULLBACK_CFG_TEMPLATE["params"]

        for strategy in self.rt.strategies.values():
            if isinstance(strategy, ORB):
                strategy.volume_mult = orb_base["volume_mult"] * factor
                strategy.min_bar_strength = orb_base["min_bar_strength"] * factor
            elif isinstance(strategy, PullbackBounce):
                strategy.min_bar_strength = pb_base["min_bar_strength"] * factor
                strategy.pullback_min_pct = pb_base["pullback_min_pct"] * factor

        self.rt.risk.risk_pct = self._base_risk_pct * (
            self._macro.risk_pct_boost if bullish else 1.0
        )
        self.rt.risk.max_positions = self._base_max_positions + (
            self._macro.extra_positions if bullish else 0
        )

    def _apply_entry_boost(self, strategy: Strategy) -> None:
        """Further loosen one boosted stock's entry thresholds for this evaluate() call.

        Called at most once per code per tick, right before strategy.evaluate(),
        only when a news or DART boost is currently active for that code. Applies
        on top of whatever _apply_macro_aggressiveness() already set this tick
        (the current macro baseline), so it composes without special-casing --
        and it never needs to be undone: next tick, _apply_macro_aggressiveness()
        resets every strategy to the macro baseline before due_codes() runs, so a
        boost that has since expired simply stops being reapplied instead of
        lingering.
        """
        relax_pct = max(self._news_relax_pct, self._dart_relax_pct)
        factor = 1.0 - relax_pct
        if isinstance(strategy, ORB):
            strategy.volume_mult *= factor
            strategy.min_bar_strength *= factor
        elif isinstance(strategy, PullbackBounce):
            strategy.min_bar_strength *= factor
            strategy.pullback_min_pct *= factor

    # -- Telegram wiring ---------------------------------------------------

    def _wire_telegram_commands(self) -> None:
        h = self._tg_handler
        rt = self.rt

        def _status() -> str:
            try:
                account = rt.broker.get_account()
                equity = account.equity
                cash = account.cash
            except Exception:
                equity = rt.portfolio.state.last_equity
                cash = 0.0
            positions = rt.portfolio.positions()
            state = rt.portfolio.state
            lines = [
                f"*봇 상태* (`{rt.decision.label}`)",
                f"자산: `{equity:,.0f} KRW`  현금: `{cash:,.0f} KRW`",
                f"고점대비: `{rt.portfolio.drawdown_pct():.2%}`  상태: `{state.status}`",
                f"진입: `{'일시정지' if self._entries_paused else '활성'}`",
                "",
                f"*포지션 {len(positions)}개*",
            ]
            for code, pos in positions.items():
                name = rt.name_of(code)
                lines.append(
                    f"  `{code} {name}` {pos.side} {int(pos.qty)}주 "
                    f"@ {pos.entry_price:,.0f} 손절 {pos.effective_stop():,.0f}"
                )
            if not positions:
                lines.append("  없음")
            return "\n".join(lines)

        def _stop() -> str:
            self._entries_paused = True
            logger.warning("Telegram /stop: new entries paused")
            self._tg_notifier.alert_paused("텔레그램 /stop 명령")
            return "⏸ 신규 진입 중지됨. 기존 포지션 관리는 계속."

        def _resume() -> str:
            self._entries_paused = False
            logger.info("Telegram /resume: entries re-enabled")
            self._tg_notifier.alert_resumed()
            return "▶️ 진입 재개."

        def _close_all() -> str:
            try:
                closed = rt.broker.flatten(rt.portfolio.positions())
                for code in list(rt.portfolio.positions()):
                    pos = rt.portfolio.get(code)
                    if pos:
                        try:
                            price = rt.broker.get_current_price(code) or pos.entry_price
                        except Exception:
                            price = pos.entry_price
                        rt.portfolio.close_position(code, exit_price=price, exit_reason="telegram /close_all")
                self._tg_notifier.alert_close_all_done(closed)
                return f"🔒 전체 청산 완료 — {closed}개 포지션."
            except Exception as exc:
                logger.error("Telegram /close_all error: %s", exc)
                return f"❌ 청산 실패: {exc}"

        h.on_status = _status
        h.on_stop = _stop
        h.on_resume = _resume
        h.on_close_all = _close_all

    def interval_for(self, code: str) -> int:
        timeframe = self.rt.universe.get(code, {}).get("timeframe", "1Day")
        if timeframe in self.intervals:
            return int(self.intervals[timeframe])
        return int(timeframe_delta(timeframe).total_seconds())

    # -- NXT pre-market bias -----------------------------------------------

    def _run_flow_scan(self) -> None:
        """Fetch foreign + institutional net-buying for all scalping stocks."""
        today = datetime.now(KST).date()
        if self._flow_scan_date == today:
            return
        self._flow_scan_date = today
        scalpers = [
            c for c, cfg in self.rt.universe.items()
            if cfg.get("strategy") == "scalping"
        ]
        if not scalpers or not self._flow_scanner.enabled:
            return
        logger.info("investor_flow: scanning %d stocks", len(scalpers))
        self._flow_scanner.scan(scalpers)

    def _maybe_send_daily_report(self) -> None:
        """Send the evening 복기 once per business day, plus a weekly rollup on Fridays.

        User-requested (2026-08-27): 끝나면 복기하고 항상 뭐가 잘못됐는지
        보여주고 매주 금요일에 일주일치를 보여줘. Fires the first time the
        loop notices the market is closed for a day it hasn't reported yet,
        at or after the regular close -- so it does not fire during the
        pre-open "closed" window, and fires exactly once even though the
        loop keeps polling this branch every tick until the next open.
        """
        today = datetime.now(KST).date()
        if self._daily_report_date == today:
            return
        if not self.rt.calendar.is_business_day(today):
            return
        if datetime.now(KST).time() < REGULAR_CLOSE:
            return
        self._daily_report_date = today

        from report import build_evening_report, build_weekly_report

        try:
            text = build_evening_report(
                self.rt.config, self.rt.portfolio, self.rt.risk, self.rt.decision.label, day=today,
            )
            self._tg_notifier.send(text)
        except Exception:
            logger.exception("daily 복기 report failed")

        if today.weekday() == 4:  # Friday
            try:
                text = build_weekly_report(
                    self.rt.config, self.rt.portfolio, self.rt.risk, self.rt.decision.label,
                    week_end=today,
                )
                self._tg_notifier.send(text)
            except Exception:
                logger.exception("weekly report failed")

    def _sync_broker_allowed_codes(self, new_cfgs: dict[str, Any]) -> None:
        """Let the broker's order-safety whitelist see codes added after startup.

        KiwoomBroker.allowed_codes is a snapshot taken once when the broker was
        built. Any code added to the universe afterward -- a screener pick
        mid-session, a fallback activation -- would otherwise have every order
        refused by that check, entries and exits alike, no matter what the
        strategy or risk manager decide. Not every broker has this attribute
        (paper/dry-run brokers don't restrict at all), hence the getattr guard.
        """
        allowed_codes = getattr(self.rt.broker, "allowed_codes", None)
        if allowed_codes is not None:
            allowed_codes.update(new_cfgs)

    def _realtime_candidates(self) -> list[tuple[str, dict[str, Any]]]:
        """Kiwoom's own real-time volume-surge/VI rankings, as a pykrx-independent
        stand-in when the screener's daily-snapshot-based scan comes back empty.

        Unlike the screener, this never depends on pykrx's "today" trading-value
        snapshot -- confirmed lagging on 2026-08-27 -- so it can still find
        something worth trading in exactly the window where the screener can't.
        Field names are from a community OpenAPI spec, not yet cross-checked
        against a live response (see diagnose-orderbook's [4]/[5] sections), so
        this stays purely supplementary: any parsing miss just yields fewer or
        zero candidates, never a bad one, and the existing static fallback is
        still there as the last resort.
        """
        rt = self.rt
        scr_cfg = rt.config.get("screener") or {}
        min_price = int(scr_cfg.get("min_price", 2000))
        max_price = int(scr_cfg.get("max_price", 0))
        n_stocks = int(scr_cfg.get("n_stocks", 5))
        existing = {str(k) for k in (rt.config.get("universe") or {})}

        # Queried per-market (KOSPI "001" / KOSDAQ "101") rather than the
        # combined "000" so each candidate can be tagged with its real
        # market -- tick size and the sell-side transaction tax both differ
        # by market (see market/rules.py), and rt.market_of() silently
        # defaults anything untagged to KOSPI.
        surges: list[Any] = []
        vis: list[Any] = []
        # ka10023/ka10054's own mrkt_tp convention (001 KOSPI / 101 KOSDAQ) --
        # distinct from KiwoomBroker.MARKET_TYPES, which is a different TR's
        # (stock_info/market_codes) 0/10 scheme.
        for mkt_code, mkt_name in (("001", KOSPI), ("101", KOSDAQ)):
            try:
                for s in rt.broker.get_volume_surge(market=mkt_code):
                    surges.append((s, mkt_name))
            except Exception as exc:
                logger.debug("realtime_candidates: volume_surge(%s) failed: %s", mkt_code, exc)
            try:
                for v in rt.broker.get_vi_triggered(market=mkt_code):
                    vis.append((v, mkt_name))
            except Exception as exc:
                logger.debug("realtime_candidates: vi_triggered(%s) failed: %s", mkt_code, exc)

        picks: list[tuple[str, str, str, float]] = []  # (code, name, market, sort_key)
        seen: set[str] = set()
        for s, mkt_name in sorted(surges, key=lambda p: p[0].surge_rate, reverse=True):
            if s.code in seen or s.code in existing:
                continue
            if _is_fund_like(s.name):
                continue
            if s.price < min_price or (max_price > 0 and s.price > max_price):
                continue
            seen.add(s.code)
            picks.append((s.code, s.name, mkt_name, s.surge_rate))
        for v, mkt_name in sorted(vis, key=lambda p: p[0].trigger_count_today, reverse=True):
            if v.code in seen or v.code in existing:
                continue
            if _is_fund_like(v.name):
                continue
            if v.trigger_price < min_price or (max_price > 0 and v.trigger_price > max_price):
                continue
            seen.add(v.code)
            picks.append((v.code, v.name, mkt_name, float(v.trigger_count_today)))

        from screener import _PULLBACK_CFG_TEMPLATE, _SCALPING_CFG_TEMPLATE
        results: list[tuple[str, dict[str, Any]]] = []
        for i, (code, name, mkt_name, _key) in enumerate(picks[:n_stocks]):
            template = _SCALPING_CFG_TEMPLATE if i % 2 == 0 else _PULLBACK_CFG_TEMPLATE
            cfg = {
                **template,
                "name": name,
                "market": mkt_name,
                "params": dict(template["params"]),
                "_screener": True,
                "_realtime": True,
            }
            results.append((code, cfg))
        return results

    def _refresh_dynamic_universe(self) -> None:
        """Re-run the screener and hot-swap dynamic stocks in the universe.

        Rules
        -----
        * Only stocks added by a previous screener run (_screener: True) are
          eligible for removal. The static universe is never touched.
        * A dynamic stock with an open position is never removed: the bot must
          be able to manage it until it closes.
        * New stocks from the latest scan are added even mid-session.
        * The total dynamic stock count is capped by screener.n_stocks.
        """
        rt = self.rt
        scr_cfg = rt.config.get("screener") or {}
        if not scr_cfg.get("enabled"):
            return

        logger.info("universe refresh: re-running screener")
        try:
            from screener import DailyScreener
            from strategies import build_strategies as _build

            # Pass the current config so existing universe is excluded.
            daily_screener = DailyScreener(rt.config)
            found = daily_screener.scan()
        except Exception as exc:
            logger.warning("universe refresh: screener failed: %s", exc)
            return

        current_universe = dict(rt.config.get("universe") or {})
        open_positions = set(rt.portfolio.positions())

        # Identify which tickers are currently screener-added and removable.
        dynamic_current: set[str] = {
            k for k, v in current_universe.items()
            if (v or {}).get("_screener") and k not in open_positions
        }

        if not found:
            realtime = self._realtime_candidates()
            if realtime:
                logger.info(
                    "universe refresh: pykrx scan empty -- using %d realtime "
                    "volume-surge/VI candidate(s) instead: %s",
                    len(realtime), ", ".join(t for t, _ in realtime),
                )
                found = realtime

        if not found:
            # A zero-candidate scan is not itself proof that nothing today is
            # worth trading. It happens two ways that look identical from
            # here: KRX/pykrx is actually unreachable (last_scan_failed), or
            # -- observed live on 2026-08-27 -- pykrx's "today" trading-value
            # snapshot is only finalized after the close, so any scan before
            # then legitimately sees KRW 0 traded for every ticker and every
            # one fails min_trading_value_m. Either way the right response is
            # the same: don't destroy a working universe over it, and if
            # there is nothing to protect, trade the static fallback stocks
            # instead of sitting idle until data eventually shows up.
            if dynamic_current:
                logger.warning(
                    "universe refresh: screener found 0 candidates -- keeping "
                    "the current %d dynamic stock(s) rather than clearing the "
                    "universe (screener failure or today's trading-value data "
                    "not available yet)", len(dynamic_current),
                )
                return
            already_fallback = any(
                (v or {}).get("_fallback") for v in current_universe.values()
            )
            if already_fallback:
                return
            fallback = _fallback_static_stocks(rt.config)
            if fallback:
                # fallback must win the merge -- see the matching comment in
                # build_runtime()'s startup path for why.
                universe = {**current_universe, **fallback}
                rt.config = {**rt.config, "universe": universe}
                self._sync_broker_allowed_codes(fallback)
                for code, asset_cfg in fallback.items():
                    rt.strategies[code] = build_strategy(
                        code, asset_cfg, rt.config.get("risk") or {}
                    )
                    self._session_rules[code] = SessionRules.from_config(asset_cfg)
                reason = (
                    "KRX/pykrx 접속 불가" if daily_screener.last_scan_failed
                    else "0개 통과 (오늘 거래대금 데이터 아직 미반영 가능)"
                )
                logger.error(
                    "universe refresh: screener 0건 (%s) -- 대체 종목 %d개 투입: %s",
                    reason, len(fallback), ", ".join(fallback),
                )
                self._tg_notifier.send(
                    f"스크리너 0건 ({reason}) -- 대체 종목 투입: "
                    + ", ".join(f"{c}({rt.name_of(c)})" for c in fallback)
                )
            return

        new_tickers: set[str] = {t for t, _ in found}

        # Stocks to drop: were dynamic, not in the new scan, no open position.
        to_remove = dynamic_current - new_tickers
        # Stocks to add: in the new scan, not already in the universe.
        to_add = [(t, cfg) for t, cfg in found if t not in current_universe]

        if not to_remove and not to_add:
            logger.info("universe refresh: no changes")
            return

        updated_universe = dict(current_universe)
        for code in to_remove:
            del updated_universe[code]
            if code in rt.strategies:
                del rt.strategies[code]
            if code in self._session_rules:
                del self._session_rules[code]
            logger.info("universe refresh: removed %s (dropped from ranking)", code)

        for ticker, asset_cfg in to_add:
            updated_universe[ticker] = asset_cfg
            logger.info("universe refresh: added %s", ticker)

        # Rebuild strategies for new tickers only.
        if to_add:
            new_cfg = {**rt.config, "universe": {t: c for t, c in to_add}}
            new_strats = _build(new_cfg)
            rt.strategies.update(new_strats)
            for code in new_strats:
                self._session_rules[code] = SessionRules.from_config(
                    updated_universe.get(code, {})
                )

        rt.config = {**rt.config, "universe": updated_universe}
        if to_add:
            self._sync_broker_allowed_codes({t: c for t, c in to_add})

        removed_names = ", ".join(to_remove) if to_remove else "없음"
        added_names = ", ".join(t for t, _ in to_add) if to_add else "없음"
        logger.info(
            "universe refresh done — removed [%s]  added [%s]  total dynamic: %d",
            removed_names, added_names,
            sum(1 for v in updated_universe.values() if (v or {}).get("_screener")),
        )
        self._tg_notifier.send(
            f"🔄 *유니버스 업데이트*\n"
            f"제거: `{removed_names}`\n"
            f"추가: `{added_names}`"
        )

    def _run_nxt_scan(self) -> None:
        """Query NXT pre-market prices and store directional bias per stock."""
        today = datetime.now(KST).date()
        if self._nxt_scan_date == today:
            return
        self._nxt_scan_date = today
        self._nxt_bias.clear()

        prev_day = self.rt.calendar.previous_business_day(today)
        rt = self.rt
        scalpers = [c for c, cfg in rt.universe.items() if cfg.get("strategy") == "scalping"]
        if not scalpers:
            return

        logger.info("NXT pre-market scan starting for %d stocks", len(scalpers))
        for code in scalpers:
            try:
                current = rt.broker.get_current_price(code)
                if not current or current <= 0:
                    continue
                bars = rt.market_data.get_bars(
                    code, "1Day",
                    start=(prev_day - timedelta(days=5)).isoformat(),
                    end=prev_day.isoformat(),
                )
                if bars is None or bars.empty:
                    continue
                prev_close = float(bars["close"].iloc[-1])
                if prev_close <= 0:
                    continue
                chg = (current - prev_close) / prev_close
                self._nxt_bias[code] = chg
                label = "bullish" if chg > 0.003 else ("bearish" if chg < -0.003 else "neutral")
                logger.info(
                    "NXT %s %s: %.2f%% vs prev close %s (%s)",
                    code, rt.name_of(code), chg * 100, f"{prev_close:,.0f}", label,
                )
            except Exception as exc:
                logger.debug("NXT scan skipped %s: %s", code, exc)

    def _nxt_allows_entry(self, code: str) -> bool:
        """Return False only when NXT pre-market data shows a bearish gap."""
        cfg = self.rt.universe.get(code, {})
        if cfg.get("strategy") != "scalping":
            return True
        bias = self._nxt_bias.get(code)
        if bias is None:
            return True  # no NXT data -> do not block
        nxt_cfg = self.rt.config.get("nxt") or {}
        threshold = float(nxt_cfg.get("bearish_threshold_pct", -0.01))
        return bias >= threshold

    #: Cadence while a stock's entry window is open, in seconds.
    ENTRY_WINDOW_INTERVAL = 60

    def due_codes(self, now: float, moment: datetime | None = None) -> list[str]:
        """Stocks to look at this tick.

        Three cadences, not one:

        * an **open position** is always due. Its hard stop is the only thing
          between it and an unbounded loss, and Kiwoom holds no resting stop
          order, so nothing checks it while this loop is not looking;
        * a stock **inside its entry window** is due at least once a minute.
          A close-based entry has a four-minute window, and its bars are
          daily: on the timeframe cadence alone it would be looked at once a
          day, almost never during those four minutes, and would therefore
          never trade at all;
        * everything else is due on its strategy's own cadence.
        """
        moment = moment or datetime.now(KST)
        held = set(self.rt.portfolio.positions())
        due: list[str] = []
        for code in self.rt.strategies:
            if code in held:
                due.append(code)
                continue
            if now - self._last_ran.get(code, float("-inf")) >= self._interval_now(
                code, moment
            ):
                due.append(code)
        return due

    def _interval_now(self, code: str, moment: datetime) -> float:
        base = float(self.interval_for(code))
        rules = self.session_rules(code)
        if rules.entry_from is not None and rules.entry_allowed(moment)[0]:
            return min(base, float(self.ENTRY_WINDOW_INTERVAL))
        return base

    def mark_ran(self, code: str, now: float) -> None:
        self._last_ran[code] = now

    # -- one cycle ---------------------------------------------------------

    def run_cycle(self, codes: Sequence[str] | None = None) -> None:
        rt = self.rt
        try:
            account = rt.broker.get_account()
        except BrokerError as exc:
            logger.error("account fetch failed, skipping this cycle: %s", exc)
            return

        rt.portfolio.mark_equity(account.equity)

        # Rules 5/6 - the kill switch is evaluated before anything else.
        status = rt.risk.check_drawdown(account.equity)
        if status.breached and not rt.portfolio.stopped:
            rt.risk.trip_kill_switch(status, rt.broker)
            self._tg_notifier.alert_kill_switch(
                account.equity, rt.portfolio.state.peak_equity
            )
        if rt.portfolio.stopped:
            logger.warning(
                "bot is STOPPED (%s) - no new orders. Run `python main.py resume` to clear.",
                rt.portfolio.state.stopped_reason,
            )
            return

        # Rule 8: KRX trades one session; auctions are not safe for market orders.
        session_ok, session_reason = rt.calendar.can_place_market_order()
        if not session_ok:
            logger.info("session gate: %s", session_reason)

        try:
            open_orders = rt.broker.open_order_codes()
        except BrokerError as exc:
            logger.error("open-order fetch failed, assuming none: %s", exc)
            open_orders = []

        capacity = rt.risk.capacity(account.equity)
        logger.info("portfolio capacity: %s", capacity.describe())

        for code in codes if codes is not None else list(rt.strategies):
            try:
                self._process_code(code, account, session_ok, session_reason, open_orders)
            except BrokerError as exc:
                logger.error("%s: broker error, skipping: %s", code, exc)
            except Exception as exc:  # never let one stock kill the loop
                logger.exception("%s: unexpected error, skipping: %s", code, exc)

    def _process_code(
        self,
        code: str,
        account,
        session_ok: bool,
        session_reason: str,
        open_orders: Sequence[str],
    ) -> None:
        rt = self.rt
        asset_cfg = rt.universe.get(code, {})
        market = rt.market_of(code)
        strategy = rt.strategies[code]
        label = f"{code} {rt.name_of(code)}".strip()

        timeframe = str(asset_cfg.get("timeframe", "1Day"))
        # get_bars() grants intraday bars a 2-bar-interval grace period before
        # calling them stale (is_current() in data.py) -- for a 3-minute
        # timeframe that's up to 6 minutes of a cached, unmoving close price.
        # Fine for scanning candidates; not for a stock we're already holding,
        # where the hard stop and profit-lock trail must see the live price
        # every cycle. A real position sat at a frozen +2.22% for six minutes
        # on 2026-08-26 while the cache served the same stale bar, giving back
        # most of the gain before the next live fetch finally caught up.
        has_position = rt.portfolio.get(code) is not None
        barset = rt.market_data.get_bars(
            code=code,
            timeframe=timeframe,
            lookback_bars=max(strategy.warmup + 60, 260),
            market=market,
            use_cache=not has_position,
        )
        bars = barset.bars
        if len(bars) < strategy.warmup:
            logger.info(
                "%s: only %d bars, strategy needs %d to warm up - holding",
                label,
                len(bars),
                strategy.warmup,
            )
            return
        if barset.synthetic:
            logger.warning("%s: SYNTHETIC bars in use - signals are illustrative only", label)

        # Every decision below reads the last bar as if it were the present:
        # the entry signal, the trailing stop, the hard-stop comparison. Bars
        # that stopped days ago make all three answer a question about the
        # past. Skipping loses a cycle; acting on them can sell at a level the
        # stock left behind.
        if not rt.market_data.is_current(bars, barset.timeframe):
            held = " A POSITION IS OPEN and is not being managed this cycle." if (
                rt.portfolio.get(code) is not None
            ) else ""
            logger.error(
                "%s: bars end at %s and are not current [%s] - skipping.%s",
                label,
                str(bars.index[-1])[:16],
                barset.source,
                held,
            )
            return

        price = float(bars["close"].iloc[-1])
        position = rt.portfolio.get(code)

        # Kiwoom has no bracket order, so the loop enforces the hard stop itself.
        if position is not None:
            # Track the highest price seen since entry for peak-based exits.
            if price > (position.highest_price or 0.0):
                position.highest_price = price
                rt.portfolio.update_position(position)
            # Track the lowest price too -- _dip_recovery_reason needs it to
            # tell a position that dipped hard and bounced from one that just
            # rose smoothly.
            if price < (position.lowest_price or float("inf")):
                position.lowest_price = price
                rt.portfolio.update_position(position)

            new_trail = strategy.update_trailing_stop(bars, position)
            if new_trail is not None and new_trail != position.trail_stop:
                position.trail_stop = new_trail
                rt.portfolio.update_position(position)
                logger.info("%s: trailing stop moved to %s", label, f"{new_trail:,.0f}")

            stop = position.effective_stop()
            if (position.is_long and price <= stop) or (position.is_short and price >= stop):
                logger.warning(
                    "%s: STOP HIT at %s (stop %s) - closing", label, f"{price:,.0f}", f"{stop:,.0f}"
                )
                self._tg_notifier.alert_stop_hit(
                    code, rt.name_of(code), price, stop
                )
                self._submit_exit(
                    code, position, price, "stop hit", open_orders, asset_cfg,
                    session_ok, session_reason, market=market, urgent=True,
                )
                return

        now = datetime.now(KST)

        # 상한가 후보 홀드 (2026-08-27, 사용자 요청): 진입가 대비 26% 이상
        # 오르면 당일에는 팔지 않고 다음날 아침에 매도한다. See
        # _near_limit_decision's docstring for the full reasoning.
        if position is not None:
            should_flag, exit_reason = _near_limit_decision(
                position, price, now, rt.config.get("risk") or {}
            )
            if should_flag:
                position.near_limit_hold = True
                rt.portfolio.update_position(position)
                entry = position.entry_price
                gain = (price - entry) / entry if entry else 0.0
                logger.warning(
                    "%s: 상한가 후보 감지 (+%.2f%%) - 당일 매도 보류, 익일 오전 매도 예정",
                    label, gain * 100,
                )
                self._tg_notifier.send(
                    f"\U0001f680 {rt.name_of(code)}({code}) +{gain:.1%} -- 상한가 후보로 판단, "
                    f"오늘은 매도 보류하고 내일 오전에 매도합니다."
                )
            if position.near_limit_hold:
                if exit_reason is not None:
                    logger.warning("%s: NEAR-LIMIT MORNING EXIT - %s", label, exit_reason)
                    self._submit_exit(
                        code, position, price, exit_reason, open_orders, asset_cfg,
                        session_ok, session_reason, market=market,
                    )
                # Whether it exited above or is still waiting for the morning
                # window, nothing else this cycle may sell it -- that is the
                # entire point of the hold.
                return

        # 눌림 후 반등 조기 익절 (2026-08-27, 사용자 요청): 진입 후 -1.8%
        # 이상 눌렸다가 반등한 포지션은, 정상 확정선(2%)까지 못 가고 다시
        # 밀리는 경우가 많다고 판단해 1.5%에서 먼저 정리한다. 한 번도 그만큼
        # 눌린 적 없는 포지션은 그대로 정상 로직(strategy.evaluate())을 탄다.
        if position is not None:
            reason = _dip_recovery_reason(position, price, rt.config.get("risk") or {})
            if reason is not None:
                logger.warning("%s: DIP-RECOVERY TAKE-PROFIT - %s", label, reason)
                self._submit_exit(
                    code, position, price, reason, open_orders, asset_cfg,
                    session_ok, session_reason, market=market,
                )
                return

        # Time-based rules outrank the strategy in both directions: a day
        # trade is closed before the auction whatever the chart says, and an
        # overnight position is closed the next morning as planned.
        rules = self.session_rules(code)
        if position is not None:
            due, why = rules.exit_due(now, position.entry_date())
            if due:
                logger.warning("%s: TIME EXIT - %s", label, why)
                self._submit_exit(
                    code, position, price, why, open_orders, asset_cfg,
                    session_ok, session_reason, market=market, urgent=True,
                )
                return

        # Late-session opportunistic take-profit. Normally a position only
        # exits once it clears lock_pct (2%) or hits its stop -- anything
        # stalled in between just rides along to the force exit. That is a
        # bad trade-off this close to it: a real gain can slip back to
        # breakeven by the time the forced exit fires, and the slot stays
        # tied up when another ORB/pullback signal could use it. Close-auction
        # (overnight) holds are exempt -- holding past the close is the plan.
        if position is not None:
            reason = _late_exit_reason(position, price, now, rules, rt.config.get("risk") or {})
            if reason is not None:
                logger.warning("%s: LATE-SESSION TAKE-PROFIT - %s", label, reason)
                self._submit_exit(
                    code, position, price, reason, open_orders, asset_cfg,
                    session_ok, session_reason, market=market,
                )
                return

        if position is None and (
            self._news_monitor.is_boosted(code) or self._dart_monitor.is_boosted(code)
        ):
            self._apply_entry_boost(strategy)

        signal = strategy.evaluate(bars, position)
        logger.info(
            "%s [%s] %s - %s (close=%s)",
            label,
            asset_cfg.get("timeframe", "?"),
            signal.action.value,
            signal.reason,
            f"{price:,.0f}",
        )

        if not signal.actionable:
            return

        if signal.action is Action.EXIT:
            # The strategy's own EXIT covers both its stop_pct stop and its
            # lock_pct/peak_trail_pct take-profit under one Action -- only the
            # stop needs the urgent (wider) buffer, so it's the one told apart
            # by its own reason text (see the f"{stop_pct:.0%} 손절 ..."
            # format both ORB and PullbackBounce use).
            self._submit_exit(
                code, position, price, signal.reason, open_orders, asset_cfg,
                session_ok, session_reason, market=market,
                urgent="손절" in signal.reason,
            )
            return

        allowed, why = rules.entry_allowed(now)
        if not allowed:
            logger.info("%s: entry outside its window - %s", label, why)
            return

        # 거래량/변동성 죽은 횡보장 필터 (2026-08-27, 사용자 요청): 전략
        # 조건이 기술적으로는 맞아도 지금 이 순간 시장 자체가 안 움직이면
        # 수수료만 나가는 매매다. 청산에는 전혀 영향 없음 -- 신규 진입만
        # 막는다.
        dead_reason = _dead_market_reason(bars, rt.config.get("volatility_filter") or {})
        if dead_reason is not None:
            logger.info("%s: entry suppressed - %s", label, dead_reason)
            return

        self._submit_entry(
            code=code,
            signal=signal,
            price=price,
            bars=bars,
            account=account,
            session_ok=session_ok,
            session_reason=session_reason,
            open_orders=open_orders,
            asset_cfg=asset_cfg,
            market=market,
            strategy=strategy,
            label=label,
        )

    # -- order paths -------------------------------------------------------

    def _price_limit_state(self, bars, market: str, price: float) -> tuple[bool, str]:
        """Is the current price pinned at the daily ±30% limit?"""
        if len(bars) < 2:
            return False, ""
        limits = self.rt.rules.price_limits(float(bars["close"].iloc[-2]), market)
        if limits.blocks_buy(price):
            return True, f"limit-up at {price:,.0f}: no sellers, order would not fill"
        if limits.blocks_sell(price):
            return True, f"limit-down at {price:,.0f}: no buyers, order would not fill"
        return False, ""

    def _submit_exit(
        self,
        code: str,
        position: Position | None,
        price: float,
        reason: str,
        open_orders: Sequence[str],
        asset_cfg: Mapping[str, Any],
        session_ok: bool,
        session_reason: str,
        market: str = KOSPI,
        urgent: bool = False,
    ) -> None:
        """Both exit paths use a marketable limit order (see
        _marketable_limit_price), never a plain market order, as long as
        orderbook data is available (it silently falls back to market
        otherwise -- the one case nothing can bound). ``urgent`` (the hard
        stop, a forced time exit -- anything that must not sit unfilled)
        crosses the book by stop_limit_buffer_ticks instead of the normal
        limit_buffer_ticks, wide enough to fill immediately in virtually all
        cases while still capping the worst-case price, unlike a market
        order which has no cap at all."""
        if position is None:
            return
        rt = self.rt
        ctx = TradeContext(
            code=code,
            side=SHORT if position.is_long else LONG,
            qty=position.qty,
            price=price,
            tradable=rt.broker.get_stock_info(code).tradable,
            session_ok=session_ok,
            session_reason=session_reason,
            min_qty=float(asset_cfg.get("min_qty", 1)),
            open_order_codes=open_orders,
            existing_position=position,
            available_cash=0.0,
            is_exit=True,
        )
        checks = rt.risk.pre_trade_checks(ctx)
        if not checks.passed:
            logger.warning("%s: exit blocked - %s", code, checks.describe())
            return
        limit_price = None
        if self._use_limit_orders:
            buffer_ticks = self._stop_limit_buffer_ticks if urgent else self._limit_buffer_ticks
            orderbook = rt.broker.get_orderbook(code)
            limit_price = _marketable_limit_price(
                orderbook, ctx.side, market, rt.rules, buffer_ticks
            )
        try:
            result = rt.broker.submit_order(
                code=code, side=ctx.side, qty=position.qty, price=limit_price,
                is_exit=True, note=f"exit: {reason}",
            )
        except BrokerError as exc:
            self._reconcile_rejected_exit(code, reason, exc)
            return
        if result.submitted:
            rt.portfolio.close_position(code, exit_price=price, exit_reason=reason)
            self._last_exit_time[code] = time.monotonic()
            self._last_exit_was_profit[code] = "익절" in reason
            self._tg_notifier.alert_exit(
                code=code,
                name=rt.name_of(code),
                qty=int(position.qty),
                price=price,
                entry_price=float(position.entry_price),
                reason=reason,
            )
        else:
            logger.info("%s: exit simulated only - position left untouched in state", code)

    def _reconcile_rejected_exit(self, code: str, reason: str, exc: Exception) -> None:
        """A sell was rejected -- confirm whether the broker actually holds
        this position before letting the loop retry it forever.

        2026-08-28 (live incident): 122640 예스티 stayed in state.json after
        the account's real sellable quantity was already 0 -- Kiwoom rejected
        every retry with "매도가능수량이 부족합니다. 0주 매도가능" and, with
        nothing to reconcile the local position against, the same rejected
        sell just repeated every cycle forever, permanently occupying a
        position slot and its risk budget. broker.KiwoomBroker.submit_order
        already retries once when Kiwoom reports a smaller-but-positive
        sellable qty (a known state-vs-broker drift); this handles the
        remaining case where the real qty is 0 and no retry can help.

        If the broker's own holdings report zero (or nothing) for this code,
        the account genuinely does not have it any more -- clear the stale
        local position (no fabricated trade record: we do not know the real
        exit price or when it happened, only that it is gone) and say so
        loudly so the user can check the actual account. Otherwise this was
        some other, possibly transient, broker error -- leave the position
        alone and let the next cycle retry as before.
        """
        rt = self.rt
        try:
            holdings = rt.broker.get_holdings()
        except BrokerError:
            logger.error("%s: exit failed and holdings check also failed: %s", code, exc)
            return
        real_qty = holdings[code].qty if code in holdings else 0.0
        if real_qty > 0:
            logger.error(
                "%s: exit failed, but broker still shows %s held - will retry next cycle: %s",
                code, real_qty, exc,
            )
            return
        logger.error(
            "%s: exit rejected and broker confirms 0 held -- local position was stale, "
            "clearing it: %s", code, exc,
        )
        rt.portfolio.state.positions.pop(code, None)
        rt.portfolio.save()
        self._tg_notifier.send(
            f"⚠️ {rt.name_of(code)}({code}) 매도 거부 -- 실제 계좌에는 이미 없는 "
            f"것으로 확인되어 로컬 포지션 기록을 정리했습니다. 원인: {exc}\n"
            "실제 체결/잔고는 HTS·MTS에서 직접 확인해주세요."
        )

    def _submit_entry(
        self,
        code: str,
        signal,
        price: float,
        bars,
        account,
        session_ok: bool,
        session_reason: str,
        open_orders: Sequence[str],
        asset_cfg: Mapping[str, Any],
        market: str,
        strategy: Strategy,
        label: str,
    ) -> None:
        rt = self.rt
        side = signal.action.side

        if not self._nxt_allows_entry(code):
            logger.info(
                "%s: entry skipped - NXT pre-market bearish (%.2f%%)",
                label, self._nxt_bias.get(code, 0) * 100,
            )
            return

        if side == LONG and not self._daily_trend.allows_long(code):
            logger.info(
                "%s: entry skipped - 일봉 EMA 추세 하락 (다중 시간프레임 필터)",
                label,
            )
            return

        # Fetched once and reused below for the marketable-limit entry price too,
        # so enabling both features never doubles the API call.
        orderbook = None
        if side == LONG and (self._orderbook_enabled or self._use_limit_orders):
            orderbook = rt.broker.get_orderbook(code)
            if orderbook is not None and self._orderbook_enabled:
                ask_bid_ratio = _ask_bid_ratio(orderbook)
                logger.info(
                    "%s: 호가 매도/매수 잔량비 %.2f (매수 %.0f / 매도 %.0f, "
                    "1차 매수 %.0f/%.0f 매도 %.0f/%.0f)",
                    label, ask_bid_ratio,
                    orderbook.total_bid_qty, orderbook.total_ask_qty,
                    orderbook.best_bid, orderbook.best_bid_qty,
                    orderbook.best_ask, orderbook.best_ask_qty,
                )
                if (
                    self._orderbook_require_confirm
                    and ask_bid_ratio < self._orderbook_min_ask_bid_ratio
                ):
                    logger.warning(
                        "%s: entry blocked - 매도잔량 우위 부족 (잔량비 %.2f < 기준 %.2f)",
                        label, ask_bid_ratio, self._orderbook_min_ask_bid_ratio,
                    )
                    return

        if side == LONG and self._strength_enabled:
            strength = rt.broker.get_execution_strength(code)
            if strength is not None:
                logger.info(
                    "%s: 체결강도 %.2f (5분 %.2f, 20분 %.2f, 60분 %.2f, %s 기준)",
                    label, strength.strength, strength.strength_5min,
                    strength.strength_20min, strength.strength_60min, strength.as_of,
                )
                if (
                    self._strength_require_confirm
                    and strength.strength < self._strength_min
                ):
                    logger.warning(
                        "%s: entry blocked - 체결강도 매도 우위 (%.2f < 기준 %.2f)",
                        label, strength.strength, self._strength_min,
                    )
                    return

        if self._entries_paused:
            logger.info("%s: entry skipped - paused via Telegram /stop", label)
            return

        lock_reason = _daily_profit_lock_reason(
            account.equity, rt.portfolio.state.day_start_equity, self._daily_profit_lock_pct,
        )
        if lock_reason is not None:
            logger.info("%s: entry skipped - %s", label, lock_reason)
            today = datetime.now(KST).date()
            if self._daily_lock_notified_date != today:
                self._daily_lock_notified_date = today
                self._tg_notifier.send(f"\U0001f512 {lock_reason} -- 오늘 남은 시간 신규 진입 없음")
            return

        cooldown_reason = _reentry_cooldown_reason(
            time.monotonic(),
            self._last_exit_time.get(code),
            self._last_exit_was_profit.get(code, False),
            self._entry_cooldown_seconds,
        )
        if cooldown_reason is not None:
            logger.info("%s: entry skipped - %s", label, cooldown_reason)
            return

        # S&P 500 선물 방향 필터
        if not self._us_monitor.check():
            logger.info("%s: entry skipped - %s", label, self._us_monitor.status_line)
            return

        # DART 공시 차단 (CB·유상증자 등)
        if self._dart_monitor.is_bearish_blocked(code):
            logger.warning("%s: entry blocked - DART 부정 공시 감지 (오늘 차단)", label)
            return
        if self._dart_monitor.is_boosted(code):
            logger.info("%s: DART 긍정 공시 부스트 활성 — 진입 우선", label)
        if self._news_monitor.is_boosted(code):
            logger.info("%s: 트럼프 뉴스 테마 부스트 활성 — 진입 우선", label)

        # Investor flow filter (외국인·기관 순매수)
        flow_sig = self._flow_scanner.signal_for(code)
        flow_ok, flow_reason = flow_sig.allows_entry(
            self._flow_scanner.require_smart_money
        )
        if not flow_ok:
            logger.warning("%s: entry blocked by investor flow — %s", label, flow_reason)
            return
        if flow_sig.bias.value != "neutral":
            logger.info("%s: investor flow — %s", label, flow_reason)

        allowed, reason = rt.risk.can_open_new(code, side, equity=account.equity)
        if not allowed:
            logger.warning("%s: entry blocked - %s", label, reason)
            return

        capacity = rt.risk.capacity(account.equity)
        sizing = rt.risk.size(
            code=code,
            equity=account.equity,
            atr=signal.atr,
            price=price,
            asset_cfg=asset_cfg,
            available_cash=account.cash,
            risk_budget=capacity.remaining_risk,
        )
        if not sizing.ok:
            # Rule 4: a non-positive stop distance means no order at all.
            logger.warning("%s: order skipped - %s", label, sizing.reason)
            return

        at_limit, limit_reason = self._price_limit_state(bars, market, price)
        ctx = TradeContext(
            code=code,
            side=side,
            qty=sizing.qty,
            price=price,
            tradable=rt.broker.get_stock_info(code).tradable,
            session_ok=session_ok,
            session_reason=session_reason,
            min_qty=float(asset_cfg.get("min_qty", 1)),
            open_order_codes=open_orders,
            existing_position=rt.portfolio.get(code),
            available_cash=account.cash,
            at_price_limit=at_limit,
            price_limit_reason=limit_reason,
        )
        checks = rt.risk.pre_trade_checks(ctx)
        if not checks.passed:
            logger.warning("%s: entry blocked - %s", label, checks.describe())
            return

        stop = rt.rules.stop_price(price, sizing.stop_distance, market)
        limit_price = None
        if self._use_limit_orders:
            limit_price = _marketable_limit_price(
                orderbook, side, market, rt.rules, self._limit_buffer_ticks
            )
        logger.info(
            "%s: %s qty=%d risk=%s (%.2f%% of equity) stop=%s tick=%d order=%s",
            label,
            side,
            int(sizing.qty),
            f"{sizing.realised_risk():,.0f}",
            100 * sizing.realised_risk() / account.equity if account.equity else 0.0,
            f"{stop:,.0f}",
            rt.rules.tick_size(price, market),
            f"limit {limit_price:,.0f}" if limit_price is not None else "market",
        )
        result = rt.broker.submit_order(
            code=code, side=side, qty=sizing.qty, price=limit_price, stop_price=stop,
            note=signal.reason,
        )
        if result.submitted:
            rt.portfolio.open_position(
                Position(
                    symbol=code,
                    side=side,
                    qty=sizing.qty,
                    entry_price=price,
                    stop_price=stop,
                    stop_distance=sizing.stop_distance,
                    strategy=strategy.name,
                    trail_stop=signal.trail_stop,
                    take_profit=signal.take_profit,
                )
            )
            self._tg_notifier.alert_entry(
                code=code,
                name=rt.name_of(code),
                side=side,
                qty=int(sizing.qty),
                price=price,
                stop=stop,
                equity=account.equity,
            )
        else:
            logger.info("%s: order simulated only (dry run) - nothing sent", label)

    # -- loops -------------------------------------------------------------

    def run_once(self) -> None:
        self.run_cycle()
        self.rt.portfolio.record_day(self.rt.portfolio.state.last_equity)

    def run_forever(self) -> None:
        logger.info(
            "entering continuous loop (tick=%ds); per-stock cadence: %s",
            self.tick_seconds,
            {c: self.interval_for(c) for c in self.rt.strategies},
        )
        self._tg_handler.start()
        self._dart_monitor.set_universe({str(c) for c in self.rt.strategies})
        self._dart_monitor.start()
        self._news_monitor.start()
        fallback_codes = [
            c for c in self.rt.strategies if self.rt.universe.get(c, {}).get("_fallback")
        ]
        if fallback_codes:
            self._tg_notifier.send(
                "스크리너 실패 (KRX/pykrx 접속 불가) -- 대체 종목으로 시작함: "
                + ", ".join(f"{c}({self.rt.name_of(c)})" for c in fallback_codes)
            )
        try:
            while True:
                if self.rt.calendar.is_nxt_premarket():
                    # 08:00-08:30: NXT pre-market -- scan prices, then nap.
                    self._run_nxt_scan()
                    self._run_flow_scan()
                    time.sleep(self.closed_sleep)
                    continue
                if not self.rt.calendar.in_session():
                    self._maybe_send_daily_report()
                    # Sleep until the earlier of next KRX open or next NXT start.
                    secs_to_open = self.rt.calendar.seconds_until_open()
                    secs_to_nxt = (
                        self.rt.calendar.next_nxt_start() - datetime.now(KST)
                    ).total_seconds()
                    wait = min(secs_to_open, max(secs_to_nxt, 0), self.closed_sleep)
                    logger.info("KRX closed; sleeping %.0fs", max(wait, 1.0))
                    time.sleep(max(wait, 1.0))
                    continue

                now = time.time()

                # 일봉 추세 스캔 -- 날짜당 한 번만 실제로 조회한다 (내부에서 캐싱).
                self._daily_trend.scan(
                    self.rt.market_data,
                    {c: self.rt.universe[c] for c in self.rt.strategies},
                )

                # 거시 게이지 -- 날짜당 한 번만 조회, 결과는 매 틱 재적용
                # (전략 인스턴스가 유니버스 리프레시로 새로 생겨도 항상 반영되게).
                self._macro.scan()
                self._apply_macro_aggressiveness()

                # Periodically hot-swap the dynamic universe mid-session.
                if (
                    self._universe_refresh_interval > 0
                    and now - self._last_universe_refresh >= self._universe_refresh_interval
                ):
                    self._refresh_dynamic_universe()
                    self._last_universe_refresh = now

                due = self.due_codes(now)
                if due:
                    self.run_cycle(due)
                    for code in due:
                        self.mark_ran(code, now)
                if self.rt.portfolio.stopped:
                    logger.error(
                        "STOPPED state reached - exiting loop. Run `python main.py resume`."
                    )
                    return
                time.sleep(self.tick_seconds)
        except KeyboardInterrupt:
            logger.info("interrupted by user - shutting down cleanly")
            self.rt.portfolio.record_day(self.rt.portfolio.state.last_equity)
        finally:
            self._dart_monitor.stop()
            self._news_monitor.stop()
            self._tg_handler.shutdown()


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_profile(args: argparse.Namespace) -> int:
    from universe_profile import SWEEP_TIMEFRAMES, atr_sweep, format_profiles, run_profile

    # pykrx serves daily bars, but minute bars only come from the broker -- so
    # without one, every 60-minute stock profiles on synthetic data and the
    # numbers mean nothing. Profiling reads charts and places no orders, which
    # is why --live is offered here at all.
    cli_live = bool(getattr(args, "live", False))
    rt = build_runtime(args, cli_live=cli_live, force_dry_run=not cli_live)

    if getattr(args, "atr_sweep", False):
        day_traders = {
            c: cfg for c, cfg in rt.universe.items() if cfg.get("strategy") == "scalping"
        }
        if not day_traders:
            print("\n  No stocks are configured for day trading; nothing to sweep.")
            return 0
        if rt.broker.dry_run:
            print(
                "\n  A sweep needs real minute bars. Run "
                "`python main.py profile --atr-sweep --live`."
            )
            return 1
        sample = next(iter(day_traders))
        strategy = rt.strategies[sample]
        print(f"\nMeasuring {len(day_traders)} stocks at {len(SWEEP_TIMEFRAMES)} "
              "timeframes; this takes a minute ...", flush=True)
        print()
        print(
            atr_sweep(
                day_traders,
                rt.market_data,
                round_trip_cost_pct=strategy.round_trip_cost_pct,
                max_cost_share=strategy.max_cost_share,
                atr_period=rt.risk.atr_period,
            )
        )
        return 0

    intraday = [c for c, cfg in rt.universe.items() if cfg.get("timeframe") != "1Day"]
    if intraday and rt.broker.dry_run:
        print(
            f"\n  No broker connected, so the {len(intraday)} intraday stocks "
            f"({', '.join(intraday)})\n  will profile on SYNTHETIC bars. "
            "Use `python main.py profile --live` for real ones."
        )

    if args.refresh_calendar:
        end = date.today()
        start = date(end.year - 1, 1, 1)
        try:
            holidays = rt.calendar.refresh_from_pykrx(start, end)
        except Exception as exc:
            print(f"\nCalendar refresh failed (needs pykrx + network): {exc}")
            return 1
        print(f"\nNon-trading weekdays {start} .. {end}:")
        print("krx:\n  holidays:")
        for day in sorted(holidays):
            print(f"    - '{day.isoformat()}'")
        print("\nPaste the block above into config.yaml.")
        return 0

    print(f"\nProfiling {len(rt.strategies)} stocks over {args.months} month(s) ...\n")
    profiles, correlations = run_profile(rt.config, rt.market_data, rt.strategies, args.months)
    print(format_profiles(profiles, correlations, rt.config.get("themes")))
    return 0


def cmd_study_bounce(args: argparse.Namespace) -> int:
    """Measure the drop-then-bounce pattern on the configured universe.

    Places no orders and reads only daily bars. It exists because "I have seen
    this happen a lot" and "this happens more often than not" are different
    claims, and only one of them can be checked.
    """
    from study import format_bounce_study, parse_codes, run_bounce_study

    cli_live = bool(getattr(args, "live", False))
    rt = build_runtime(args, cli_live=cli_live, force_dry_run=not cli_live)

    universe: Mapping[str, Mapping[str, Any]] = rt.universe
    survivorship = False
    codes = parse_codes(getattr(args, "codes", "") or "")
    if getattr(args, "market", None):
        from study import market_codes

        window_start = months_to_start(args.months, datetime.now(KST)).date()
        try:
            codes, list_source = market_codes(
                args.market,
                as_of=window_start,
                limit=args.limit,
                broker=None if rt.broker.dry_run else rt.broker,
            )
        except Exception as exc:
            print(f"\n  could not read the {args.market} ticker list: {exc}")
            return 1
        if not codes:
            print(
                f"\n  No {args.market} ticker list could be read. KRX's listing\n"
                "  endpoint has been returning non-JSON, and the OHLCV snapshot\n"
                "  fallback did not answer either. See logs/bot.log for what was\n"
                "  tried. You can still pass codes by hand with --codes.",
            )
            return 1
        survivorship = True
        print(
            f"\n{args.market.upper()}: {len(codes)} tickers from {list_source}. "
            "This takes a while -- one request per stock.",
            flush=True,
        )
        if "CURRENT" in list_source:
            print(
                "  Note: this is what is listed TODAY. Stocks that crashed and were\n"
                "  delisted are missing entirely, which flatters a bounce study.",
                flush=True,
            )
    if codes:
        # Testing the rule on stocks it was not chosen on is the only check
        # that separates a real pattern from a parameter search that happened
        # to fit nine names. Names come from the broker; nothing here is
        # configured, and nothing here is ever ordered.
        # Resolving a name costs a request each, which is worth it for a
        # pasted list and absurd for a whole board.
        resolve = len(codes) <= 40
        universe = {
            code: {
                "name": (rt.market_data.get_name(code) or "") if resolve else "",
                "market": KOSPI,
            }
            for code in codes
        }
        if not survivorship:
            print(f"\nOut-of-sample: {len(codes)} codes not in config.yaml", flush=True)

    costs = rt.config.get("costs") or {}
    tax = max(float(v) for v in (costs.get("sell_tax_bps") or {"x": 0.0}).values())
    cost_pct = (
        2 * float(costs.get("commission_bps", 0.0))
        + 2 * float(costs.get("slippage_bps", 0.0))
        + tax
    ) / 10_000.0

    from study import PATTERNS

    what = (
        f"{args.down_days} consecutive down sessions"
        if args.pattern == "drop"
        else PATTERNS[args.pattern][1].lower()
    )
    print(
        f"\nScanning {args.months} month(s) of daily bars for {what} ...",
        flush=True,
    )
    def progress(done: int, total: int) -> None:
        if total > 40 and done % 25 == 0:
            print(f"  {done}/{total} stocks ...", flush=True)

    stats = run_bounce_study(
        universe,
        rt.market_data,
        progress=progress,
        months=args.months,
        down_days=args.down_days,
        drop_pct=abs(args.drop) / 100.0,
        limit_down=bool(args.limit_down),
        pattern=args.pattern,
    )
    print()
    if getattr(args, "rank", None):
        from study import format_selection_study

        print(
            format_selection_study(
                stats,
                cost_pct=cost_pct,
                top_n=args.rank,
                min_events=args.min_events,
            )
        )
        return 0
    print(
        format_bounce_study(
            stats,
            down_days=args.down_days,
            drop_pct=abs(args.drop) / 100.0,
            cost_pct=cost_pct,
            limit_down=bool(args.limit_down),
            survivorship_note=survivorship,
            per_stock=len(universe) <= 40,
            pattern=args.pattern,
        )
    )
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    from backtest import Backtester, format_result

    rt = build_runtime(args, cli_live=False, force_dry_run=True)
    end = datetime.now(KST)
    start = months_to_start(args.months, end)
    # flush=True throughout: redirected output is block-buffered, and a long
    # run that prints nothing for ten minutes looks like a hang.
    print(
        f"\nLoading {args.months} month(s) of bars ({start.date()} -> {end.date()}) ...",
        flush=True,
    )

    barsets = {}
    for code, asset_cfg in rt.universe.items():
        if not asset_cfg.get("enabled", True):
            continue
        timeframe = asset_cfg.get("timeframe", "1Day")
        strategy = rt.strategies.get(code)
        # Fetch the indicator run-up *before* the requested window, so the whole
        # window can actually be traded rather than spent warming up.
        fetch_from = warmup_start(start, timeframe, strategy.warmup if strategy else 0)
        barset = rt.market_data.get_bars(
            code=code,
            timeframe=timeframe,
            start=fetch_from,
            end=end,
            market=rt.market_of(code),
        )
        barsets[code] = barset
        label = f"{code} {rt.name_of(code)}".strip()
        warm = strategy.warmup if strategy else 0
        tradable = max(0, len(barset) - warm)
        print(
            f"  {label:<22} {len(barset):>6d} bars  [{barset.source}]"
            f"  warmup {warm} -> {tradable} tradable",
            flush=True,
        )

    total_bars = sum(len(b) for b in barsets.values())
    print(f"\nSimulating {total_bars:,} bars ...", flush=True)

    def progress(done: int, total: int) -> None:
        print(f"  {done * 100 // total:>3d}%  ({done:,} / {total:,} bars)", flush=True)

    engine = Backtester(rt.config)
    result = engine.run(barsets, rt.strategies, rt.universe, progress=progress)
    print()
    print(format_result(result, names={c: rt.name_of(c) for c in rt.universe}))
    return 0


class _SingleInstanceLock:
    """OS-level lock so a second trading loop can't run against the same
    state.json at the same time.

    2026-08-28 (diagnosed from a live log): the user restarted the bot by
    pulling a fix and running `python main.py live --live` again without
    first closing the previous window. Nothing stopped that -- `resume`/
    `stop` only flip a flag inside state.json, they never touch the actual
    running process -- so the old process kept trading with its stale,
    pre-fix code (e.g. an ETN the new _is_fund_like() would have excluded)
    at the same time the new one ran, both hitting Kiwoom independently
    against the same real account.

    Uses the OS's own file lock (msvcrt on Windows, flock on POSIX) rather
    than a PID file: a PID file can be left behind by a crash and then
    wrongly block a real restart, but an OS lock is released by the kernel
    the instant the process exits, by any means, so it can never go stale.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = None

    def acquire(self) -> bool:
        fh = open(self.path, "a+")
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()))
        fh.flush()
        self._fh = fh
        return True

    def release(self) -> None:
        fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            fh.close()


def cmd_trade(args: argparse.Namespace, cli_live: bool) -> int:
    force_dry_run = bool(getattr(args, "dry_run", False))
    rt = build_runtime(args, cli_live=cli_live, force_dry_run=force_dry_run, run_screener=True)

    lock_path = Path(rt.portfolio.store.path).resolve().parent / ".bot.lock"
    lock = _SingleInstanceLock(lock_path)
    if not lock.acquire():
        print(
            f"\n!! 이미 다른 봇 프로세스가 실행 중입니다 ({lock_path}).\n"
            "   같은 계좌로 두 프로세스가 동시에 매매하면 중복 주문이 나갈 수 있습니다.\n"
            "   먼저 실행 중인 창을 닫거나 Ctrl+C로 종료한 뒤 다시 실행하세요.\n"
        )
        return 1

    try:
        return _cmd_trade_locked(args, rt)
    finally:
        lock.release()


def _cmd_trade_locked(args: argparse.Namespace, rt: "Runtime") -> int:
    if rt.decision.live:
        try:
            account = rt.broker.get_account()
        except BrokerError as exc:
            print(f"cannot read the live account, aborting: {exc}")
            return 1
        print_live_banner(
            account_equity=account.equity,
            max_loss_per_trade=rt.risk.max_loss_per_trade(account.equity),
            max_portfolio_loss=rt.risk.max_portfolio_loss(account.equity),
            endpoint=rt.decision.endpoint,
            countdown_seconds=int(
                (rt.config.get("live_banner") or {}).get("countdown_seconds", 10)
            ),
        )

    if rt.portfolio.stopped:
        print(
            f"\nBot is STOPPED: {rt.portfolio.state.stopped_reason}\n"
            "Run `python main.py resume` to clear it.\n"
        )
        return 1

    if rt.broker.dry_run:
        source = (
            "real account and real charts"
            if getattr(rt.broker, "mode_suffix", "") == "-OBSERVE"
            else "SIMULATED data - no credentials, so bars are synthetic"
        )
        print(
            f"\n[{getattr(rt.broker, 'label', 'DRY-RUN')}] {source}; "
            "no orders will be sent.\n"
        )

    phase = rt.calendar.phase()
    print(f"KRX session phase: {phase.value}")

    engine = TradingEngine(rt)
    watch = bool(getattr(args, "watch", False))
    force_dry_run = bool(getattr(args, "dry_run", False))
    # --dry-run defaults to a single cycle so it stays a quick smoke test.
    # --watch turns it into an all-day observation run: the same loop, the same
    # signals, still sending nothing. That is the honest way to build trust in
    # the signals before any real money follows them.
    if getattr(args, "once", False) or (force_dry_run and not watch):
        engine.run_once()
    else:
        engine.run_forever()
    return 0


def _clean_pasted_value(raw: str, name: str) -> str:
    """Recover the bare value from whatever the user pasted.

    People paste the whole ``NAME=value`` line, or a multi-line block that the
    console flattens into one prompt. Joining on whitespace kills embedded
    newlines and spaces; a Kiwoom key contains neither.
    """
    value = "".join(raw.split())
    lowered = value.lower()
    for prefix in (f"{name}=", f"{name}:"):
        if lowered.startswith(prefix.lower()):
            value = value[len(prefix):]
            break
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value


#: Plausible lengths. Kiwoom app keys and secrets run about 43 characters;
#: anything far outside this means the paste grabbed the wrong thing.
_VALUE_LIMITS: dict[str, tuple[int, int, str]] = {
    "KEY": (20, 120, "an app key or secret is normally about 43 characters"),
    "ACCOUNT": (6, 24, "an account number is normally 8-12 digits"),
}


def _mask(value: str) -> str:
    """A preview that proves the paste landed without printing the secret."""
    if len(value) <= 12:
        return "*" * len(value)
    return f"{value[:4]}{'*' * (len(value) - 8)}{value[-4:]}"


def _prompt_value(label: str, name: str, kind: str) -> str:
    """Ask for one value and refuse obviously wrong input.

    Input is hidden, because a console transcript gets pasted into chat and
    support threads. What makes that safe is the feedback: the length is
    checked against a real key and echoed with a masked preview, so a paste
    that swallowed extra lines is caught here rather than surfacing later as
    an unexplained authentication failure.
    """
    import getpass

    low, high, hint = _VALUE_LIMITS[kind]
    for attempt in range(3):
        try:
            raw = getpass.getpass(f"  {label} (typing is hidden): ")
        except Exception:  # no tty, e.g. piped input
            raw = input(f"  {label}: ")
        value = _clean_pasted_value(raw, name)
        if not value:
            return ""
        if low <= len(value) <= high:
            print(f"    -> captured {len(value)} characters: {_mask(value)}")
            return value
        print(f"    !! that is {len(value)} characters; {hint}.")
        if len(value) > high:
            print("       You probably pasted more than the value itself.")
            print("       Copy ONLY the key, with no label and no extra lines.")
        if attempt < 2:
            print("       Try again, or press Enter to skip this one.\n")
    print("    skipped after 3 attempts")
    return ""


def _write_env(path: Path, entries: Mapping[str, str]) -> None:
    """Rewrite only the named lines, preserving everything else."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    remaining = dict(entries)
    out: list[str] = []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        if key in remaining:
            out.append(f"{key}={remaining.pop(key)}")
        else:
            out.append(line)
    out.extend(f"{k}={v}" for k, v in remaining.items())
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def cmd_setup(args: argparse.Namespace) -> int:
    """Write credentials into .env without opening an editor."""
    path = Path(".env")

    if args.enable_live:
        # The two environment confirmations are the human half of the live
        # gate. Writing them takes a deliberate flag plus the exact phrase, and
        # the third condition -- --live on the run command -- still stands.
        print()
        print("!" * 78)
        print("  ENABLING LIVE TRADING IN .env".center(78))
        print("!" * 78)
        print("  This writes the two environment confirmations. After this, any")
        print("  command carrying --live trades REAL MONEY.")
        print(f"\n  Type exactly:  {LIVE_CONFIRM_TOKEN}")
        typed = input("  > ").strip()
        if typed != LIVE_CONFIRM_TOKEN:
            print("\n  Phrase did not match. .env was not modified.")
            return 1
        _write_env(path, {"KIWOOM_PAPER": "false", "KIWOOM_LIVE_CONFIRM": LIVE_CONFIRM_TOKEN})
        print(f"\n  Live confirmations written to {path.resolve()}")
        print("  Next:  python main.py check --live")
        return 0

    target = "LIVE" if args.live else "PAPER"
    print()
    print("=" * 78)
    print(f"  .env SETUP  -  writing the {target} key pair".center(78))
    print("=" * 78)
    print("  Paste ONE value per prompt with a right-click, then press Enter.")
    print("  Typing is hidden; a masked preview confirms what was captured.")
    print("  Press Enter on an empty prompt to leave that value unchanged.\n")

    entries: dict[str, str] = {}
    for label, name, kind in (
        (f"{target} App Key", f"KIWOOM_{target}_APP_KEY", "KEY"),
        (f"{target} Secret Key", f"KIWOOM_{target}_SECRET_KEY", "KEY"),
        ("Account number", "KIWOOM_ACCOUNT_NO", "ACCOUNT"),
    ):
        value = _prompt_value(label, name, kind)
        if value:
            entries[name] = value

    if not entries:
        print("\n  Nothing captured; .env was not modified.")
        return 1

    _write_env(path, entries)
    print(f"\n  Wrote {len(entries)} value(s) to {path.resolve()}")
    for name, value in entries.items():
        print(f"    {name:<26} {len(value)} chars")

    missing = [n for n in (f"KIWOOM_{target}_APP_KEY", f"KIWOOM_{target}_SECRET_KEY")
               if n not in entries]
    if missing:
        print("\n  Still empty: " + ", ".join(missing))
        print("  Run this again and fill them; check cannot connect without both.")

    if target == "LIVE":
        print("\n  For live trading, also run:  python main.py setup --enable-live")
        print("\n  Next:  python main.py check --live")
    else:
        print("\n  Next:  python main.py check")
    print("=" * 78)
    return 0



def cmd_adopt(args: argparse.Namespace) -> int:
    """Bring existing broker holdings under the bot's management.

    The bot only manages what is in ``state.json``; a stock bought by hand is
    invisible to it, so it would happily buy the same name again. Adoption
    writes those positions into state -- but the quantity was chosen by a
    person, not by the sizing rule, so the risk it implies has to be measured
    and shown rather than assumed to fit the 1% budget.
    """
    rt = build_runtime(args, cli_live=bool(getattr(args, "live", False)), force_dry_run=False)

    if isinstance(rt.broker, DryRunBroker):
        print("\n  No broker credentials loaded; nothing to adopt.")
        return 1

    try:
        account = rt.broker.get_account()
        holdings = rt.broker.get_holdings()
    except BrokerError as exc:
        print(f"\n  Could not read the account: {exc}")
        return 1

    print()
    print("=" * 78)
    print("  ADOPT EXISTING HOLDINGS".center(78))
    print("=" * 78)
    print(f"  Account equity : {account.equity:>14,.0f} KRW")
    print(f"  Risk budget    : {rt.risk.max_loss_per_trade(account.equity):>14,.0f} KRW per position"
          f"  ({rt.risk.risk_pct:.0%})")
    print("=" * 78)

    if not holdings:
        print("\n  The account holds nothing. Nothing to adopt.")
        return 0

    foreign = {c: h for c, h in holdings.items() if c not in rt.universe}
    mine = {c: h for c, h in holdings.items() if c in rt.universe}

    if foreign:
        print("\n  Left alone -- not in this bot's universe:")
        for code, h in foreign.items():
            print(f"    {code}   {h.qty:,.0f} sh @ {h.avg_price:,.0f}")
        print("    The bot will never buy or sell these. To have it manage one,")
        print("    add the code to `universe` in config.yaml first.")

    if not mine:
        print("\n  None of the holdings are in the universe; nothing to adopt.")
        return 0

    proposals: list[tuple[str, Position, float, float]] = []
    print("\n  Proposed stops:")
    for code, h in mine.items():
        asset_cfg = rt.universe[code]
        strategy = rt.strategies.get(code)
        barset = rt.market_data.get_bars(
            code=code,
            timeframe=asset_cfg.get("timeframe", "1Day"),
            lookback_bars=max((strategy.warmup if strategy else 30) + 30, 120),
            market=rt.market_of(code),
        )
        atr_value = last_valid(atr_indicator(barset.bars, rt.risk.atr_period)) or 0.0
        price = h.current_price or float(barset.bars["close"].iloc[-1])
        if atr_value <= 0 or price <= 0:
            print(f"    {code}: no ATR available, cannot place a stop -- skipped")
            continue

        distance = atr_value * rt.risk.hard_stop_atr_mult
        stop = rt.rules.stop_price(price, distance, rt.market_of(code))
        risk = (price - stop) * h.qty
        risk_pct = risk / account.equity if account.equity else 0.0

        label = f"{code} {rt.name_of(code)}".strip()
        flag = "  OVER BUDGET" if risk_pct > rt.risk.risk_pct else ""
        print(f"\n    {label}")
        print(f"      holding    : {h.qty:,.0f} sh, avg {h.avg_price:,.0f}, now {price:,.0f}")
        print(f"      ATR({rt.risk.atr_period})     : {atr_value:,.0f}")
        print(f"      stop at    : {stop:,.0f}   ({distance:,.0f} below current)")
        print(f"      risk       : {risk:,.0f} KRW = {risk_pct:.2%} of equity{flag}")

        proposals.append((code, Position(
            symbol=code,
            side=LONG,
            qty=h.qty,
            entry_price=h.avg_price or price,
            stop_price=stop,
            stop_distance=distance,
            strategy=(strategy.name if strategy else ""),
            entry_time=str(datetime.now(KST)),
        ), risk, risk_pct))

    if not proposals:
        print("\n  Nothing could be adopted.")
        return 1

    total_risk = sum(p[2] for p in proposals)
    print("\n" + "-" * 78)
    print(f"  Total risk if adopted: {total_risk:,.0f} KRW = "
          f"{total_risk / account.equity:.2%} of equity"
          f"   (portfolio cap {rt.risk.max_total_risk_pct:.0%})")
    over = [p for p in proposals if p[3] > rt.risk.risk_pct]
    if over:
        print()
        print("  Some positions risk more than the per-trade budget. That is what")
        print("  holding a size the sizing rule did not choose costs. Your options:")
        print("    - adopt anyway, and the stop caps the loss at the figure above")
        print("    - sell part of the position by hand first, then adopt")
        print("    - leave it unmanaged and trade it yourself")

    if not args.yes:
        print()
        answer = input("  Adopt these positions into state.json? [yes/no] ").strip().lower()
        if answer not in ("yes", "y"):
            print("\n  Nothing was written.")
            return 1

    for code, position, _, _ in proposals:
        rt.portfolio.open_position(position)
    rt.portfolio.mark_equity(account.equity)

    print(f"\n  Adopted {len(proposals)} position(s) into {rt.portfolio.store.path}")
    print("  The bot will now manage their stops and exits, and will not buy them again.")
    print("\n  Next:  python main.py status")
    print("=" * 78)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Read-only connectivity probe. Places no orders, ever.

    Everything the trading loop relies on is exercised here - token, account,
    holdings, working orders, per-stock tradability and the intraday chart -
    but only through GET-shaped calls. This is what makes it safe to point at a
    live account: the endpoint paths and api-id codes in ``broker.py`` were
    written from documentation and have never met a real account, and this is
    the way to find that out without money on the line.
    """
    cli_live = bool(getattr(args, "live", False))
    rt = build_runtime(args, cli_live=cli_live, force_dry_run=False)

    print()
    print("=" * 78)
    print("  CONNECTION CHECK  -  READ ONLY, NO ORDERS ARE SENT".center(78))
    print("=" * 78)
    print(f"  Mode      : {rt.decision.label}")
    print(f"  Endpoint  : {rt.decision.endpoint}")
    print(f"  Credential: {rt.credentials.loaded_for} key set")

    # Length is the fastest way to spot a paste that swallowed extra text --
    # a Kiwoom key is about 43 characters, and authentication fails with the
    # same "App Key verification failed" message whatever the junk was.
    app_len = len(rt.credentials.app_key.reveal())
    secret_len = len(rt.credentials.secret_key.reveal())
    suspicious = " <- NOT a plausible key length" if not (
        20 <= app_len <= 120 and 20 <= secret_len <= 120
    ) else ""
    print(f"  Key length: app {app_len}, secret {secret_len}{suspicious}")
    print(f"  Session   : {rt.calendar.phase().value}")
    print("=" * 78)
    if suspicious:
        print()
        print("  The stored key does not look like a Kiwoom key (~43 characters).")
        print("  Run `python main.py setup` again and paste ONLY the key value.")

    if isinstance(rt.broker, DryRunBroker):
        print(
            "\n  No credentials loaded, so this is the simulated broker - nothing was\n"
            f"  contacted. This run reads the {rt.decision.label} key pair.\n"
        )
        print("  -- What .env actually contains " + "-" * 46)
        env = os.environ
        for name in (
            "KIWOOM_PAPER_APP_KEY",
            "KIWOOM_PAPER_SECRET_KEY",
            "KIWOOM_LIVE_APP_KEY",
            "KIWOOM_LIVE_SECRET_KEY",
            "KIWOOM_ACCOUNT_NO",
        ):
            value = env.get(name) or ""
            # Length only - never the value itself (safety rule 9).
            state = f"set, {len(value)} chars" if value else "EMPTY"
            used = " <- this run uses these" if rt.decision.label in name else ""
            print(f"  {name:<26} {state}{used}")
        for name in ("KIWOOM_PAPER", "KIWOOM_LIVE_CONFIRM"):
            value = env.get(name)
            # These two are switches, not secrets, so the value is safe to show.
            shown = repr(value) if value is not None else "not set"
            print(f"  {name:<26} {shown}")

        print()
        if (env.get("KIWOOM_LIVE_APP_KEY") or "") and rt.decision.paper:
            print("  The LIVE pair is filled but this run resolved to PAPER, so those")
            print("  keys were deliberately not read. Either move them to the PAPER")
            print("  variables, or supply all three live confirmations.")
        else:
            print(f"  Fill KIWOOM_{rt.decision.label}_APP_KEY and _SECRET_KEY in .env,")
            print("  with no spaces around '=', then run this again.")
        print("=" * 78)
        return 1

    results: list[tuple[str, bool, str]] = []
    auth_failure: str | None = None

    def probe(label: str, fn) -> Any:
        nonlocal auth_failure
        if auth_failure is not None:
            # Once the token is refused every later probe fails identically;
            # running them just repeats the same error a dozen times.
            results.append((label, False, "skipped: authentication failed"))
            return None
        try:
            value = fn()
        except BrokerAuthError as exc:
            auth_failure = mask_text(str(exc))
            results.append((label, False, auth_failure[:200]))
            return None
        except Exception as exc:
            results.append((label, False, mask_text(str(exc))[:88]))
            return None
        results.append((label, True, ""))
        return value

    # ka00001 is the one api-id confirmed against Kiwoom's published sample, so
    # a failure here means credentials or endpoint, not a wrong TR code.
    accounts = probe(
        "인증 + 계좌번호조회 (ka00001)",
        rt.broker.get_account_numbers
        if hasattr(rt.broker, "get_account_numbers")
        else (lambda: []),
    )
    account = probe("잔고조회", rt.broker.get_account)
    holdings = probe(
        "보유종목 조회",
        rt.broker.get_holdings if hasattr(rt.broker, "get_holdings") else (lambda: {}),
    )
    open_orders = probe("미체결 주문 조회", rt.broker.open_order_codes)

    for code in rt.universe:
        label = f"종목정보 {code} {rt.name_of(code)}".strip()
        probe(label, lambda c=code: rt.broker.get_stock_info(c))

    # The intraday chart path has never run against a real account, and four of
    # the eight stocks depend on it for every signal they produce.
    end = datetime.now(KST)
    for code, cfg in rt.universe.items():
        timeframe = cfg.get("timeframe", "1Day")
        if timeframe == "1Day":
            continue
        probe(
            f"{timeframe} 차트 {code}",
            lambda c=code, t=timeframe: rt.broker.get_chart(
                c, t, months_to_start(1, end), end
            ),
        )

    print()
    for label, ok, detail in results:
        mark = "  OK  " if ok else " FAIL "
        line = f"  [{mark}] {label}"
        print(line if ok else f"{line}\n           -> {detail}")

    failed = [r for r in results if not r[1]]
    print()
    print("-" * 78)
    if account is not None:
        print(f"  Account equity : {account.equity:>16,.0f} KRW")
        print(f"  Cash available : {account.cash:>16,.0f} KRW")
        print(
            f"  Risk per trade : {rt.risk.max_loss_per_trade(account.equity):>16,.0f} KRW"
            f"   ({rt.risk.risk_pct:.0%})"
        )
        print(
            f"  Risk, whole book: {rt.risk.max_portfolio_loss(account.equity):>15,.0f} KRW"
            f"   ({rt.risk.max_total_risk_pct:.0%})"
        )
    if accounts:
        print(f"  Accounts       : {accounts}")
    if holdings:
        print("  Broker holdings:")
        for code, h in holdings.items():
            known = "" if code in rt.universe else "   <- NOT in this bot's universe"
            label = f"{code} {rt.name_of(code)}".strip()
            print(
                f"    {label:<22} {h.qty:>10,.0f} sh @ {h.avg_price:>10,.0f}"
                f"  now {h.current_price:>10,.0f}{known}"
            )
    if open_orders:
        print(f"  Working orders : {open_orders}")

    # A cash figure of zero can mean the account holds no cash, or that the
    # field carrying it is named something we did not try. Those look
    # identical from the outside and lead to opposite conclusions, so the raw
    # response is printed whenever the numbers do not add up. Amounts are not
    # secrets; the token and keys never appear in this payload.
    raw = dict(getattr(rt.broker, "last_balance_fields", {}) or {})
    if account is not None and holdings:
        stock_value = sum(h.qty * h.current_price for h in holdings.values())
        unexplained = account.equity - stock_value - account.cash
        if account.cash <= 0 or abs(unexplained) > max(1000.0, stock_value * 0.01):
            print()
            print("  Equity does not reconcile against the holdings:")
            print(f"    stocks at market : {stock_value:>16,.0f} KRW")
            print(f"    cash             : {account.cash:>16,.0f} KRW")
            print(f"    reported equity  : {account.equity:>16,.0f} KRW")
            print(f"    unexplained      : {unexplained:>16,.0f} KRW")
            print()
            print("  Raw balance fields (amounts only, no credentials):")
            for name, value in sorted(raw.items()):
                print(f"    {name:<24} {value}")
            print()
            print("  If a field here holds your deposit and the bot read 0, the field")
            print("  name belongs in KiwoomBroker.CASH_FIELDS. Until then the bot will")
            print("  refuse every buy for lack of cash.")

    print("-" * 78)
    if auth_failure:
        print("  AUTHENTICATION FAILED - nothing else could be tested.")
        print(f"  {auth_failure}")
        print()
        if "8030" in auth_failure or "투자구분" in auth_failure:
            # Kiwoom names this one exactly, so there is nothing to guess at.
            other = "LIVE" if rt.decision.paper else "PAPER"
            print(f"  Kiwoom says the key belongs to the {other} environment, but this")
            print(f"  run used the {rt.decision.label} endpoint. The key and the endpoint")
            print("  have to match. To use these keys against the live account:")
            print()
            print("      python main.py setup --live          (put the keys in the LIVE slots)")
            print("      python main.py setup --enable-live   (open the gate, type the phrase)")
            print("      python main.py check --live")
            print()
            print("  Run them one at a time, in that order. Every one of the three is")
            print("  required; skipping any leaves the run on the mock account.")
        else:
            print("  Likely causes, in order:")
            print("    1. the key pair belongs to the other environment")
            print("       (a mock key cannot authenticate against the live host)")
            print("    2. the token request field names in broker.py do not match Kiwoom's")
            print("    3. the key was revoked or re-issued")
    elif failed:
        print(f"  {len(failed)} check(s) FAILED - do not trade until these pass.")
        print("  Most likely cause: an endpoint path or api-id in broker.py does not")
        print("  match Kiwoom's current API. They are overridable in config.yaml")
        print("  under kiwoom.endpoints / kiwoom.api_ids.")
    else:
        print("  All checks passed. No orders were sent.")
    print("=" * 78)
    return 1 if failed else 0


def cmd_status(args: argparse.Namespace) -> int:
    rt = build_runtime(args, cli_live=False, force_dry_run=True)
    state = rt.portfolio.state
    capacity = rt.risk.capacity(state.last_equity)
    print()
    print("=" * 78)
    print("  BOT STATUS".center(78))
    print("=" * 78)
    print(f"  Status            : {state.status}")
    if state.stopped:
        print(f"  Stopped reason    : {state.stopped_reason}")
        print(f"  Stopped at        : {state.stopped_at}")
    print(f"  Mode              : {rt.decision.label}  ({rt.decision.endpoint})")
    print(f"  KRX session       : {rt.calendar.phase().value}")
    print(f"  Last equity       : {state.last_equity:>16,.0f} KRW")
    print(f"  Peak equity       : {state.peak_equity:>16,.0f} KRW")
    print(
        f"  Drawdown          : {rt.portfolio.drawdown_pct():.2%} "
        f"(halt at {rt.risk.max_drawdown_pct:.0%})"
    )
    print(
        f"  Risk per trade    : {rt.risk.risk_pct:.2%} "
        f"= {rt.risk.max_loss_per_trade(state.last_equity):,.0f} KRW"
    )
    print(f"  Portfolio         : {capacity.describe()}")

    positions = rt.portfolio.positions()
    print(f"\n  Open positions    : {len(positions)}")
    for code, pos in positions.items():
        label = f"{code} {rt.name_of(code)}".strip()
        print(
            f"    {label:<22} {pos.side:<5} qty={int(pos.qty):<8} "
            f"entry={pos.entry_price:>10,.0f} stop={pos.effective_stop():>10,.0f}"
        )

    rows = rt.portfolio.trades.read_all()
    print(f"\n  Recorded trades   : {len(rows)}  ({rt.portfolio.trades.path})")
    for row in rows[-5:]:
        print(
            f"    {row.get('timestamp','')} {row.get('symbol',''):<9} "
            f"{row.get('side',''):<5} pnl={row.get('pnl','')}"
        )
    print("=" * 78)
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    rt = build_runtime(args, cli_live=False, force_dry_run=not args.close_all)
    rt.portfolio.stop("manual stop requested by operator")
    print(f"\nBot state set to STOPPED ({rt.portfolio.store.path}).")
    if args.close_all:
        try:
            cancelled = rt.broker.cancel_all_orders()
            closed = rt.broker.close_all_positions()
            print(f"Cancelled {cancelled} order(s), closed {closed} position(s).")
        except BrokerError as exc:
            print(f"Broker teardown failed: {exc}")
            return 1
        for code in list(rt.portfolio.state.positions):
            rt.portfolio.state.positions.pop(code, None)
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
    print(f"Drawdown peak re-anchored to {rt.portfolio.state.peak_equity:,.0f} KRW.")
    return 0


def cmd_diagnose_orderbook(args: argparse.Namespace) -> int:
    """실시간 호가/체결강도 필드명 확인용 진단 -- 주문은 절대 못 나간다.

    force_dry_run=True로 build_runtime을 호출하므로 ReadOnlyBroker가 실제
    KiwoomBroker를 감싼다 (읽기는 전부 통과, 쓰기는 전부 거부) -- cmd_status
    와 동일한 안전장치. stock_info(ka10001) 응답을 파싱하지 않고 원본
    그대로 찍어서, 매수/매도 호가·잔량 필드가 실제로 어떤 키 이름으로
    오는지 확인한다. 이게 확정돼야 실시간 호가 불균형 필터를 안전하게
    연결할 수 있다.
    """
    cli_live = bool(getattr(args, "live", False))
    rt = build_runtime(args, cli_live=cli_live, force_dry_run=True)
    code = str(args.code)

    print()
    print("=" * 78)
    print("  ORDERBOOK FIELD DIAGNOSTIC -- READ ONLY, NO ORDERS CAN BE SENT".center(78))
    print("=" * 78)
    print(f"  Mode : {rt.decision.label}")
    print(f"  Code : {code}")
    print()

    broker = rt.broker
    inner = getattr(broker, "inner", broker)
    if not hasattr(inner, "_call"):
        print("  이 브로커는 원시 TR 호출을 지원하지 않습니다 (인증 정보나")
        print("  네트워크가 없어 dry-run/synthetic 브로커로 대체된 상태입니다).")
        return 1

    try:
        data = inner._call("stock_info", "stock_info", {"stk_cd": code}, f"diagnose({code})")
    except Exception as exc:
        print(f"  stock_info 조회 실패: {exc}")
        data = None

    if data is not None:
        print("  [1] stock_info(ka10001) 응답 필드 -- 참고용, 호가는 없음:")
        for key, value in data.items():
            print(f"    {key:<24} = {value!r}")
        print()

    try:
        ob = inner._call("quote", "orderbook", {"stk_cd": code}, f"diagnose_orderbook({code})")
    except Exception as exc:
        print(f"  quote(ka10004) 조회 실패: {exc}")
        print("  -- endpoint(/api/dostk/mrkcond)나 api-id(ka10004)가 아직 안 맞을 수")
        print("     있습니다. 이 에러 메시지를 그대로 Claude에게 붙여넣어 주세요.")
        return 1

    print("  [2] quote(ka10004, 호가) 응답 필드:")
    for key, value in ob.items():
        print(f"    {key:<24} = {value!r}")
    print()

    try:
        strength = inner._call(
            "quote", "strength_hourly", {"stk_cd": code}, f"diagnose_strength({code})"
        )
    except Exception as exc:
        print(f"  strength_hourly(ka10046) 조회 실패: {exc}")
        print("  -- 이 에러 메시지를 그대로 Claude에게 붙여넣어 주세요.")
    else:
        print("  [3] strength_hourly(ka10046, 체결강도) 응답 필드:")
        for key, value in strength.items():
            if isinstance(value, list):
                print(f"    {key} (리스트, {len(value)}건) -- 최신 3건만 표시:")
                for i, row in enumerate(value[:3]):
                    if isinstance(row, dict):
                        print(f"      -- row {i} --")
                        for rk, rv in row.items():
                            print(f"      {rk:<24} = {rv!r}")
                    else:
                        print(f"      row {i} = {row!r}")
            else:
                print(f"    {key:<24} = {value!r}")
        print()

    def _diagnose_ranking(endpoint_key: str, api_key: str, body: dict, list_key: str, label: str) -> None:
        try:
            data = inner._call(endpoint_key, api_key, body, f"diagnose_{api_key}")
        except Exception as exc:
            print(f"  {label} 조회 실패: {exc}")
            print("  -- 이 에러 메시지를 그대로 Claude에게 붙여넣어 주세요.")
            return
        print(f"  {label} 응답 필드:")
        rows = data.get(list_key)
        if rows is None:
            print(f"    (키 '{list_key}'를 못 찾음 -- 전체 최상위 키: {list(data.keys())})")
        elif isinstance(rows, list):
            print(f"    {list_key} (리스트, {len(rows)}건) -- 최신 3건만 표시:")
            for i, row in enumerate(rows[:3]):
                if isinstance(row, dict):
                    print(f"      -- row {i} --")
                    for rk, rv in row.items():
                        print(f"      {rk:<24} = {rv!r}")
                else:
                    print(f"      row {i} = {row!r}")
        for key, value in data.items():
            if key != list_key:
                print(f"    {key:<24} = {value!r}")
        print()

    _diagnose_ranking(
        "ranking", "volume_surge",
        {"mrkt_tp": "000", "sort_tp": "1", "tm_tp": "2", "trde_qty_tp": "5",
         "tm": "", "stk_cnd": "1", "pric_tp": "0", "stex_tp": "3"},
        "trde_qty_sdnin", "[4] volume_surge(ka10023, 거래량급증)",
    )
    _diagnose_ranking(
        "stock_info", "vi_triggered",
        {"mrkt_tp": "000", "bf_mkrt_tp": "1", "stk_cd": "", "motn_tp": "0",
         "skip_stk": "000000000", "trde_qty_tp": "0", "min_trde_qty": "",
         "max_trde_qty": "", "trde_prica_tp": "0", "min_trde_prica": "",
         "max_trde_prica": "", "motn_drc": "1", "stex_tp": "3"},
        "motn_stk", "[5] vi_triggered(ka10054, VI발동종목)",
    )

    print("  위 [2]~[5] 목록을 통째로 복사해서 Claude에게 붙여넣어 주세요.")
    print("  매수호가/매도호가/매수잔량/매도잔량, 체결강도, 거래량급증, VI발동")
    print("  비슷한 필드를 찾아서 실시간 필터를 마저 연결하겠습니다.")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py", description="KRX 8-stock trading bot (paper by default)"
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    prof = sub.add_parser("profile", help="measure the universe and suggest strategies")
    prof.add_argument("--months", type=int, default=6, help="months of history (default: 6)")
    prof.add_argument(
        "--live",
        action="store_true",
        help="read intraday charts from the live account (read-only, no orders)",
    )
    prof.add_argument(
        "--atr-sweep",
        action="store_true",
        help="measure ATR at 3/5/10/15/30/60-minute bars for the day-trade stocks",
    )
    prof.add_argument(
        "--refresh-calendar",
        action="store_true",
        help="derive the KRX holiday list from pykrx and print it for config.yaml",
    )

    study = sub.add_parser(
        "study-bounce",
        help="measure whether a multi-day drop is followed by a bounce",
    )
    study.add_argument("--months", type=int, default=24, help="history to scan (default: 24)")
    study.add_argument(
        "--down-days", type=int, default=2, help="consecutive losing sessions (default: 2)"
    )
    study.add_argument(
        "--drop", type=float, default=15.0,
        help="minimum combined fall over those days, in percent (default: 15)",
    )
    study.add_argument(
        "--limit-down",
        action="store_true",
        help="require every day to close at the -30%% limit instead of --drop",
    )
    study.add_argument(
        "--codes",
        default="",
        help="test these codes instead of the configured universe, e.g. "
             "096770,058470 (an out-of-sample check: same rule, different stocks)",
    )
    study.add_argument(
        "--pattern",
        choices=("drop", "strong-close", "volume-spike", "ma-cross", "new-high"),
        default="drop",
        help="which rule to measure. drop: N down sessions then a bounce. "
             "strong-close: the configured close_auction entry. volume-spike, "
             "ma-cross, new-high: common ideas, same standard",
    )
    study.add_argument(
        "--rank",
        type=int,
        metavar="N",
        help="rank stocks on the first half of the history and report how the "
             "top N did on the second -- the only honest way to ask which "
             "stocks are worth selecting",
    )
    study.add_argument(
        "--min-events",
        type=int,
        default=5,
        help="events a stock needs in the selection window to be ranked (default: 5)",
    )
    study.add_argument(
        "--market",
        choices=("kospi", "kosdaq", "all"),
        help="scan a whole board instead of a code list (slow, but it is the "
             "only way to get enough independent dates to settle anything)",
    )
    study.add_argument(
        "--limit", type=int, default=300, help="cap on --market stocks (default: 300)"
    )
    study.add_argument(
        "--live", action="store_true", help="read bars through the live account"
    )

    bt = sub.add_parser("backtest", help="run a historical backtest")
    bt.add_argument("--months", type=int, default=6, help="months of history (default: 6)")

    paper = sub.add_parser("paper", help="trade the mock account")
    paper.add_argument("--dry-run", action="store_true", help="print orders, send nothing")
    paper.add_argument("--once", action="store_true", help="run a single cycle and exit")
    paper.add_argument(
        "--watch",
        action="store_true",
        help="keep looping; with --dry-run this observes all day without ordering",
    )
    paper.add_argument(
        "--live", action="store_true", help="request live trading (still needs both env vars)"
    )

    live = sub.add_parser("live", help="trade the live account (requires triple confirmation)")
    live.add_argument("--once", action="store_true", help="run a single cycle and exit")
    live.add_argument("--dry-run", action="store_true", help="print orders, send nothing")
    live.add_argument(
        "--watch",
        action="store_true",
        help="keep looping; with --dry-run this observes all day without ordering",
    )
    live.add_argument(
        "--live", action="store_true", help="explicit live flag (the sub-command implies it)"
    )

    setup = sub.add_parser(
        "setup", help="write the app key and secret into .env (no editor needed)"
    )
    setup.add_argument(
        "--live", action="store_true", help="write the LIVE key pair instead of PAPER"
    )
    setup.add_argument(
        "--enable-live",
        action="store_true",
        help="write the two live confirmations into .env (requires typing the phrase)",
    )

    adopt = sub.add_parser(
        "adopt", help="bring existing broker holdings under the bot's management"
    )
    adopt.add_argument("--live", action="store_true", help="read the live account")
    adopt.add_argument("--yes", action="store_true", help="skip the confirmation prompt")

    chk = sub.add_parser(
        "check", help="read-only connectivity probe against the broker (no orders)"
    )
    chk.add_argument(
        "--live", action="store_true", help="probe the live account (still needs both env vars)"
    )

    sub.add_parser("status", help="show state, positions and drawdown")

    stop = sub.add_parser("stop", help="halt trading and persist STOPPED")
    stop.add_argument(
        "--close-all", action="store_true", help="also cancel orders and flatten positions"
    )

    sub.add_parser("resume", help="clear STOPPED and allow trading again")

    diag = sub.add_parser(
        "diagnose-orderbook",
        help="read-only: print raw bid/ask + 체결강도 fields to find key names",
    )
    diag.add_argument("code", help="6-digit stock code, e.g. 005930")
    diag.add_argument(
        "--live", action="store_true", help="query the live account (still read-only)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    make_console_tolerant()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "profile":
        return cmd_profile(args)
    if args.command == "study-bounce":
        return cmd_study_bounce(args)
    if args.command == "backtest":
        return cmd_backtest(args)
    if args.command == "paper":
        # `paper --live` is still an explicit live request; the env vars decide.
        return cmd_trade(args, cli_live=bool(args.live))
    if args.command == "live":
        # The `live` sub-command is itself the explicit command-line request.
        return cmd_trade(args, cli_live=True)
    if args.command == "setup":
        return cmd_setup(args)
    if args.command == "adopt":
        return cmd_adopt(args)
    if args.command == "check":
        return cmd_check(args)
    if args.command == "status":
        return cmd_status(args)
    if args.command == "stop":
        return cmd_stop(args)
    if args.command == "resume":
        return cmd_resume(args)
    if args.command == "diagnose-orderbook":
        return cmd_diagnose_orderbook(args)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
