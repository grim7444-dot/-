"""Investor flow scanner — foreign and institutional net buying.

Fetches the previous N days of foreign (외국인) and institutional (기관)
net purchase data from pykrx.  The result is a per-ticker FlowSignal that
the trading engine uses as an additional entry filter.

Design notes
------------
* pykrx data is T+1: what we get this morning is yesterday's flow.  That is
  still directionally useful for a day-trade bias.
* The scanner runs once per session (alongside the NXT pre-market scan) and
  caches results for the rest of the day.
* When pykrx is unavailable the scanner logs a warning and returns NEUTRAL
  for every ticker so the bot carries on with its normal logic — this module
  never blocks the bot from running.
* The filter is configurable: ``require_smart_money`` makes both foreign AND
  institutional net-buying a hard entry condition.  When false the signal is
  advisory only and just logged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger("bot.investor_flow")


class FlowBias(str, Enum):
    BULLISH = "bullish"    # both foreign AND institutional net-buying
    MIXED   = "mixed"      # only one of the two is a net buyer
    BEARISH = "bearish"    # both net-selling
    NEUTRAL = "neutral"    # data unavailable


@dataclass
class FlowSignal:
    ticker: str
    bias: FlowBias
    foreign_net: float   # KRW net over the lookback window (positive = buying)
    institution_net: float
    lookback_days: int

    def allows_entry(self, require_smart_money: bool) -> tuple[bool, str]:
        if self.bias == FlowBias.NEUTRAL:
            return True, "flow data n/a — not blocking"
        if not require_smart_money:
            return True, self._describe()
        if self.bias == FlowBias.BEARISH:
            return False, f"외국인+기관 동반 매도 ({self._describe()}) — entry blocked"
        return True, self._describe()

    def _describe(self) -> str:
        f = self.foreign_net / 1_000_000
        i = self.institution_net / 1_000_000
        sign_f = "+" if f >= 0 else ""
        sign_i = "+" if i >= 0 else ""
        return (
            f"외국인 {sign_f}{f:,.0f}M  기관 {sign_i}{i:,.0f}M  "
            f"({self.lookback_days}일) [{self.bias.value}]"
        )


def _fetch_flow(ticker: str, fromdate: str, todate: str) -> tuple[float, float] | None:
    """Return (foreign_net_krw, institution_net_krw) or None on failure."""
    try:
        from pykrx import stock as krx
        # get_market_net_purchases_of_investors_by_date returns a DataFrame
        # with investor types as columns (외국인, 기관합계, 개인, ...).
        df = krx.get_market_net_purchases_of_investors_by_date(
            fromdate, todate, ticker
        )
        if df is None or df.empty:
            return None
        foreign_col = next(
            (c for c in df.columns if "외국인" in str(c) or "외국" in str(c)), None
        )
        institution_col = next(
            (c for c in df.columns if "기관" in str(c)), None
        )
        if foreign_col is None or institution_col is None:
            logger.debug("investor_flow: unexpected columns %s for %s", list(df.columns), ticker)
            return None
        foreign_net = float(df[foreign_col].sum())
        institution_net = float(df[institution_col].sum())
        return foreign_net, institution_net
    except Exception as exc:
        logger.debug("investor_flow fetch failed %s: %s", ticker, exc)
        return None


class InvestorFlowScanner:
    """Session-scoped scanner for foreign and institutional buying bias."""

    def __init__(self, config: dict[str, Any]) -> None:
        cfg = config.get("investor_flow") or {}
        self.enabled: bool = bool(cfg.get("enabled", False))
        self.lookback_days: int = int(cfg.get("lookback_days", 5))
        self.require_smart_money: bool = bool(cfg.get("require_smart_money", False))
        self._cache: dict[str, FlowSignal] = {}
        self._scan_date: date | None = None

    def scan(self, tickers: list[str]) -> dict[str, FlowSignal]:
        """Fetch flow for *tickers* and cache results for today.

        Safe to call even when pykrx is down — returns NEUTRAL signals.
        """
        if not self.enabled:
            return {t: FlowSignal(t, FlowBias.NEUTRAL, 0.0, 0.0, 0) for t in tickers}

        today = date.today()
        if self._scan_date == today and all(t in self._cache for t in tickers):
            return {t: self._cache[t] for t in tickers}

        self._scan_date = today
        todate = (today - timedelta(days=1)).isoformat()
        fromdate = (today - timedelta(days=self.lookback_days + 5)).isoformat()

        results: dict[str, FlowSignal] = {}
        for ticker in tickers:
            if ticker in self._cache:
                results[ticker] = self._cache[ticker]
                continue
            flow = _fetch_flow(ticker, fromdate, todate)
            if flow is None:
                sig = FlowSignal(ticker, FlowBias.NEUTRAL, 0.0, 0.0, self.lookback_days)
            else:
                foreign_net, institution_net = flow
                if foreign_net > 0 and institution_net > 0:
                    bias = FlowBias.BULLISH
                elif foreign_net < 0 and institution_net < 0:
                    bias = FlowBias.BEARISH
                else:
                    bias = FlowBias.MIXED
                sig = FlowSignal(ticker, bias, foreign_net, institution_net, self.lookback_days)
                logger.info(
                    "investor_flow %s: %s",
                    ticker,
                    sig._describe(),
                )
            self._cache[ticker] = sig
            results[ticker] = sig

        return results

    def signal_for(self, ticker: str) -> FlowSignal:
        """Return cached signal, or NEUTRAL if not yet scanned."""
        return self._cache.get(
            ticker,
            FlowSignal(ticker, FlowBias.NEUTRAL, 0.0, 0.0, self.lookback_days),
        )
