"""Universe profiling.

``python main.py profile`` measures what each stock actually *is* - volatility,
liquidity, trend persistence, and how the names move together - and suggests a
strategy for each. That is more reliable than assigning strategies from a
company's name or sector, and it doubles as a verification pass: the official
listing name and market come back from the data source, so a wrong code or a
wrong KOSPI/KOSDAQ tag in ``config.yaml`` shows up immediately.

It also answers the question that decides whether a strategy can run at all:
does this stock have enough history to warm up its indicators? A stock listed
three months ago cannot support a 200-period lookback, and the profile says so
rather than letting the bot quietly emit HOLD forever.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

import numpy as np
import pandas as pd

from data import BarSet, MarketData, months_to_start, warmup_start
from indicators import atr as atr_indicator
from indicators import ema
from market.calendar import KST
from market.rules import KOSPI, KrxRules
from strategies.base import Strategy

logger = logging.getLogger("bot.profile")


@dataclass
class StockProfile:
    code: str
    configured_name: str = ""
    resolved_name: str | None = None
    configured_market: str = KOSPI
    resolved_market: str | None = None
    bars: int = 0
    source: str = ""
    last_close: float = 0.0
    atr_abs: float = 0.0
    atr_pct: float = 0.0
    #: Median (high - low) / close over the last 20 bars. ATR counts gaps and
    #: this does not, so a large split between them says the volatility lives
    #: in the overnight jump rather than in the session -- and a huge split
    #: says the bars are wrong.
    median_range_pct: float = 0.0
    first_bar: str = ""
    last_bar: str = ""
    daily_turnover: float = 0.0
    trend_strength: float = 0.0
    warmup_needed: int = 0
    tick: int = 0
    recent: pd.DataFrame | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def has_enough_history(self) -> bool:
        return self.bars >= self.warmup_needed

    @property
    def name_mismatch(self) -> bool:
        return bool(
            self.resolved_name
            and self.configured_name
            and self.resolved_name.strip() != self.configured_name.strip()
        )

    @property
    def market_mismatch(self) -> bool:
        return bool(self.resolved_market and self.resolved_market != self.configured_market)

    def suggested_strategy(self) -> str:
        """Pick a strategy from measured behaviour, not from the company name."""
        if not self.has_enough_history:
            return "breakout (short params - insufficient history)"
        if self.trend_strength >= 0.55:
            return "trend_following"
        if self.atr_pct >= 0.035:
            return "breakout"
        return "trend_following"


def _trend_strength(close: pd.Series, fast: int = 20, slow: int = 60) -> float:
    """Fraction of bars where the fast EMA sat above the slow EMA.

    Near 0.5 means the stock chops and a crossover system will whipsaw; far
    from 0.5 means it sustains direction long enough for a trend system to pay.
    """
    if len(close) < slow + 5:
        return 0.5
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    both = pd.concat([fast_ema, slow_ema], axis=1).dropna()
    if both.empty:
        return 0.5
    above = (both.iloc[:, 0] > both.iloc[:, 1]).mean()
    # Persistence in either direction counts, so fold around 0.5.
    return float(0.5 + abs(above - 0.5))


def profile_stock(
    code: str,
    barset: BarSet,
    asset_cfg: Mapping[str, Any],
    strategy: Strategy | None,
    rules: KrxRules,
    market_data: MarketData | None = None,
) -> StockProfile:
    bars = barset.bars
    prof = StockProfile(
        code=code,
        configured_name=str(asset_cfg.get("name") or ""),
        configured_market=str(asset_cfg.get("market") or KOSPI),
        bars=len(bars),
        source=barset.source,
        warmup_needed=strategy.warmup if strategy else 0,
    )

    if market_data is not None:
        prof.resolved_name = market_data.get_name(code)
        prof.resolved_market = market_data.get_market(code)

    if bars.empty:
        prof.notes.append("no bars available")
        return prof

    prof.last_close = float(bars["close"].iloc[-1])
    prof.tick = rules.tick_size(prof.last_close, prof.configured_market)

    prof.first_bar = str(bars.index[0])[:16]
    prof.last_bar = str(bars.index[-1])[:16]
    prof.recent = bars.tail(5)

    atr_series = atr_indicator(bars, 14).dropna()
    if not atr_series.empty and prof.last_close > 0:
        prof.atr_abs = float(atr_series.iloc[-1])
        prof.atr_pct = prof.atr_abs / prof.last_close

    spans = ((bars["high"] - bars["low"]) / bars["close"]).tail(20).dropna()
    if not spans.empty:
        prof.median_range_pct = float(spans.median())

    turnover = (bars["close"] * bars["volume"]).tail(60)
    prof.daily_turnover = float(turnover.mean()) if not turnover.empty else 0.0

    prof.trend_strength = _trend_strength(bars["close"])

    if barset.synthetic:
        prof.notes.append("SYNTHETIC bars - figures are illustrative only")
    # Position size is (equity x 1%) / ATR, so an ATR that is too large by a
    # factor of three buys a third of the shares it should -- and one that is
    # too small buys three times too many. It is the single number most worth
    # checking against the broker's own chart before trading.
    if prof.atr_pct >= 0.08 and not barset.synthetic:
        prof.notes.append(
            f"ATR is {prof.atr_pct:.1%} of price ({prof.atr_abs:,.0f} KRW) - high "
            f"for a daily bar. Compare against the same chart in the Kiwoom app "
            f"before trading; sizing divides by this number"
        )
    if not prof.has_enough_history:
        prof.notes.append(
            f"only {prof.bars} bars, strategy needs {prof.warmup_needed} to warm up"
        )
    if prof.name_mismatch:
        prof.notes.append(
            f"name mismatch: config says {prof.configured_name!r}, "
            f"listing says {prof.resolved_name!r}"
        )
    if prof.market_mismatch:
        prof.notes.append(
            f"market mismatch: config says {prof.configured_market}, "
            f"listing says {prof.resolved_market} - this changes the transaction tax"
        )
    return prof


def correlation_matrix(barsets: Mapping[str, BarSet]) -> pd.DataFrame:
    """Daily-return correlation across the universe.

    Two names above ~0.7 are effectively one position; the theme groups in
    ``config.yaml`` should reflect whatever this shows.
    """
    series: dict[str, pd.Series] = {}
    for code, barset in barsets.items():
        if barset.bars.empty:
            continue
        closes = barset.bars["close"]
        # Resample intraday series to daily so mixed timeframes are comparable.
        daily = closes.resample("1D").last().dropna()
        returns = daily.pct_change().dropna()
        if len(returns) >= 20:
            series[code] = returns
    if len(series) < 2:
        return pd.DataFrame()
    return pd.DataFrame(series).corr()


def format_profiles(
    profiles: list[StockProfile],
    correlations: pd.DataFrame,
    themes: Mapping[str, Any] | None = None,
) -> str:
    width = 100
    lines = ["=" * width, "UNIVERSE PROFILE".center(width), "=" * width, ""]

    header = (
        f"  {'code':<8} {'name':<14} {'mkt':<6} {'bars':>6} {'close':>11} "
        f"{'ATR':>9} {'ATR%':>7} {'range%':>7} {'turnover':>14} {'trend':>7}  suggested"
    )
    lines.append(header)
    lines.append("  " + "-" * (width - 4))
    for p in profiles:
        name = (p.resolved_name or p.configured_name or "")[:13]
        lines.append(
            f"  {p.code:<8} {name:<14} {(p.resolved_market or p.configured_market):<6} "
            f"{p.bars:>6d} {p.last_close:>11,.0f} {p.atr_abs:>9,.0f} "
            f"{p.atr_pct:>6.2%} {p.median_range_pct:>6.2%} "
            f"{p.daily_turnover:>14,.0f} {p.trend_strength:>7.2f}  {p.suggested_strategy()}"
        )

    flagged = [p for p in profiles if p.notes]
    if flagged:
        lines += ["", "  -- Notes " + "-" * (width - 14)]
        for p in flagged:
            for note in p.notes:
                lines.append(f"  {p.code}: {note}")

    # Everything above is derived. These are the numbers the derivation used,
    # printed so they can be read straight off the broker's own chart -- the
    # only way to tell a stock that really moves 16% a day from bars that are
    # wrong. Five rows per stock is short enough to actually check.
    with_bars = [p for p in profiles if p.recent is not None and not p.recent.empty]
    if with_bars:
        lines += ["", "  -- Last 5 bars (compare against the Kiwoom chart) "
                  + "-" * (width - 54)]
        for p in with_bars:
            name = (p.resolved_name or p.configured_name or "")[:13]
            lines.append(
                f"  {p.code} {name}   [{p.source}]   {p.first_bar} .. {p.last_bar}"
            )
            lines.append(
                f"      {'when':<17}{'open':>10}{'high':>10}{'low':>10}"
                f"{'close':>10}{'high-low':>11}"
            )
            for when, row in p.recent.iterrows():
                lines.append(
                    f"      {str(when)[:16]:<17}{row['open']:>10,.0f}"
                    f"{row['high']:>10,.0f}{row['low']:>10,.0f}"
                    f"{row['close']:>10,.0f}{row['high'] - row['low']:>11,.0f}"
                )

    if not correlations.empty:
        lines += ["", "  -- Return correlation " + "-" * (width - 26)]
        lines.append("  " + correlations.round(2).to_string().replace("\n", "\n  "))
        high = [
            (a, b, correlations.loc[a, b])
            for i, a in enumerate(correlations.index)
            for b in correlations.index[i + 1 :]
            if abs(correlations.loc[a, b]) >= 0.7
        ]
        if high:
            lines.append("")
            lines.append("  Pairs above 0.7 - consider grouping these under one theme:")
            for a, b, value in high:
                lines.append(f"    {a} / {b}: {value:+.2f}")

    if themes:
        lines += ["", "  -- Configured themes " + "-" * (width - 25)]
        for theme, members in themes.items():
            lines.append(f"    {theme:<20} {', '.join(members)}")

    lines += [
        "",
        "  Liquidity check: if 1% risk sizing asks for more shares than trade",
        "  comfortably at the touch, cut max_position_notional_pct in config.yaml.",
        "=" * width,
    ]
    return "\n".join(lines)


def run_profile(
    config: Mapping[str, Any],
    market_data: MarketData,
    strategies: Mapping[str, Strategy],
    months: int = 6,
) -> tuple[list[StockProfile], pd.DataFrame]:
    rules = KrxRules(config)
    universe = config.get("universe") or {}
    end = datetime.now(KST)
    start = months_to_start(months, end)

    barsets: dict[str, BarSet] = {}
    profiles: list[StockProfile] = []
    for code, asset_cfg in universe.items():
        code = str(code)
        if not asset_cfg.get("enabled", True):
            continue
        timeframe = asset_cfg.get("timeframe", "1Day")
        strategy = strategies.get(code)
        # Include the indicator run-up so "has enough history" reflects what
        # the bot would actually be able to compute.
        barset = market_data.get_bars(
            code=code,
            timeframe=timeframe,
            start=warmup_start(start, timeframe, strategy.warmup if strategy else 0),
            end=end,
            market=str(asset_cfg.get("market") or KOSPI),
        )
        barsets[code] = barset
        profiles.append(
            profile_stock(code, barset, asset_cfg, strategies.get(code), rules, market_data)
        )

    return profiles, correlation_matrix(barsets)
